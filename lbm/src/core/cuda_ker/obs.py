import cupy as cp

# -----------------------------------------------------------
_nb_bb_cache = {}
_nb_ibb_cache = {}

_cuda_src = r'''
extern "C" __global__
void bounce_back_kernel(
    const int   N,              // N_boundary
    const int   Q,
    const int   Nx,
    const int   Ny,
    const int  *idx0,
    const int  *idx1,
    const int  *idx2,
    const int  *idx3,
    const int  *idx4,
    const int  *idxO,
    const float *p,
    const float *g_up,
    float       *g_out)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= N) return;

    float f0 = __ldg(g_up + idx0[k]);
    float f1 = __ldg(g_up + idx1[k]);
    float f2 = __ldg(g_up + idx2[k]);
    float f3 = __ldg(g_up + idx3[k]);
    float f4 = __ldg(g_up + idx4[k]);

    float pv = p[k];
    float pp = pv * 2.0f;

    float out = (pv < 0.5f)
        ? pv*(pp+1.f)*f0 + (1.f+pp)*(1.f-pp)*f1 - pv*(1.f-pp)*f2
        : f0/(pv*(pp+1.f)) + ((pp-1.f)/pv)*f3 + ((1.f-pp)/(1.f+pp))*f4;

    g_out[idxO[k]] = out;
}
'''.strip()

_module_bb = cp.RawModule(code=_cuda_src, options=('--std=c++14',))
_kernel_bb = _module_bb.get_function('bounce_back_kernel')


def _lin_idx(b, q, i, j, Q, Nx, Ny):
    return (((b * Q + q) * Nx + i) * Ny + j).astype(cp.int32)


def _prep(boundary, batch, ns, sc, *, Q, Nx, Ny):
    bi = cp.ascontiguousarray(boundary[:, 0].astype(cp.int32))
    bj = cp.ascontiguousarray(boundary[:, 1].astype(cp.int32))
    bq = cp.ascontiguousarray(boundary[:, 2].astype(cp.int32))
    bb = cp.ascontiguousarray(batch.astype(cp.int32))

    ns_gpu = cp.asarray(ns, dtype=cp.int32)
    sc_x   = cp.asarray(sc[:, 0], dtype=cp.int32)
    sc_y   = cp.asarray(sc[:, 1], dtype=cp.int32)

    qb = ns_gpu[bq]
    cx = sc_x[qb]
    cy = sc_y[qb]

    im  = bi + cx
    jm  = bj + cy
    imm = bi + (cx << 1)
    jmm = bj + (cy << 1)

    return {
        'N'   : boundary.shape[0],
        'idx0': _lin_idx(bb, bq,  bi,  bj , Q, Nx, Ny),
        'idx1': _lin_idx(bb, bq,  im,  jm , Q, Nx, Ny),
        'idx2': _lin_idx(bb, bq, imm, jmm, Q, Nx, Ny),
        'idx3': _lin_idx(bb, qb,  bi,  bj , Q, Nx, Ny),
        'idx4': _lin_idx(bb, qb,  im,  jm , Q, Nx, Ny),
        'idxO': _lin_idx(bb, qb,  bi,  bj , Q, Nx, Ny),
    }


def nb_bounce_back_obstacle(IBB, boundary, obs_ibb, batch,
                            ns, sc, g_up, g, u, lattice, batch_size):
    global _nb_bb_cache, _nb_ibb_cache
    
    if not IBB:
        raise ValueError("IBB must be True")

    if g_up.dtype != cp.float32 or g.dtype != cp.float32:
        raise TypeError("g_up and g must be float32")
    if not g_up.flags.c_contiguous or not g.flags.c_contiguous:
        raise ValueError("g_up and g must be C-contiguous")

    key = (int(boundary.data.ptr), int(ns.data.ptr), int(sc.data.ptr),
           boundary.shape[0])
    Q        = ns.shape[0]
    Nx, Ny   = g_up.shape[2], g_up.shape[3]

    if key not in _nb_bb_cache:
        print("bounce_back_obstacle boundary cache miss")
        _nb_bb_cache = {}
        _nb_bb_cache[key] = _prep(boundary, batch, ns, sc,
                                  Q=Q, Nx=Nx, Ny=Ny)
    buf = _nb_bb_cache[key]

    if key not in _nb_ibb_cache:
        print("bounce_back_obstacle obs_ibb cache miss")
        _nb_ibb_cache = {}
        _nb_ibb_cache[key] = cp.asarray(obs_ibb, dtype=cp.float32).ravel()
    p_gpu = _nb_ibb_cache[key]

    N      = buf['N']
    block  = 256
    grid   = (N + block - 1) // block

    _kernel_bb(
        (grid,), (block,),
        (cp.int32(N), cp.int32(Q), cp.int32(Nx), cp.int32(Ny),
         buf['idx0'], buf['idx1'], buf['idx2'],
         buf['idx3'], buf['idx4'], buf['idxO'],
         p_gpu,
         g_up, g),
        stream=cp.cuda.get_current_stream())
