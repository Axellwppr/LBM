import cupy as cp
_zou_mod = cp.RawModule(code=r'''
#include <cuda_runtime.h>

extern "C" __global__
void zou_he_kernel(
    const int   B, const int Nx, const int Ny,
    const float* u_left_x,  const float* u_left_y,
    const float* u_right_y,            
    const float* u_top_x,   const float* u_top_y,
    const float* u_bot_x,   const float* u_bot_y,
    const float* rho_right,             
    float*       u,      /* shape (B,2,Nx,Ny), C-order */
    float*       rho,    /* shape (B,  Nx,Ny),   C-order */
    float*       g)      /* shape (B, 9,Nx,Ny), C-order */
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (x >= Nx || y >= Ny || b >= B) return;

    #define UIDX(c, xx, yy)  (((((b<<1) + (c)) * Nx + (xx)) * Ny + (yy)))
    #define GIDX(q, xx, yy)  (((((b * 9)  + (q)) * Nx + (xx)) * Ny + (yy)))
    #define RIDX(xx, yy)     (((b * Nx    + (xx))      * Ny + (yy)))

    float gq[9];
    #pragma unroll
    for (int q = 0; q < 9; ++q) {
        gq[q] = g[GIDX(q, x, y)];
    }

    const float cst1 = 2.0f / 3.0f;  // 2/3
    const float cst2 = 1.0f / 6.0f;  // 1/6
    const float cst3 = 1.0f / 2.0f;  // 1/2

    float ux, uy, rh;


    // --------------------------------------------
    if (x == 0 && y == 0) {
        ux = u_bot_x[b * Nx + 1];
        uy = u_bot_y[b * Nx + 1];
        u[UIDX(0, 0, 0)] = ux;
        u[UIDX(1, 0, 0)] = uy;

        rh = rho[RIDX(1, 0)];
        rho[RIDX(0, 0)] = rh;

        // g1 = g2 + (2/3)*rho*ux
        gq[1] = gq[2] + cst1 * rh * ux;
        // g3 = g4 + (2/3)*rho*uy
        gq[3] = gq[4] + cst1 * rh * uy;
        // g5 = g6 + (1/6)*rho*ux + (1/6)*rho*uy
        gq[5] = gq[6] + cst2 * rh * ux + cst2 * rh * uy;
        // g7 = 0, g8 = 0
        gq[7] = 0.0f;
        gq[8] = 0.0f;
        // g0 = rho - sum(q=1..8)
        {
            float sum = 0.0f;
            #pragma unroll
            for (int q = 1; q < 9; ++q) sum += gq[q];
            gq[0] = rh - sum;
        }
        #pragma unroll
        for (int q = 0; q < 9; ++q) {
            g[GIDX(q, 0, 0)] = gq[q];
        }
        return;
    }

    // --------------------------------------------
    if (x == 0 && y == Ny - 1) {
        ux = u_top_x[b * Nx + 1];
        uy = u_top_y[b * Nx + 1];
        u[UIDX(0, 0, Ny - 1)] = ux;
        u[UIDX(1, 0, Ny - 1)] = uy;

        rh = rho[RIDX(1, Ny - 1)];
        rho[RIDX(0, Ny - 1)] = rh;

        // g1 = g2 + (2/3)*rho*ux
        gq[1] = gq[2] + cst1 * rh * ux;
        // g4 = g3 - (2/3)*rho*uy
        gq[4] = gq[3] - cst1 * rh * uy;
        // g8 = g7 + (1/6)*rho*ux - (1/6)*rho*uy
        gq[8] = gq[7] + cst2 * rh * ux - cst2 * rh * uy;
        // g5 = 0, g6 = 0
        gq[5] = 0.0f;
        gq[6] = 0.0f;
        // g0 = rho - sum(q=1..8)
        {
            float sum = 0.0f;
            #pragma unroll
            for (int q = 1; q < 9; ++q) sum += gq[q];
            gq[0] = rh - sum;
        }
        #pragma unroll
        for (int q = 0; q < 9; ++q) {
            g[GIDX(q, 0, Ny - 1)] = gq[q];
        }
        return;
    }

    // --------------------------------------------
    // 3) top-right corner: (x==Nx-1, y==Ny-1)
    if (x == Nx - 1 && y == Ny - 1) {
        ux = u[UIDX(0, Nx - 2, Ny - 1)];
        uy = u[UIDX(1, Nx - 2, Ny - 1)];
        u[UIDX(0, Nx - 1, Ny - 1)] = ux;
        u[UIDX(1, Nx - 1, Ny - 1)] = uy;

        rh = rho[RIDX(Nx - 2, Ny - 1)];
        rho[RIDX(Nx - 1, Ny - 1)] = rh;

        // g2 = g1 - (2/3)*rho*ux
        gq[2] = gq[1] - cst1 * rh * ux;
        // g4 = g3 - (2/3)*rho*uy
        gq[4] = gq[3] - cst1 * rh * uy;
        // g6 = g5 - (1/6)*rho*ux - (1/6)*rho*uy
        gq[6] = gq[5] - cst2 * rh * ux - cst2 * rh * uy;
        // g7=0, g8=0
        gq[7] = 0.0f;
        gq[8] = 0.0f;
        // g0 = rho - sum(q=1..8)
        {
            float sum = 0.0f;
            #pragma unroll
            for (int q = 1; q < 9; ++q) sum += gq[q];
            gq[0] = rh - sum;
        }
        #pragma unroll
        for (int q = 0; q < 9; ++q) {
            g[GIDX(q, Nx - 1, Ny - 1)] = gq[q];
        }
        return;
    }

    // --------------------------------------------
    if (x == Nx - 1 && y == 0) {
        ux = u[UIDX(0, Nx - 2, 0)];
        uy = u[UIDX(1, Nx - 2, 0)];
        u[UIDX(0, Nx - 1, 0)] = ux;
        u[UIDX(1, Nx - 1, 0)] = uy;

        rh = rho[RIDX(Nx - 2, 0)];
        rho[RIDX(Nx - 1, 0)] = rh;

        // g2 = g1 - (2/3)*rho*ux
        gq[2] = gq[1] - cst1 * rh * ux;
        // g3 = g4 + (2/3)*rho*uy
        gq[3] = gq[4] + cst1 * rh * uy;
        // g7 = g8 - (1/6)*rho*ux + (1/6)*rho*uy
        gq[7] = gq[8] - cst2 * rh * ux + cst2 * rh * uy;
        // g5=0, g6=0
        gq[5] = 0.0f;
        gq[6] = 0.0f;
        // g0 = rho - sum(q=1..8)
        {
            float sum = 0.0f;
            #pragma unroll
            for (int q = 1; q < 9; ++q) sum += gq[q];
            gq[0] = rh - sum;
        }
        #pragma unroll
        for (int q = 0; q < 9; ++q) {
            g[GIDX(q, Nx - 1, 0)] = gq[q];
        }
        return;
    }

    // --------------------------------------------
    if (x == 0) {
        ux = u_left_x[b * Ny + y];
        uy = u_left_y[b * Ny + y];
        u[UIDX(0, 0, y)] = ux;
        u[UIDX(1, 0, y)] = uy;

        float g0 = gq[0], g2 = gq[2], g3 = gq[3], g4 = gq[4], g6 = gq[6], g7 = gq[7];
        // rho = (g0 + g3 + g4 + 2*(g2+g6+g7)) / (1 - ux)
        rh = (g0 + g3 + g4 + 2.0f * (g2 + g6 + g7)) / (1.0f - ux + 1e-8f);
        // g1 = g2 + (2/3)*rho*ux
        gq[1] = g2 + cst1 * rh * ux;
        // g5 = g6 - 1/2*(g3-g4) + 1/6*rho*ux + 1/2*rho*uy
        gq[5] = g6 - cst3 * (g3 - g4) + cst2 * rh * ux + cst3 * rh * uy;
        // g8 = g7 + 1/2*(g3-g4) + 1/6*rho*ux - 1/2*rho*uy
        gq[8] = g7 + cst3 * (g3 - g4) + cst2 * rh * ux - cst3 * rh * uy;

        rho[RIDX(0, y)] = rh;

        g[GIDX(1, 0, y)] = gq[1];
        g[GIDX(5, 0, y)] = gq[5];
        g[GIDX(8, 0, y)] = gq[8];
        // g0 = rho - sum(q=1..8)
        {
            float sum = gq[1] + g2 + g3 + g4 + gq[5] + g6 + g7 + gq[8];
            gq[0] = rh - sum;
        }
        g[GIDX(0, 0, y)] = gq[0];
        return;
    }

    // --------------------------------------------
    if (x == Nx - 1) {
        uy = u_right_y[b * Ny + y];
        rh = rho_right[b * Ny + y];
        u[UIDX(1, Nx - 1, y)] = uy;

        float g0 = gq[0], g1 = gq[1], g3 = gq[3], g4 = gq[4], g5 = gq[5], g8 = gq[8];
        // ux = (g0 + g3 + g4 + 2*(g1+g5+g8))/rho - 1
        ux = (g0 + g3 + g4 + 2.0f * (g1 + g5 + g8)) / rh - 1.0f;
        // g2 = g1 - (2/3)*rho*ux
        gq[2] = g1 - cst1 * rh * ux;
        // g6 = g5 + 1/2*(g3-g4) - 1/6*rho*ux - 1/2*rho*uy
        gq[6] = g5 + cst3 * (g3 - g4) - cst2 * rh * ux - cst3 * rh * uy;
        // g7 = g8 - 1/2*(g3-g4) - 1/6*rho*ux + 1/2*rho*uy
        gq[7] = g8 - cst3 * (g3 - g4) - cst2 * rh * ux + cst3 * rh * uy;

        u[UIDX(0, Nx - 1, y)] = ux;
        rho[RIDX(Nx - 1, y)] = rh;
        g[GIDX(2, Nx - 1, y)] = gq[2];
        g[GIDX(6, Nx - 1, y)] = gq[6];
        g[GIDX(7, Nx - 1, y)] = gq[7];
        // g0 = rho - sum(q=1..8)
        {
            float sum = g1 + gq[2] + g3 + g4 + g5 + gq[6] + gq[7] + g8;
            gq[0] = rh - sum;
        }
        g[GIDX(0, Nx - 1, y)] = gq[0];
        return;
    }

    // --------------------------------------------
    if (y == Ny - 1) {
        ux = u_top_x[b * Nx + x];
        uy = u_top_y[b * Nx + x];
        u[UIDX(0, x, Ny - 1)] = ux;
        u[UIDX(1, x, Ny - 1)] = uy;

        float g0 = gq[0], g1 = gq[1], g2 = gq[2], g3 = gq[3], g5 = gq[5], g7 = gq[7];
        // rho = (g0 + g1 + g2 + 2*(g3+g5+g7)) / (1 + uy)
        rh = (g0 + g1 + g2 + 2.0f * (g3 + g5 + g7)) / (1.0f + uy + 1e-8f);
        // g4 = g3 - (2/3)*rho*uy
        gq[4] = g3 - cst1 * rh * uy;
        // g8 = g7 - 1/2*(g1-g2) + 1/2*rho*ux - 1/6*rho*uy
        gq[8] = g7 - cst3 * (g1 - g2) + cst3 * rh * ux - cst2 * rh * uy;
        // g6 = g5 + 1/2*(g1-g2) - 1/2*rho*ux - 1/6*rho*uy
        gq[6] = g5 + cst3 * (g1 - g2) - cst3 * rh * ux - cst2 * rh * uy;

        rho[RIDX(x, Ny - 1)] = rh;
        g[GIDX(4, x, Ny - 1)] = gq[4];
        g[GIDX(8, x, Ny - 1)] = gq[8];
        g[GIDX(6, x, Ny - 1)] = gq[6];
        // g0 = rho - sum(q=1..8)
        {
            float sum = g1 + g2 + g3 + gq[4] + g5 + gq[6] + g7 + gq[8];
            gq[0] = rh - sum;
        }
        g[GIDX(0, x, Ny - 1)] = gq[0];
        return;
    }

    // --------------------------------------------
    if (y == 0) {
        ux = u_bot_x[b * Nx + x];
        uy = u_bot_y[b * Nx + x];
        u[UIDX(0, x, 0)] = ux;
        u[UIDX(1, x, 0)] = uy;

        float g0 = gq[0], g1 = gq[1], g2 = gq[2], g4 = gq[4], g6 = gq[6], g8 = gq[8];
        // rho = (g0 + g1 + g2 + 2*(g4+g6+g8)) / (1 - uy)
        rh = (g0 + g1 + g2 + 2.0f * (g4 + g6 + g8)) / (1.0f - uy + 1e-8f);
        // g3 = g4 + (2/3)*rho*uy
        gq[3] = g4 + cst1 * rh * uy;
        // g5 = g6 - 1/2*(g1-g2) + 1/2*rho*ux + 1/6*rho*uy
        gq[5] = g6 - cst3 * (g1 - g2) + cst3 * rh * ux + cst2 * rh * uy;
        // g7 = g8 + 1/2*(g1-g2) - 1/2*rho*ux + 1/6*rho*uy
        gq[7] = g8 + cst3 * (g1 - g2) - cst3 * rh * ux + cst2 * rh * uy;

        rho[RIDX(x, 0)] = rh;
        g[GIDX(3, x, 0)] = gq[3];
        g[GIDX(5, x, 0)] = gq[5];
        g[GIDX(7, x, 0)] = gq[7];
        // g0 = rho - sum(q=1..8)
        {
            float sum = g1 + g2 + gq[3] + g4 + g6 + gq[5] + gq[7] + g8;
            gq[0] = rh - sum;
        }
        g[GIDX(0, x, 0)] = gq[0];
        return;
    }

    // --------------------------------------------
    return;

    #undef UIDX
    #undef GIDX
    #undef RIDX
}

''', options=('--std=c++14',))
_zou_kernel = _zou_mod.get_function('zou_he_kernel')

# _zou_cache = {}
def _prepare_zou(latt):
    # key = id(latt)
    # if key in _zou_cache:
        # return _zou_cache[key]

    B  = latt.batch_size
    
    Nx = latt.lx + 1
    Ny = latt.ly + 1

    u_left_x  = latt.u_left[:, 0, :].ravel()
    u_left_y  = latt.u_left[:, 1, :].ravel()

    u_right_y = latt.u_right[:, 1, :].ravel()

    u_top_x   = latt.u_top[:,   0, :].ravel()
    u_top_y   = latt.u_top[:,   1, :].ravel()

    
    u_bot_x   = latt.u_bot[:,   0, :].ravel()
    u_bot_y   = latt.u_bot[:,   1, :].ravel()

    
    rho_right = latt.rho_right.ravel()

    buf = dict(
        B          = B,
        Nx         = Nx,
        Ny         = Ny,
        u_left_x   = u_left_x,
        u_left_y   = u_left_y,
        u_right_y  = u_right_y,
        u_top_x    = u_top_x,
        u_top_y    = u_top_y,
        u_bot_x    = u_bot_x,
        u_bot_y    = u_bot_y,
        rho_right  = rho_right,
    )
    # _zou_cache[key] = buf
    return buf

import cupy as cp

def _launch_zou_he(latt):
    # # ---------- structural sanity ----------
    # assert isinstance(latt.batch_size, int) and latt.batch_size > 0, \
    #     f"batch_size must be positive int, got {latt.batch_size}"
    B  = latt.batch_size
    Nx = latt.lx + 1
    Ny = latt.ly + 1

    # # ---------- helper to check array ----------
    # def _chk(arr, shape, name):
    #     assert isinstance(arr, cp.ndarray), f"{name} must be CuPy nd-array"
    #     assert arr.dtype == cp.float32,     f"{name} dtype must be float32"
    #     assert arr.device == latt.u.device, f"{name} on different GPU"
    #     assert arr.shape == shape,          f"{name} shape {arr.shape} != {shape}"
    #     return arr

    # # ---------- core & boundary fields ----------
    # _chk(latt.u,   (B, 2, Nx, Ny), "u")
    # _chk(latt.rho, (B,    Nx, Ny), "rho")
    # _chk(latt.g,   (B, 9, Nx, Ny), "g")

    # # ---------- velocity BC arrays ----------
    # _chk(latt.u_left,  (B, 2, Ny), "u_left")
    # _chk(latt.u_right, (B, 2, Ny), "u_right")
    # _chk(latt.u_top,   (B, 2, Nx), "u_top")
    # _chk(latt.u_bot,   (B, 2, Nx), "u_bot")

    # # ---------- pressure outlet ----------
    # _chk(latt.rho_right, (B, Ny), "rho_right")
    # assert cp.all(latt.rho_right > 0), "rho_right must be strictly positive"

    # # ---------- NaN / Inf early detection -------------
    # for name in ("u", "rho", "g"):
    #     arr = getattr(latt, name)
    #     assert cp.isfinite(arr).all(), f"{name} contains NaN or Inf"

    # ---------- geometry cache ----------
    buf = _prepare_zou(latt)   # will validate / create cached views

    # ---------- launch ----------
    B,Nx,Ny = buf['B'], buf['Nx'], buf['Ny']
    block = (16,16,1)
    grid  = ((Nx+15)//16, (Ny+15)//16, B)

    _zou_kernel(
        grid, block,
        (
            cp.int32(B), cp.int32(Nx), cp.int32(Ny),
            buf['u_left_x'],  buf['u_left_y'],
            buf['u_right_y'],
            buf['u_top_x'],   buf['u_top_y'],
            buf['u_bot_x'],   buf['u_bot_y'],
            buf['rho_right'],
            latt.u, latt.rho, latt.g
        ),
        stream=cp.cuda.get_current_stream()
    )
