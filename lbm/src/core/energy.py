import cupy as cp
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve

def solve_temperature_convdiff_seq(lat, T_in=20.0, T_obs=50.0, alpha=0.143e-6):
    bs, nx, ny = lat.batch_size, lat.nx, lat.ny
    N          = nx * ny
    dx         = lat.dx
    dy         = (lat.y_max - lat.y_min) / (ny - 1)

    
    i = cp.arange(nx)[:, None]                 # (nx,1)
    j = cp.arange(ny)[None, :]                 # (1,ny)
    idx = (i * ny + j).astype(cp.int32)        # (nx,ny) → [0,N)

    is_left = (i == 0)
    is_left_full = cp.broadcast_to(is_left, (nx, ny))   # (nx,ny)
    hasE, hasW = (i < nx - 1), (i > 0)
    hasN, hasS = (j < ny - 1), (j > 0)

    for b in range(bs):
        u_x = lat.u[b, 0]                       # (nx,ny)
        u_y = lat.u[b, 1]
        is_obs  = lat.lattice[b].astype(bool)
        
        dir_mask     = is_left_full | is_obs
        
        dir_val = (T_in * is_left_full.astype(cp.float32)
                + T_obs * is_obs      .astype(cp.float32))
        unk_mask = ~dir_mask

        F_x = u_x * dy
        F_y = u_y * dx
        D_x = alpha * dy / dx
        D_y = alpha * dx / dy

        a_E = cp.where(unk_mask & hasE, D_x + cp.maximum(-F_x, 0), 0)
        a_W = cp.where(unk_mask & hasW, D_x + cp.maximum( F_x, 0), 0)
        a_N = cp.where(unk_mask & hasN, D_y + cp.maximum(-F_y, 0), 0)
        a_S = cp.where(unk_mask & hasS, D_y + cp.maximum( F_y, 0), 0)
        a_P = -(a_E + a_W + a_N + a_S)

        rows, cols, data = [], [], []
        rhs = cp.zeros(N, dtype=cp.float32)

        r_dir = idx[dir_mask].ravel()
        rows.append(r_dir); cols.append(r_dir)
        data.append(cp.ones_like(r_dir, dtype=cp.float32))
        rhs[r_dir] = dir_val[dir_mask].ravel()

        r_unk = idx[unk_mask].ravel()
        rows.append(r_unk); cols.append(r_unk)
        data.append(a_P[unk_mask].ravel())

        def add(mask, shift, coeff):
            if mask.any():
                rows.append(idx[mask].ravel())
                cols.append((idx[mask] + shift).ravel())
                data.append(coeff[mask].ravel())

        add(unk_mask & hasE,  ny,  a_E)
        add(unk_mask & hasW, -ny,  a_W)
        add(unk_mask & hasN,   1,  a_N)
        add(unk_mask & hasS,  -1,  a_S)

        data = cp.concatenate(data)
        # avoid nan
        if cp.isnan(data).any() or cp.isnan(rhs).any():
            continue

        A = sparse.coo_matrix((data,
                               (cp.concatenate(rows), cp.concatenate(cols))),
                              shape=(N, N)).tocsr()
        
        lat.t[b] = spsolve(A, rhs).reshape(nx, ny)