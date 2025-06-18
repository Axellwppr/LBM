# Generic imports
import os
import time

# Custom imports
from lbm.src.core.lattice import *

########################
# Run lbm simulation
########################
def run_momentum(lattice, app):
    app.initialize(lattice)

    start_time = time.time()
    it         = 0
    compute    = True

    # Solve
    print('### Solving')
    while (compute):
        app.set_inlets(lattice, it)
        lattice.macro()
        app.outputs(lattice, it)
        lattice.equilibrium()
        lattice.collision_stream()
        app.set_bc(lattice)
        app.observables(lattice, it)
        compute = app.check_stop(it)
        it += 1

    # Count time
    end_time = time.time()
    print("# Loop time = {:f}".format(end_time - start_time))

import cupy as cp
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
def solve_pressure_poisson(lat, rho=998.0):
    bs, nx, ny  = lat.batch_size, lat.nx, lat.ny          # 批次、网格尺寸
    N           = nx * ny                                 # 单批未知数
    dx          = lat.dx                                  # Δx
    dy          = (lat.y_max - lat.y_min) / (ny - 1)      # Δy
    dx2         = dx * dx

    # ---------- 仅需一次的网格索引 / 邻接掩码 ------------------
    i = cp.arange(nx)[:, None]                            # (nx,1)
    j = cp.arange(ny)[None, :]                            # (1,ny)
    idx = (i * ny + j).astype(cp.int32)                   # (nx,ny) → [0,N)

    is_right        = (i == nx - 1)                       # (nx,1) 右边 Dirichlet=0
    # hasE, hasW      = (i < nx - 1), (i > 0)               # (nx,1)
    # hasN, hasS      = (j < ny - 1), (j > 0)               # (1,ny)

    for b in range(bs):                                   # ——— 逐 batch ———
        # —— 速度场，两障碍点置零避免假梯度 ——
        u  = lat.u[b, 0]                                  # (nx,ny)
        v  = lat.u[b, 1]
        solid = lat.lattice[b].astype(bool)               # (nx,ny)
        fluid = ~solid                               # NEW
        
        hasE = (i < nx-1) & fluid & cp.roll(fluid,-1,axis=0)   # NEW
        hasW = (i > 0   ) & fluid & cp.roll(fluid, 1,axis=0)   # NEW
        hasN = (j < ny-1) & fluid & cp.roll(fluid,-1,axis=1)   # NEW
        hasS = (j > 0   ) & fluid & cp.roll(fluid, 1,axis=1)   # NEW

        u = cp.where(solid, 0.0, u)
        v = cp.where(solid, 0.0, v)

        # —— 二阶中心差分计算速度梯度 (roll 完再除 2Δ) ——
        du_dx = (cp.roll(u, -1, axis=0) - cp.roll(u, 1, axis=0)) / (2*dx)
        du_dy = (cp.roll(u, -1, axis=1) - cp.roll(u, 1, axis=1)) / (2*dy)
        dv_dx = (cp.roll(v, -1, axis=0) - cp.roll(v, 1, axis=0)) / (2*dx)
        dv_dy = (cp.roll(v, -1, axis=1) - cp.roll(v, 1, axis=1)) / (2*dy)

        # —— 泊松源项 S_P —— ρ≈1
        S = -(du_dx**2 + 2*du_dy*dv_dx + dv_dy**2) * rho        # (nx,ny)

        # —— 边界条件掩码 —— （仅右侧为 Dirichlet，其余全 Neumann）
        dir_mask = cp.broadcast_to(is_right, (nx,ny)) | solid   # NEW
        unk_mask = fluid & ~dir_mask                            # NEW

        # —— 迎风 / Laplacian 系数（neighbour=1，中心= -n_nb）
        n_nb = (unk_mask & hasE).astype(cp.int8) + \
               (unk_mask & hasW).astype(cp.int8) + \
               (unk_mask & hasN).astype(cp.int8) + \
               (unk_mask & hasS).astype(cp.int8)

        a_P = -n_nb                                       # (nx,ny) 中心
        a_E = (unk_mask & hasE).astype(cp.int8)           # 1 / 0
        a_W = (unk_mask & hasW).astype(cp.int8)
        a_N = (unk_mask & hasN).astype(cp.int8)
        a_S = (unk_mask & hasS).astype(cp.int8)

        # —— 稀疏矩阵 A 及 RHS 组装 ——
        rows, cols, data = [], [], []
        rhs = cp.zeros(N, dtype=cp.float32)

        # ① Dirichlet（右侧 p=0）
        r_dir = idx[dir_mask].ravel()                     # (nd,)
        rows.append(r_dir);  cols.append(r_dir)
        data.append(cp.ones_like(r_dir, dtype=cp.float32))
        # RHS 默认 0，无需显式赋值

        # ② 未知节点中心
        r_unk = idx[unk_mask].ravel()                     # (nu,)
        rows.append(r_unk);  cols.append(r_unk)
        data.append(a_P[unk_mask].ravel())
        rhs[r_unk] = (S[unk_mask] * dx2).ravel()          # 右端 = S·Δx²

        # ③ 邻居项
        def add(mask, shift, coeff):
            if mask.any():
                rows.append(idx[mask].ravel())
                cols.append((idx[mask] + shift).ravel())
                data.append(coeff[mask].ravel())

        add(unk_mask & hasE,  ny,  a_E)
        add(unk_mask & hasW, -ny,  a_W)
        add(unk_mask & hasN,   1,  a_N)
        add(unk_mask & hasS,  -1,  a_S)

        # —— 求解 —— (CSR → GPU)
        A = sparse.coo_matrix((cp.concatenate(data),
                               (cp.concatenate(rows), cp.concatenate(cols))),
                              shape=(N, N)).tocsr()
        lat.p[b] = spsolve(A, rhs).reshape(nx, ny)

def calc_pressure(lattice):
    lattice.p[:] = (lattice.rho - 1.0) / 3.0
    lattice.p[:] -= lattice.p[:, -1:, :]