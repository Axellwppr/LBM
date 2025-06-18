import cupy as cp

### ************************************************
### Compute equilibrium state
def nb_equilibrium(u, c, w, rho, g_eq, batch_size):

    # Compute velocity term for all batches
    v = 1.5*(u[:,0,:,:]**2 + u[:,1,:,:]**2)

    # Compute equilibrium for all batches
    for q in range(9):
        t            = 3.0*(u[:,0,:,:]*c[q,0] + u[:,1,:,:]*c[q,1])
        g_eq[:,q,:,:]  = (1.0 + t + 0.5*t**2 - v[:,:,:])
        g_eq[:,q,:,:] *= rho[:,:,:]*w[q]

### ************************************************
### Collision and streaming
def nb_col_str(g, g_eq, g_up, om_p, om_m, c, ns, batch_size, nx, ny, lx, ly):

    # Take care of q=0 first for all batches
    g_up[:,0,:,:] = (1.0-om_p)*g[:,0,:,:] + om_p*g_eq[:,0,:,:]
    g   [:,0,:,:] = g_up[:,0,:,:]

    # Collide other indices for all batches
    for q in range(1,9):
        qb = ns[q]

        g_up[:,q,:,:] = ((1.0-0.5*(om_p+om_m))*g[:,q,:,:]   -
                            0.5*(om_p-om_m)*g[:,qb,:,:]   +
                            0.5*(om_p+om_m)*g_eq[:,q,:,:] +
                            0.5*(om_p-om_m)*g_eq[:,qb,:,:])

    # Stream for all batches
    g[:,1,1:nx, :  ] = g_up[:,1,0:lx, :  ]
    g[:,2,0:lx, :  ] = g_up[:,2,1:nx, :  ]
    g[:,3, :,  1:ny] = g_up[:,3, :,  0:ly]
    g[:,4, :,  0:ly] = g_up[:,4, :,  1:ny]
    g[:,5,1:nx,1:ny] = g_up[:,5,0:lx,0:ly]
    g[:,6,0:lx,0:ly] = g_up[:,6,1:nx,1:ny]
    g[:,7,0:lx,1:ny] = g_up[:,7,1:nx,0:ly]
    g[:,8,1:nx,0:ly] = g_up[:,8,0:lx,1:ny]

### ************************************************
### Obstacle halfway bounce-back no-slip b.c.
from lbm.src.core.cuda_ker.obs import nb_bounce_back_obstacle
from lbm.src.core.cuda_ker.bdy import _launch_zou_he as zou_he_cuda_kernel

def nb_bounce_back_obstacle_old(IBB, boundary, obs_ibb, batch, ns, sc, g_up, g, u, lattice, batch_size):
    if not IBB:
        raise ValueError("IBB must be True")

    i_gpu = boundary[:, 0]  # (N_boundary,)
    j_gpu = boundary[:, 1]  # (N_boundary,)
    q_gpu = boundary[:, 2]  # (N_boundary,)
    b_gpu = batch           # (N_boundary,)

    qb_gpu = ns[q_gpu]                       # (N_boundary,)

    cx_qb = sc[qb_gpu, 0]                    # (N_boundary,)
    cy_qb = sc[qb_gpu, 1]                    # (N_boundary,)

    im_gpu  = i_gpu + cx_qb                  # (N_boundary,)
    jm_gpu  = j_gpu + cy_qb                  # (N_boundary,)
    imm_gpu = i_gpu + (cx_qb * 2)            # (N_boundary,)
    jmm_gpu = j_gpu + (cy_qb * 2)            # (N_boundary,)

    # p, pp
    p_gpu  = obs_ibb                         # (N_boundary,)
    pp_gpu = p_gpu * 2.0                     # (N_boundary,)

    f0 = g_up[b_gpu, q_gpu,    i_gpu,  j_gpu]   # (N_boundary,)
    f1 = g_up[b_gpu, q_gpu,    im_gpu, jm_gpu]  # (N_boundary,)
    f2 = g_up[b_gpu, q_gpu,    imm_gpu, jmm_gpu]# (N_boundary,)
    f3 = g_up[b_gpu, qb_gpu,   i_gpu,  j_gpu]   # (N_boundary,)
    f4 = g_up[b_gpu, qb_gpu,   im_gpu, jm_gpu]  # (N_boundary,)

    out0 = p_gpu * (pp_gpu + 1.0) * f0 \
           + (1.0 + pp_gpu) * (1.0 - pp_gpu) * f1 \
           - p_gpu * (1.0 - pp_gpu) * f2

    out1 = (1.0 / (p_gpu * (pp_gpu + 1.0))) * f0 \
           + ((pp_gpu - 1.0) / p_gpu) * f3 \
           + ((1.0 - pp_gpu) / (1.0 + pp_gpu)) * f4

    mask = p_gpu < 0.5
    out_all = cp.empty_like(p_gpu)            # (N_boundary,)
    out_all[mask]      = out0[mask]
    out_all[~mask]     = out1[~mask]

    g[b_gpu, qb_gpu, i_gpu, j_gpu] = out_all

### ************************************************
### Zou-He left wall velocity b.c.
def nb_zou_he_left_wall_velocity(lx, ly, u, u_left, rho, g, batch_size):

    cst1 = 2.0/3.0
    cst2 = 1.0/6.0
    cst3 = 1.0/2.0

    u[:,0,0,:] = u_left[:,0,:]
    u[:,1,0,:] = u_left[:,1,:]

    rho[:,0,:] = (g[:,0,0,:] + g[:,3,0,:] + g[:,4,0,:] +
                2.0*g[:,2,0,:] + 2.0*g[:,6,0,:] +
                2.0*g[:,7,0,:] )/(1.0 - u[:,0,0,:])

    g[:,1,0,:] = (g[:,2,0,:] + cst1*rho[:,0,:]*u[:,0,0,:])

    g[:,5,0,:] = (g[:,6,0,:] - cst3*(g[:,3,0,:] - g[:,4,0,:]) +
                cst2*rho[:,0,:]*u[:,0,0,:] +
                cst3*rho[:,0,:]*u[:,1,0,:] )

    g[:,8,0,:] = (g[:,7,0,:] + cst3*(g[:,3,0,:] - g[:,4,0,:]) +
                cst2*rho[:,0,:]*u[:,0,0,:] -
                cst3*rho[:,0,:]*u[:,1,0,:] )

### ************************************************
### Zou-He right wall pressure b.c.
def nb_zou_he_right_wall_pressure(lx, ly, u, rho_right, u_right, rho, g, batch_size):

    cst1 = 2.0/3.0
    cst2 = 1.0/6.0
    cst3 = 1.0/2.0

    rho[:,lx,:] = rho_right[:,:]
    u[:,1,lx,:] = u_right[:,1,:]

    u[:,0,lx,:] = (g[:,0,lx,:] + g[:,3,lx,:] + g[:,4,lx,:] +
                    2.0*g[:,1,lx,:] + 2.0*g[:,5,lx,:] +
                    2.0*g[:,8,lx,:])/rho[:,lx,:] - 1.0

    g[:,2,lx,:] = (g[:,1,lx,:] - cst1*rho[:,lx,:]*u[:,0,lx,:])

    g[:,6,lx,:] = (g[:,5,lx,:] + cst3*(g[:,3,lx,:] - g[:,4,lx,:]) -
                    cst2*rho[:,lx,:]*u[:,0,lx,:] -
                    cst3*rho[:,lx,:]*u[:,1,lx,:] )

    g[:,7,lx,:] = (g[:,8,lx,:] - cst3*(g[:,3,lx,:] - g[:,4,lx,:]) -
                    cst2*rho[:,lx,:]*u[:,0,lx,:] +
                    cst3*rho[:,lx,:]*u[:,1,lx,:] )

### ************************************************
### Zou-He no-slip top wall velocity b.c.
def nb_zou_he_top_wall_velocity(lx, ly, u, u_top, rho, g, batch_size):

    cst1 = 2.0/3.0
    cst2 = 1.0/6.0
    cst3 = 1.0/2.0

    u[:,0,:,ly] = u_top[:,0,:]
    u[:,1,:,ly] = u_top[:,1,:]

    rho[:,:,ly] = (g[:,0,:,ly] + g[:,1,:,ly] + g[:,2,:,ly] +
                2.0*g[:,3,:,ly] + 2.0*g[:,5,:,ly] +
                2.0*g[:,7,:,ly])/(1.0 + u[:,1,:,ly])

    g[:,4,:,ly] = (g[:,3,:,ly] - cst1*rho[:,:,ly]*u[:,1,:,ly])

    g[:,8,:,ly] = (g[:,7,:,ly] - cst3*(g[:,1,:,ly] - g[:,2,:,ly]) +
                    cst3*rho[:,:,ly]*u[:,0,:,ly] -
                    cst2*rho[:,:,ly]*u[:,1,:,ly] )

    g[:,6,:,ly] = (g[:,5,:,ly] + cst3*(g[:,1,:,ly] - g[:,2,:,ly]) -
                    cst3*rho[:,:,ly]*u[:,0,:,ly] -
                    cst2*rho[:,:,ly]*u[:,1,:,ly] )

### ************************************************
### Zou-He no-slip bottom wall velocity b.c.
def nb_zou_he_bottom_wall_velocity(lx, ly, u, u_bot, rho, g, batch_size):

    cst1 = 2.0/3.0
    cst2 = 1.0/6.0
    cst3 = 1.0/2.0

    u[:,0,:,0] = u_bot[:,0,:]
    u[:,1,:,0] = u_bot[:,1,:]

    rho[:,:,0] = (g[:,0,:,0] + g[:,1,:,0] + g[:,2,:,0] +
                2.0*g[:,4,:,0] + 2.0*g[:,6,:,0] +
                2.0*g[:,8,:,0] )/(1.0 - u[:,1,:,0])

    g[:,3,:,0] = (g[:,4,:,0] + cst1*rho[:,:,0]*u[:,1,:,0])

    g[:,5,:,0] = (g[:,6,:,0] - cst3*(g[:,1,:,0] - g[:,2,:,0]) +
                cst3*rho[:,:,0]*u[:,0,:,0] +
                cst2*rho[:,:,0]*u[:,1,:,0] )

    g[:,7,:,0] = (g[:,8,:,0] + cst3*(g[:,1,:,0] - g[:,2,:,0]) -
                cst3*rho[:,:,0]*u[:,0,:,0] +
                cst2*rho[:,:,0]*u[:,1,:,0] )

### ************************************************
### Zou-He no-slip bottom left corner velocity b.c.
def nb_zou_he_bottom_left_corner_velocity(lx, ly, u, rho, g, batch_size):

    u[:,0,0,0] = u[:,0,1,0]
    u[:,1,0,0] = u[:,1,1,0]

    rho[:,0,0] = rho[:,1,0]

    g[:,1,0,0] = (g[:,2,0,0] + (2.0/3.0)*rho[:,0,0]*u[:,0,0,0])

    g[:,3,0,0] = (g[:,4,0,0] + (2.0/3.0)*rho[:,0,0]*u[:,1,0,0])

    g[:,5,0,0] = (g[:,6,0,0] + (1.0/6.0)*rho[:,0,0]*u[:,0,0,0]
                            + (1.0/6.0)*rho[:,0,0]*u[:,1,0,0] )

    g[:,7,0,0] = 0.0
    g[:,8,0,0] = 0.0

    g[:,0,0,0] = (rho[:,0,0]
                - g[:,1,0,0] - g[:,2,0,0] - g[:,3,0,0] - g[:,4,0,0]
                - g[:,5,0,0] - g[:,6,0,0] - g[:,7,0,0] - g[:,8,0,0] )

### ************************************************
### Zou-He no-slip top left corner velocity b.c.
def nb_zou_he_top_left_corner_velocity(lx, ly, u, rho, g, batch_size):

    u[:,0,0,ly] = u[:,0,1,ly]
    u[:,1,0,ly] = u[:,1,1,ly]

    rho[:,0,ly] = rho[:,1,ly]

    g[:,1,0,ly] = (g[:,2,0,ly] + (2.0/3.0)*rho[:,0,ly]*u[:,0,0,ly])

    g[:,4,0,ly] = (g[:,3,0,ly] - (2.0/3.0)*rho[:,0,ly]*u[:,1,0,ly])

    g[:,8,0,ly] = (g[:,7,0,ly] + (1.0/6.0)*rho[:,0,ly]*u[:,0,0,ly]
                            - (1.0/6.0)*rho[:,0,ly]*u[:,1,0,ly])


    g[:,5,0,ly] = 0.0
    g[:,6,0,ly] = 0.0

    g[:,0,0,ly] = (rho[:,0,ly]
                    - g[:,1,0,ly] - g[:,2,0,ly] - g[:,3,0,ly] - g[:,4,0,ly]
                    - g[:,5,0,ly] - g[:,6,0,ly] - g[:,7,0,ly] - g[:,8,0,ly] )

### ************************************************
### Zou-He no-slip top right corner velocity b.c.
def nb_zou_he_top_right_corner_velocity(lx, ly, u, rho, g, batch_size):

    u[:,0,lx,ly] = u[:,0,lx-1,ly]
    u[:,1,lx,ly] = u[:,1,lx-1,ly]

    rho[:,lx,ly] = rho[:,lx-1,ly]

    g[:,2,lx,ly] = (g[:,1,lx,ly] - (2.0/3.0)*rho[:,lx,ly]*u[:,0,lx,ly])

    g[:,4,lx,ly] = (g[:,3,lx,ly] - (2.0/3.0)*rho[:,lx,ly]*u[:,1,lx,ly])

    g[:,6,lx,ly] = (g[:,5,lx,ly] - (1.0/6.0)*rho[:,lx,ly]*u[:,0,lx,ly]
                                - (1.0/6.0)*rho[:,lx,ly]*u[:,1,lx,ly])

    g[:,7,lx,ly] = 0.0
    g[:,8,lx,ly] = 0.0

    g[:,0,lx,ly] = (rho[:,lx,ly]
                    - g[:,1,lx,ly] - g[:,2,lx,ly] - g[:,3,lx,ly] - g[:,4,lx,ly]
                    - g[:,5,lx,ly] - g[:,6,lx,ly] - g[:,7,lx,ly] - g[:,8,lx,ly] )

### ************************************************
### Zou-He no-slip bottom right corner velocity b.c.
def nb_zou_he_bottom_right_corner_velocity(lx, ly, u, rho, g, batch_size):

    u[:,0,lx,0] = u[:,0,lx-1,0]
    u[:,1,lx,0] = u[:,1,lx-1,0]

    rho[:,lx,0] = rho[:,lx-1,0]

    g[:,2,lx,0] = (g[:,1,lx,0] - (2.0/3.0)*rho[:,lx,0]*u[:,0,lx,0])

    g[:,3,lx,0] = (g[:,4,lx,0] + (2.0/3.0)*rho[:,lx,0]*u[:,1,lx,0])

    g[:,7,lx,0] = (g[:,8,lx,0] - (1.0/6.0)*rho[:,lx,0]*u[:,0,lx,0]
                            + (1.0/6.0)*rho[:,lx,0]*u[:,1,lx,0])

    g[:,5,lx,0] = 0.0
    g[:,6,lx,0] = 0.0

    g[:,0,lx,0] = (rho[:,lx,0]
                    - g[:,1,lx,0] - g[:,2,lx,0] - g[:,3,lx,0] - g[:,4,lx,0]
                    - g[:,5,lx,0] - g[:,6,lx,0] - g[:,7,lx,0] - g[:,8,lx,0] )
