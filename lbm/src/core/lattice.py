# Generic imports
import os
import math
import cupy                  as cp
import numpy                 as np
from   datetime              import datetime

# Custom imports
from   lbm.src.core.cupy     import *
from   lbm.src.core.obstacle import *
from   lbm.src.plot.plot     import *

### ************************************************
### Class defining lattice object
class lattice:
    ### ************************************************
    ### Constructor
    def __init__(self, app):

        # Set default values
        self.name      = 'lattice'
        self.x_min     = 0.0
        self.x_max     = 1.0
        self.y_min     = 0.0
        self.y_max     = 1.0
        self.nx        = 100
        self.ny        = 100
        self.batch_size = 1  # Add default batch size
        self.tau_lbm   = 1.0
        self.dx        = 1.0
        self.dt        = 1.0
        self.Cx        = self.dx
        self.Ct        = self.dt
        self.Cr        = 1.0
        self.Cn        = self.Cx**2/self.Ct
        self.Cu        = self.Cx/self.Ct
        self.Cf        = self.Cr*self.Cx**2/self.Ct
        self.dpi       = 100
        self.u_lbm     = 0.03
        self.L_lbm     = 100
        self.nu_lbm    = 0.01
        self.Re_lbm    = 100.0
        self.rho_lbm   = 1.0
        self.IBB       = False
        self.stop      = 'it'
        self.t_max     = 1.0
        self.it_max    = 1000
        self.obs_cv_ct = 1.0e-1
        self.obs_cv_nb = 500

        # Check if provided app has other parameters
        if hasattr(app, "name"):      self.name      = app.name
        if hasattr(app, "x_min"):     self.x_min     = app.x_min
        if hasattr(app, "x_max"):     self.x_max     = app.x_max
        if hasattr(app, "y_min"):     self.y_min     = app.y_min
        if hasattr(app, "y_max"):     self.y_max     = app.y_max
        if hasattr(app, "nx"):        self.nx        = app.nx
        if hasattr(app, "ny"):        self.ny        = app.ny
        if hasattr(app, "batch_size"): self.batch_size = app.batch_size
        if hasattr(app, "tau_lbm"):   self.tau_lbm   = app.tau_lbm
        if hasattr(app, "dx"):        self.dx        = app.dx
        if hasattr(app, "dt"):        self.dt        = app.dt
        if hasattr(app, "Cx"):        self.Cx        = app.Cx
        if hasattr(app, "Ct"):        self.Ct        = app.Ct
        if hasattr(app, "Cr"):        self.Cr        = app.Ct
        if hasattr(app, "Cn"):        self.Cn        = app.Cn
        if hasattr(app, "Cu"):        self.Cu        = app.Cu
        if hasattr(app, "Cf"):        self.Cf        = app.Cf
        if hasattr(app, "dpi"):       self.dpi       = app.dpi
        if hasattr(app, "u_lbm"):     self.u_lbm     = app.u_lbm
        if hasattr(app, "L_lbm"):     self.L_lbm     = app.L_lbm
        if hasattr(app, "nu_lbm"):    self.nu_lbm    = app.nu_lbm
        if hasattr(app, "Re_lbm"):    self.Re_lbm    = app.Re_lbm
        if hasattr(app, "rho_lbm"):   self.rho_lbm   = app.rho_lbm
        if hasattr(app, "IBB"):       self.IBB       = app.IBB
        if hasattr(app, "stop"):      self.stop      = app.stop
        if hasattr(app, "t_max"):     self.t_max     = app.t_max
        if hasattr(app, "it_max"):    self.it_max    = app.it_max
        if hasattr(app, "obs_cv_ct"): self.obs_cv_ct = app.obs_cv_ct
        if hasattr(app, "obs_cv_nb"): self.obs_cv_nb = app.obs_cv_nb

        # Output dirs
        time             = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        self.results_dir = './results/'
        self.output_dir  = self.results_dir+str(time)+'/'
        self.png_dir     = self.output_dir+'./png/'

        if (not os.path.exists(self.results_dir)):
            os.makedirs(self.results_dir, exist_ok=True)
        if (not os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir,  exist_ok=True)
        if (not os.path.exists(self.png_dir)):
            os.makedirs(self.png_dir,     exist_ok=True)

        # Default LBM parameters and fields
        self.set_default_lbm()

        # Printings
        print('##################')
        print('### LBM solver ###')
        print('##################')
        print('')
        print('# name       = '+self.name)
        print('# batch_size = '+str(self.batch_size))
        print('# u_lbm      = '+'{:f}'.format(self.u_lbm))
        print('# L_lbm      = '+'{:f}'.format(self.L_lbm))
        print('# nu_lbm     = '+'{:f}'.format(self.nu_lbm))
        print('# Re_lbm     = '+'{:f}'.format(self.Re_lbm))
        print('# tau_p_lbm  = '+'{:f}'.format(self.tau_p_lbm))
        print('# tau_m_lbm  = '+'{:f}'.format(self.tau_m_lbm))
        print('# dt         = '+'{:f}'.format(self.dt))
        print('# dx         = '+'{:f}'.format(self.dx))
        print('# nx         = '+str(self.nx))
        print('# ny         = '+str(self.ny))
        print('# IBB        = '+str(self.IBB))
        print('')

    ### ************************************************
    ### Set default LBM parameters
    def set_default_lbm(self):

        # Default LBM parameters
        self.output_it  = 0
        self.lx         = self.nx - 1
        self.ly         = self.ny - 1
        self.q          = 9
        self.Cs         = 1.0/math.sqrt(3.0)

        # TRT parameters
        self.tau_p_lbm  = self.tau_lbm
        self.lambda_trt = 1.0/4.0 # Best for stability
        self.tau_m_lbm  = self.lambda_trt/(self.tau_p_lbm - 0.5) + 0.5
        self.om_p_lbm   = 1.0/self.tau_p_lbm
        self.om_m_lbm   = 1.0/self.tau_m_lbm
        self.om_lbm     = 1.0/self.tau_lbm

        # D2Q9 Velocities
        self.c  = cp.array([ [ 0, 0],
                             [ 1, 0], [-1, 0],
                             [ 0, 1], [ 0,-1],
                             [ 1, 1], [-1,-1],
                             [-1, 1], [ 1,-1]])
        
        self.c_np = self.c.get()

        # Weights
        # Cardinal values, then extra-cardinal values, then central value
        idx_card       = [cp.linalg.norm(ci)<1.1 for ci in self.c]
        idx_extra_card = [cp.linalg.norm(ci)>1.1 for ci in self.c]

        self.w                             = cp.ones(self.q)
        self.w[cp.asarray(idx_card)]       = 1./9.
        self.w[cp.asarray(idx_extra_card)] = 1./36.
        self.w[0]                          = 4./9.

        # Array for bounce-back
        self.ns = cp.array([0,2,1,4,3,6,5,8,7])
        self.ns_np = self.ns.get()

        # Density arrays with batch dimension
        self.g       = cp.zeros((self.batch_size, self.q,  self.nx, self.ny), dtype=cp.float32)
        self.g_eq    = cp.zeros((self.batch_size, self.q,  self.nx, self.ny), dtype=cp.float32)
        self.g_up    = cp.zeros((self.batch_size, self.q,  self.nx, self.ny), dtype=cp.float32)

        # Boundary conditions with batch dimension
        self.u_left     = cp.zeros((self.batch_size, 2, self.ny), dtype=cp.float32)
        self.u_right    = cp.zeros((self.batch_size, 2, self.ny), dtype=cp.float32)
        self.u_top      = cp.zeros((self.batch_size, 2, self.nx), dtype=cp.float32)
        self.u_bot      = cp.zeros((self.batch_size, 2, self.nx), dtype=cp.float32)
        self.rho_right  = cp.zeros((self.batch_size,    self.ny), dtype=cp.float32)

        # Lattice array is oriented as follows :
        # +x     = left-right
        # +y     = bottom-top
        # origin = bottom left
        self.lattice = cp.zeros((self.batch_size, self.nx, self.ny), dtype=cp.int32)
        self.lattice_np = np.zeros((self.batch_size, self.nx, self.ny), dtype=np.int32)

        # Physical fields with batch dimension
        self.rho     = cp.ones ((self.batch_size,    self.nx, self.ny), dtype=cp.float32)
        self.u       = cp.zeros((self.batch_size, 2, self.nx, self.ny), dtype=cp.float32)
        self.t       = cp.zeros((self.batch_size, self.nx, self.ny), dtype=cp.float32)
        self.p       = cp.zeros((self.batch_size, self.nx, self.ny), dtype=cp.float32)
    ### ************************************************
    ### Compute macroscopic fields
    def macro(self):

        # Compute density
        self.rho[:,:,:] = cp.sum(self.g[:,:,:,:], axis=1)

        # Compute velocity
        self.u[:,0,:,:] = cp.tensordot(self.c[:,0],
                                       self.g[:,:,:,:],
                                       axes=(0,1))/self.rho[:,:,:]
        self.u[:,1,:,:] = cp.tensordot(self.c[:,1],
                                       self.g[:,:,:,:],
                                       axes=(0,1))/self.rho[:,:,:]

    ### ************************************************
    ### Compute equilibrium state
    def equilibrium(self):

        nb_equilibrium(self.u, self.c, self.w, self.rho, self.g_eq, self.batch_size)

    ### ************************************************
    ### Collision and streaming
    def collision_stream(self):

        nb_col_str(self.g, self.g_eq, self.g_up,
                   self.om_p_lbm, self.om_m_lbm,
                   self.c, self.ns,
                   self.batch_size, self.nx, self.ny,
                   self.lx, self.ly)

    ### ************************************************
    ### Compute drag and lift
    def drag_lift(self, obs, R_ref, U_ref, L_ref):
        Cx = 0.0
        Cy = 0.0

        # Cx, Cy = nb_drag_lift(obs.boundary, self.ns, self.c,
                            #   self.g_up, self.g, R_ref, U_ref, L_ref, self.batch_size)

        return Cx, Cy

    ### ************************************************
    ### Obstacle halfway bounce-back no-slip b.c.
    def bounce_back_obstacle(self, flattened_obstacles):
        nb_bounce_back_obstacle(self.IBB, flattened_obstacles[0], flattened_obstacles[1], flattened_obstacles[2],
                                self.ns, self.c,
                                self.g_up, self.g, self.u, self.lattice, self.batch_size)

    ### ************************************************
    ### Zou-He left wall velocity b.c.
    def zou_he_left_wall_velocity(self):

        nb_zou_he_left_wall_velocity(self.lx, self.ly, self.u,
                                     self.u_left, self.rho, self.g, self.batch_size)

    ### ************************************************
    ### Zou-He right wall pressure b.c.
    def zou_he_right_wall_pressure(self):

        nb_zou_he_right_wall_pressure(self.lx, self.ly, self.u,
                                      self.rho_right, self.u_right,
                                      self.rho, self.g, self.batch_size)

    ### ************************************************
    ### Zou-He no-slip top wall velocity b.c.
    def zou_he_top_wall_velocity(self):

        nb_zou_he_top_wall_velocity(self.lx, self.ly, self.u,
                                    self.u_top, self.rho, self.g, self.batch_size)

    ### ************************************************
    ### Zou-He no-slip bottom wall velocity b.c.
    def zou_he_bottom_wall_velocity(self):

        nb_zou_he_bottom_wall_velocity(self.lx, self.ly, self.u,
                                       self.u_bot, self.rho, self.g, self.batch_size)

    ### ************************************************
    ### Zou-He bottom left corner
    def zou_he_bottom_left_corner(self):

        nb_zou_he_bottom_left_corner_velocity(self.lx, self.ly, self.u,
                                              self.rho, self.g, self.batch_size)

    ### ************************************************
    ### Zou-He top left corner
    def zou_he_top_left_corner(self):

        nb_zou_he_top_left_corner_velocity(self.lx, self.ly, self.u,
                                           self.rho, self.g, self.batch_size)

    ### ************************************************
    ### Zou-He top right corner
    def zou_he_top_right_corner(self):

        nb_zou_he_top_right_corner_velocity(self.lx, self.ly, self.u,
                                            self.rho, self.g, self.batch_size)

    ### ************************************************
    ### Zou-He bottom right corner
    def zou_he_bottom_right_corner(self):

        nb_zou_he_bottom_right_corner_velocity(self.lx, self.ly, self.u,
                                               self.rho, self.g, self.batch_size)
    
    def zou_he_cuda(self):
        zou_he_cuda_kernel(self)

    ### ************************************************
    ### Add obstacle
    def add_obstacle(self, obstacle, batch):

        # Initial print
        tag = obstacle.tag
        print('# Obstacle ',str(tag))

        # Compute polygon bnds
        polygon      = obstacle.polygon # polygon is a numpy array
        poly_bnds    = np.zeros((4))
        poly_bnds[0] = np.amin(polygon[:,0])
        poly_bnds[1] = np.amax(polygon[:,0])
        poly_bnds[2] = np.amin(polygon[:,1])
        poly_bnds[3] = np.amax(polygon[:,1])

        # Declare lattice arrays
        obs = np.empty((0,2), dtype=int)
        bnd = np.empty((0,3), dtype=int)
        ibb = np.empty((0),   dtype=float)

        # Fill lattice
        for i in range(self.nx):
            for j in range(self.ny):
                pt = self.get_coords(i, j)

                # Check if pt is inside polygon bbox
                if ((pt[0] > poly_bnds[0]) and
                    (pt[0] < poly_bnds[1]) and
                    (pt[1] > poly_bnds[2]) and
                    (pt[1] < poly_bnds[3])):

                    if (self.is_inside(polygon, pt)):
                        self.lattice_np[batch, i,j] = tag
                        obs = np.append(obs, np.array([[i,j]]), axis=0)

        # Printings
        print('# '+str(obs.shape[0])+' locations in obstacle')

        # Build boundary of obstacle, i.e. 1st layer of fluid
        for k in range(len(obs)):
            i = obs[k,0]
            j = obs[k,1]

            for q in range(1,9):
                qb  = self.ns_np[q]
                cx  = self.c_np[q,0]
                cy  = self.c_np[q,1]
                ii  = i + cx
                jj  = j + cy

                if ((ii > self.nx-1) or (jj > self.ny-1)): continue
                if (not self.lattice_np[batch, ii,jj]):
                    bnd = np.append(bnd, np.array([[ii,jj,qb]]), axis=0)

        # Some cells were counted multiple times, unique-sort them
        bnd = np.unique(bnd, axis=0)

        # Printings
        print('# '+str(bnd.shape[0])+' locations on boundary')

        # Compute lattice-boundary distances if IBB is True
        if (self.IBB):
            for k in range(len(bnd)):
                i    = bnd[k,0]
                j    = bnd[k,1]
                q    = bnd[k,2]
                pt   = self.get_coords(i, j)
                x    = polygon[:,0] - pt[0]
                y    = polygon[:,1] - pt[1]
                dist = np.sqrt(np.square(x) + np.square(y))
                mpt  = np.argmin(dist)
                mdst = dist[mpt]/(self.dx*np.linalg.norm(self.c_np[q]))
                ibb  = np.append(ibb, mdst)

        # Check area of obstacle
        area = 0.0
        for i in range(self.nx):
            for j in range(self.ny):
                if (self.lattice_np[batch, i,j] == tag): area += self.dx**2

        # Printings
        print('# Area = '+'{:f}'.format(area))

        # Last print
        print('')

        self.lattice[batch, :, :] = cp.array(self.lattice_np[batch, :, :])
        return area, bnd, ibb

    ### ************************************************
    ### Get lattice coordinates from integers
    def get_coords(self, i, j):

        # Compute and return the coordinates of the lattice node (i,j)
        dx = (self.x_max - self.x_min)/(self.nx - 1)
        dy = (self.y_max - self.y_min)/(self.ny - 1)
        x  = self.x_min + i*dx
        y  = self.y_min + j*dy

        return [x, y]

    ### ************************************************
    ### Determine if a pt is inside or outside a closed polygon
    def is_inside(self, poly, pt):

        # Initialize
        j         = len(poly) - 1
        odd_nodes = False

        # Check if point is inside or outside
        # This is a valid algorithm for any non-convex polygon
        for i in range(len(poly)):
            if (((poly[i,1] < pt[1] and poly[j,1] >= pt[1])  or
                 (poly[j,1] < pt[1] and poly[i,1] >= pt[1])) and
                 (poly[i,0] < pt[0] or  poly[j,0]  < pt[0])):

                # Compute slope
                slope = (poly[j,0] - poly[i,0])/(poly[j,1] - poly[i,1])

                # Check side
                if ((poly[i,0] + (pt[1] - poly[i,1])*slope) < pt[0]):
                    odd_nodes = not odd_nodes

            # Increment
            j = i

        return odd_nodes