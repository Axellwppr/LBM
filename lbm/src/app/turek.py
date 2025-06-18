# Generic imports
import math
import numpy as np
import cupy as cp
from typing import List

# Custom imports
from lbm.src.core.lattice  import *
from lbm.src.core.obstacle import *
from lbm.src.utils.shapes  import *
from lbm.src.plot.plot     import *

def create_polar_obstacle(rs):
    obs = obstacle('turek', 0, 0, 'polar', 0, [0.0,0.0])
    shape = generate_polar_boundary(rs)
    obs.set_polygon(shape) # polygon is a numpy array
    return obs

def create_keypoints_obstacle(data):
    obs = obstacle('turek', 0, 0, 'keypoints', 0, [0.0,0.0])
    shape = generate_shape_from_keypoints(data)
    obs.set_polygon(shape) # polygon is a numpy array
    return obs

###############################################
### Turek benchmark
class turek():
    def __init__(self, datas: List[np.ndarray], shape_type: str = "polar"):

        # Free arguments
        self.name        = 'turek'
        self.Re_lbm      = 100.0
        self.L_lbm       = 151
        self.u_lbm       = 0.0025 * 3.0 / 2.0
        self.rho_lbm     = 1.0
        self.it_max      = 200
        self.x_min       =-0.004
        self.x_max       = 0.008
        self.y_min       =-0.002
        self.y_max       = 0.002
        self.IBB         = True
        self.stop        = 'it'
        self.obs_cv_ct   = 1.0e-3
        self.obs_cv_nb   = 1000

        # Output parameters
        self.output_freq = 500
        self.output_it   = 0
        self.dpi         = 200

        self.batch_size  = len(datas)

        # Deduce remaining lbm parameters
        self.compute_lbm_parameters()

        # Obstacle
        self.obstacles = []
        
        for data in datas:
            if shape_type == "polar":
                rs36 = interpolate_to_36(data)
                obs = create_polar_obstacle(rs36)
            elif shape_type == "keypoints":
                obs = create_keypoints_obstacle(data)
            self.obstacles.append(obs)
        
        self.flattened_obstacles = None

    ### Compute remaining lbm parameters
    def compute_lbm_parameters(self):

        self.Cs      = 1.0/math.sqrt(3.0)
        self.ny      = self.L_lbm
        self.u_avg   = 2.0*self.u_lbm/3.0
        self.L       = self.y_max - self.y_min
        self.D_lbm   = math.floor(self.ny*self.L/(self.y_max-self.y_min))
        self.nu_lbm  = self.u_avg*self.D_lbm/self.Re_lbm
        self.tau_lbm = 0.5 + self.nu_lbm/(self.Cs**2)
        self.dt      = self.Re_lbm*self.nu_lbm/self.D_lbm**2
        self.dx      = (self.y_max-self.y_min)/self.ny
        self.dy      = self.dx
        self.nx      = math.floor(self.ny*(self.x_max-self.x_min)/
                                  (self.y_max-self.y_min))
        self.sigma   = math.floor(10*self.nx)

    ### Add obstacles and initialize fields
    def initialize(self, lattice):

        # Add obstacles to lattice
        self.add_obstacles(lattice, self.obstacles)

        # Initialize fields
        self.calc_poiseuille(lattice)
        self.set_inlets(lattice, 0)
        lattice.u[:, :,cp.where(lattice.lattice > 0.0)] = 0.0
        lattice.rho *= self.rho_lbm

        # Compute first equilibrium
        lattice.equilibrium()
        lattice.g = lattice.g_eq.copy()
    
    def calc_poiseuille(self, lattice):
        poiseuille = np.zeros((self.batch_size, 2, self.ny))
        for j in range(self.ny):
            pt = lattice.get_coords(0, j)
            poiseuille[:, :, j] = self.u_lbm*self.poiseuille(pt).reshape(1, 2)
        self.poiseuille = cp.array(poiseuille)

    ### Set inlet fields
    def set_inlets(self, lattice, it):

        lx = lattice.lx
        ly = lattice.ly

        val  = it
        ret  = (1.0 - math.exp(-val**2/(2.0*self.sigma**2)))
        
        lattice.u_left[:,:,:] = self.poiseuille * ret

        # lattice.u_top[:, 0, :] = lattice.u[:, 0, :, -1]
        # lattice.u_bot[:, 0, :] = lattice.u[:, 0, :, 0]
        # lattice.u_right[:, 1, :] = lattice.u[:, 1, -1, :]

        lattice.u_top[:, 0,:]   = 0.0
        lattice.u_bot[:, 0,:]   = 0.0
        lattice.u_right[:,1,:] = 0.0
        lattice.rho_right[:,:] = self.rho_lbm

    ### Set boundary conditions
    def set_bc(self, lattice):

        # Obstacle
        lattice.bounce_back_obstacle(self.flattened_obstacles)

        lattice.zou_he_cuda()
        # Wall BCs
        # lattice.zou_he_bottom_wall_velocity()
        # lattice.zou_he_left_wall_velocity()
        # lattice.zou_he_top_wall_velocity()
        # lattice.zou_he_right_wall_pressure()
        # lattice.zou_he_bottom_left_corner()
        # lattice.zou_he_top_left_corner()
        # lattice.zou_he_top_right_corner()
        # lattice.zou_he_bottom_right_corner()

    ### Write outputs
    def outputs(self, lattice, it):
        if (it%self.output_freq != 0): return
        self.output_it += 1
        # if self.output_it > self.it_max or self.output_it == 1:
        if self.output_it > self.it_max or self.output_it % 50 == 1:
            plot_norm(lattice, 0.0, 1.5, self.output_it, self.dpi)
        print(self.output_it)

    ### Compute observables
    def observables(self, lattice, it):
        pass

    ### Poiseuille flow
    def poiseuille(self, pt):

        x    = pt[0]
        y    = pt[1]
        H    = self.y_max - self.y_min
        u    = np.zeros(2)
        u[0] = 4.0*(self.y_max-y)*(y-self.y_min)/H**2

        return u

    ### Add obstacles
    def add_obstacles(self, lattice, obstacles):
        for i in range(len(obstacles)):
            obs   = obstacles[i]
            obstacles[i].set_tag(i+1)
            area, bnd, ibb = lattice.add_obstacle(obstacles[i], batch=i)
            obstacles[i].fill(area, bnd, ibb)
        self.flattened_obstacles = self.flatten_obstacles(obstacles)
    
    def flatten_obstacles(self, obstacles):
        ibbs = []
        bnds = []
        batch = []
        
        for i in range(len(obstacles)):
            obs = obstacles[i]
            length = len(obs.ibb)
            ibbs.append(obs.ibb)
            bnds.append(obs.boundary)
            batch.append(cp.full(length, i))
        
        ibbs = np.concatenate(ibbs)
        bnds = np.concatenate(bnds)
        batch = np.concatenate(batch)
        
        ibbs = cp.array(ibbs)
        bnds = cp.array(bnds)
        batch = cp.array(batch)
        
        return bnds, ibbs, batch

    ### Check stopping criterion
    def check_stop(self, it):
        if self.output_it > self.it_max:
            return False
        return True