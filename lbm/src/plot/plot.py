# Generic imports
import os
import math
import numpy              as np
import matplotlib.pyplot  as plt

### ************************************************
### Output 2D flow amplitude
def plot_norm(lattice, val_min, val_max, output_it, dpi):
    
    # Loop through all batches
    num_batches = lattice.u.shape[0]
    for batch in range(num_batches):
        # Compute norm
        v = np.sqrt(lattice.u[batch, 0,:,:].get()**2+lattice.u[batch, 1,:,:].get()**2)

        # Mask obstacles
        v[np.where(lattice.lattice[batch, :,:].get() > 0.0)] = -1.0
        vm = np.ma.masked_where((v < 0.0), v)
        vm = np.rot90(vm)

        # Plot
        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(vm))
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(vm,
                   cmap = 'viridis',
                   vmin = val_min*lattice.u_lbm,
                   vmax = val_max*lattice.u_lbm,
                   interpolation = 'spline16')

        filename = lattice.png_dir+f'u_norm_{output_it}_batch{batch}.png'
        plt.axis('off')
        plt.savefig(filename, dpi=dpi)
        plt.close()
    
def plot_temperature(lattice, val_min=20.0, val_max=50.0, output_it=0, dpi=200):
    
    # Loop through all batches
    num_batches = lattice.t.shape[0]
    for batch in range(num_batches):
        # 取 numpy
        T_plot = lattice.t[batch].get()
        obs = lattice.lattice[batch].get()
        
        # 障碍掩膜
        # T_plot = np.where(obs > 0, -1.0, T_plot)
        T_masked = np.ma.masked_where(T_plot < 0, T_plot)
        T_masked = np.rot90(T_masked)  # 保持物理朝向一致

        # 物理比例尺
        aspect = (lattice.y_max - lattice.y_min) / (lattice.x_max - lattice.x_min)
        figsize = plt.figaspect(aspect)

        plt.clf()
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(0, 0, 1, 1)   # 无白边

        im = ax.imshow(T_masked,
                       cmap='viridis',
                       vmin=val_min,
                       vmax=val_max,
                       interpolation='spline16')

        filename = lattice.png_dir + f'temperature_{output_it}_batch{batch}.png'
        plt.axis('off')
        plt.savefig(filename, dpi=dpi)
        plt.close()

def plot_pressure(lattice, val_min=0.0, val_max=10.0, output_it=0, dpi=200):
    
    # Loop through all batches
    num_batches = lattice.p.shape[0]
    for batch in range(num_batches):
        P_plot = lattice.p[batch].get()
        val_min = P_plot.min()
        val_max = P_plot.max()
        obs = lattice.lattice[batch].get()
        P_masked = np.rot90(P_plot)
        
        aspect = (lattice.y_max - lattice.y_min) / (lattice.x_max - lattice.x_min)
        figsize = plt.figaspect(aspect)
        
        plt.clf()
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(0, 0, 1, 1)
        
        im = ax.imshow(P_masked,
                       cmap='viridis',
                       vmin=val_min,
                       vmax=val_max,
                       interpolation='spline16')
        
        filename = lattice.png_dir + f'pressure_{output_it}_batch{batch}.png'
        plt.axis('off')
        plt.savefig(filename, dpi=dpi)
        plt.close()