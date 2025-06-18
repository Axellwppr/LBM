# Generic imports
import os
import os.path
import PIL
import math
import scipy.special
import matplotlib
import numpy             as np
import matplotlib.pyplot as plt

### ************************************************
### Class defining shape object
class shape:
    ### ************************************************
    ### Constructor
    def __init__(self,
                 name,
                 position,
                 control_pts,
                 n_control_pts,
                 n_sampling_pts,
                 radius,
                 edgy,
                 output_dir):

        self.name           = name
        self.position       = position
        self.control_pts    = control_pts
        self.n_control_pts  = n_control_pts
        self.n_sampling_pts = n_sampling_pts
        self.curve_pts      = np.array([])
        self.area           = 0.0
        self.size_x         = 0.0
        self.size_y         = 0.0
        self.radius         = radius
        self.edgy           = edgy
        self.output_dir     = output_dir

        if (not os.path.exists(self.output_dir)): os.makedirs(self.output_dir)

    ### ************************************************
    ### Reset object
    def reset(self):

        # Reset object
        self.name           = 'shape'
        self.control_pts    = np.array([])
        self.n_control_pts  = 0
        self.n_sampling_pts = 0
        self.radius         = np.array([])
        self.edgy           = np.array([])
        self.curve_pts      = np.array([])
        self.area           = 0.0

    ### ************************************************
    ### Build shape
    def build(self):

        # Center set of points
        center = np.mean(self.control_pts, axis=0)
        self.control_pts -= center

        # Sort points counter-clockwise
        control_pts, radius, edgy  = ccw_sort(self.control_pts,
                                              self.radius,
                                              self.edgy)

        local_curves = []
        delta        = np.zeros([self.n_control_pts,2])
        radii        = np.zeros([self.n_control_pts,2])
        delta_b      = np.zeros([self.n_control_pts,2])

        # Compute all informations to generate curves
        for i in range(self.n_control_pts):
            # Collect points
            prv  = (i-1)
            crt  = i
            nxt  = (i+1)%self.n_control_pts
            pt_m = control_pts[prv,:]
            pt_c = control_pts[crt,:]
            pt_p = control_pts[nxt,:]

            # Compute delta vector
            diff         = pt_p - pt_m
            diff         = diff/np.linalg.norm(diff)
            delta[crt,:] = diff

            # Compute edgy vector
            delta_b[crt,:] = 0.5*(pt_m + pt_p) - pt_c

            # Compute radii
            dist         = compute_distance(pt_m, pt_c)
            radii[crt,0] = 0.5*dist*radius[crt]
            dist         = compute_distance(pt_c, pt_p)
            radii[crt,1] = 0.5*dist*radius[crt]

        # Generate curves
        for i in range(self.n_control_pts):
            crt  = i
            nxt  = (i+1)%self.n_control_pts
            pt_c = control_pts[crt,:]
            pt_p = control_pts[nxt,:]
            dist = compute_distance(pt_c, pt_p)
            smpl = math.ceil(self.n_sampling_pts*math.sqrt(dist))

            local_curve = generate_bezier_curve(pt_c,           pt_p,
                                                delta[crt,:],   delta[nxt,:],
                                                delta_b[crt,:], delta_b[nxt,:],
                                                radii[crt,1],   radii[nxt,0],
                                                edgy[crt],      edgy[nxt],
                                                smpl)
            local_curves.append(local_curve)

        curve          = np.concatenate([c for c in local_curves])
        x, y           = curve.T
        z              = np.zeros(x.size)
        self.curve_pts = np.column_stack((x,y,z))
        self.curve_pts = remove_duplicate_pts(self.curve_pts)

        # Center set of points
        center            = np.mean(self.curve_pts, axis=0)
        self.curve_pts   -= center
        self.control_pts[:,0:2] -= center[0:2]

        # Reprocess to position
        self.control_pts[:,0:2] += self.position[0:2]
        self.curve_pts  [:,0:2] += self.position[0:2]

    ### ************************************************
    ### Write image
    def generate_image(self, *args, **kwargs):

        # Handle optional argument
        plot_pts = kwargs.get('plot_pts',  True)
        xmin     = kwargs.get('xmin',     -1.0)
        xmax     = kwargs.get('xmax',      1.0)
        ymin     = kwargs.get('ymin',     -1.0)
        ymax     = kwargs.get('ymax',      1.0)

        # Plot shape
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.fill([xmin,xmax,xmax,xmin],
                 [ymin,ymin,ymax,ymax],
                 color=(0.784,0.773,0.741),
                 linewidth=2.5,
                 zorder=0)
        plt.fill(self.curve_pts[:,0],
                 self.curve_pts[:,1],
                 'black',
                 linewidth=0,
                 zorder=1)

        # Plot points
        # Each point gets a different color
        colors = matplotlib.cm.ocean(np.linspace(0, 1,
                                                 self.n_control_pts))
        plt.scatter(self.control_pts[:,0],
                    self.control_pts[:,1],
                    color=colors,
                    s=16,
                    zorder=2,
                    alpha=0.5)

        # Save image
        filename = self.output_dir+self.name+'.png'

        plt.savefig(filename,
                    dpi=200)
        plt.close(plt.gcf())
        plt.cla()
        trim_white(filename)

    ### ************************************************
    ### Write csv
    def write_csv(self):
        filename = self.output_dir+self.name+'.csv'
        with open(filename,'w') as file:
            # Write header
            file.write('{} {}\n'.format(self.n_control_pts,
                                        self.n_sampling_pts))

            # Write control points coordinates
            for i in range(0,self.n_control_pts):
                file.write('{} {} {} {}\n'.format(self.control_pts[i,0],
                                                  self.control_pts[i,1],
                                                  self.radius[i],
                                                  self.edgy[i]))

    ### ************************************************
    ### Read csv and initialize shape with it
    def read_csv(self, filename, *args, **kwargs):
        # Handle optional argument
        keep_numbering = kwargs.get('keep_numbering', False)

        if (not os.path.isfile(filename)):
            print('I could not find csv file: '+filename)
            print('Exiting now')
            exit()

        self.reset()
        sfile  = filename.split('.')
        sfile  = sfile[-2]
        sfile  = sfile.split('/')
        name   = sfile[-1]

        if (keep_numbering):
            sname = name.split('_')
            name  = sname[0]

        x      = []
        y      = []
        radius = []
        edgy   = []

        with open(filename) as file:
            header         = file.readline().split()
            n_control_pts  = int(header[0])
            n_sampling_pts = int(header[1])

            for i in range(0,n_control_pts):
                coords = file.readline().split()
                x.append(float(coords[0]))
                y.append(float(coords[1]))
                radius.append(float(coords[2]))
                edgy.append(float(coords[3]))
                control_pts = np.column_stack((x,y))

        self.__init__(name,
                      control_pts,
                      n_control_pts,
                      n_sampling_pts,
                      radius,
                      edgy)

    ### ************************************************
    ### Modify shape given a deformation field
    def modify_shape_from_field(self, deformation, pts_list):

        # Deform shape
        for i in range(len(pts_list)):
            self.control_pts[pts_list[i],0] = deformation[i,0]
            self.control_pts[pts_list[i],1] = deformation[i,1]
            self.edgy[pts_list[i]]          = deformation[i,2]

### End of class Shape
### ************************************************

### ************************************************
### Compute distance between two points
def compute_distance(p1, p2):

    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

### ************************************************
### Generate cylinder points
def generate_cylinder_pts(n_pts):
    if (n_pts < 4):
        print('Not enough points to generate cylinder')
        exit()

    pts = np.zeros([n_pts, 2])
    ang = 2.0*math.pi/n_pts
    for i in range(0,n_pts):
        pts[i,:] = [0.5*math.cos(float(i)*ang),
                    0.5*math.sin(float(i)*ang)]

    return pts

### ************************************************
### Generate square points
def generate_square_pts(n_pts):
    if (n_pts != 4):
        print('You should have n_pts = 4 for square')
        exit()

    pts       = np.zeros([n_pts, 2])
    pts[0,:]  = [ 1.0, 1.0]
    pts[1,:]  = [-1.0, 1.0]
    pts[2,:]  = [-1.0,-1.0]
    pts[3,:]  = [ 1.0,-1.0]

    pts[:,:] *= 0.5

    return pts

### ************************************************
### Remove duplicate points in input coordinates array
### WARNING : this routine is highly sub-optimal
def remove_duplicate_pts(pts):
    to_remove = []

    for i in range(len(pts)):
        for j in range(len(pts)):
            # Check that i and j are not identical
            if (i == j):
                continue

            # Check that i and j are not removed points
            if (i in to_remove) or (j in to_remove):
                continue

            # Compute distance between points
            pi = pts[i,:]
            pj = pts[j,:]
            dist = compute_distance(pi,pj)

            # Tag the point to be removed
            if (dist < 1.0e-8):
                to_remove.append(j)

    # Sort elements to remove in reverse order
    to_remove.sort(reverse=True)

    # Remove elements from pts
    for pt in to_remove:
        pts = np.delete(pts, pt, 0)

    return pts

### ************************************************
### Counter Clock-Wise sort
###  - Take a cloud of points and compute its geometric center
###  - Translate points to have their geometric center at origin
###  - Compute the angle from origin for each point
###  - Sort angles by ascending order
def ccw_sort(pts, rad, edg):
    geometric_center = np.mean(pts,axis=0)
    translated_pts   = pts - geometric_center
    angles           = np.arctan2(translated_pts[:,1], translated_pts[:,0])
    x                = angles.argsort()
    pts2             = np.array(pts)
    rad2             = np.array(rad)
    edg2             = np.array(edg)

    return pts2[x,:], rad2[x], edg2[x]

### ************************************************
### Compute Bernstein polynomial value
def compute_bernstein(n,k,t):
    k_choose_n = scipy.special.binom(n,k)

    return k_choose_n * (t**k) * ((1.0-t)**(n-k))

### ************************************************
### Sample Bezier curves given set of control points
### and the number of sampling points
### Bezier curves are parameterized with t in [0,1]
### and are defined with n control points P_i :
### B(t) = sum_{i=0,n} B_i^n(t) * P_i
def sample_bezier_curve(control_pts, n_sampling_pts):
    n_control_pts = len(control_pts)
    t             = np.linspace(0, 1, n_sampling_pts)
    curve         = np.zeros((n_sampling_pts, 2))

    for i in range(n_control_pts):
        curve += np.outer(compute_bernstein(n_control_pts-1, i, t),
                          control_pts[i])

    return curve

### ************************************************
### Generate Bezier curve between two pts
def generate_bezier_curve(p1,       p2,
                          delta1,   delta2,
                          delta_b1, delta_b2,
                          radius1,  radius2,
                          edgy1,    edgy2,
                          n_sampling_pts):

    # Lambda function to wrap angles
    #wrap = lambda angle: (angle >= 0.0)*angle + (angle < 0.0)*(angle+2*np.pi)

    # Sample the curve if necessary
    if (n_sampling_pts != 0):
        # Create array of control pts for cubic Bezier curve
        # First and last points are given, while the two intermediate
        # points are computed from edge points, angles and radius
        control_pts      = np.zeros((4,2))
        control_pts[0,:] = p1[:]
        control_pts[3,:] = p2[:]

        # Compute baseline intermediate control pts ctrl_p1 and ctrl_p2
        ctrl_p1_base = radius1*delta1
        ctrl_p2_base =-radius2*delta2

        ctrl_p1_edgy = radius1*delta_b1
        ctrl_p2_edgy = radius2*delta_b2

        control_pts[1,:] = p1 + edgy1*ctrl_p1_base + (1.0-edgy1)*ctrl_p1_edgy
        control_pts[2,:] = p2 + edgy2*ctrl_p2_base + (1.0-edgy2)*ctrl_p2_edgy

        # Compute points on the Bezier curve
        curve = sample_bezier_curve(control_pts, n_sampling_pts)

    # Else return just a straight line
    else:
        curve = p1
        curve = np.vstack([curve,p2])

    return curve
from scipy.interpolate import CubicSpline

class ClampedCubicSpline:
    """
    包装 CubicSpline，实现周期性边界下的区间clamp，自动补首尾节点。
    """
    def __init__(self, x, y, **kwargs):
        # 自动补首尾节点，保证周期性区间clamp正确
        if not (np.isclose(x[0], 0) and np.isclose(x[-1], 2*np.pi)):
            x = np.concatenate([x, [2*np.pi]])
            y = np.concatenate([y, [y[0]]])
        self.spline = CubicSpline(x, y, **kwargs)
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.n = len(self.x) - 1

    def __call__(self, xq):
        xq = np.asarray(xq)
        # 模2π归一到[0,2π)
        xq = np.mod(xq, 2 * np.pi)
        inds = np.searchsorted(self.x, xq, side='right') - 1
        inds[inds < 0] = 0
        inds[inds >= self.n] = 0  # 闭合周期，最后区间clamp到[末点,首点]
        yq_raw = self.spline(xq)
        
        y_min = np.minimum(self.y[inds], self.y[(inds + 1) % self.n])
        y_max = np.maximum(self.y[inds], self.y[(inds + 1) % self.n])
        return np.clip(yq_raw, y_min, y_max)

def generate_polar_boundary(r_known):
    # 1. 检查输入
    if not isinstance(r_known, np.ndarray) or r_known.shape != (36,):
        raise ValueError("polar_radii 必须是长度为 36 的一维 NumPy 数组")

    # 2. 构造已知角度（度）和扩展末尾 360°
    theta_known_deg = np.concatenate((np.arange(0, 360, 10), [360]))  # shape=(37,)
    r_known_extended = np.concatenate((r_known, [r_known[0]]))       # shape=(37,)

    # 3. 转换为弧度
    theta_known_rad = np.deg2rad(theta_known_deg)

    # 4. 周期性三次样条插值：bc_type='periodic' 保证 f(0)=f(2π) 及一阶导数连续
    cs = ClampedCubicSpline(theta_known_rad, r_known_extended, bc_type='periodic')

    # 5. 在 [0, 360) 以 2° 为步长采样
    theta_target_deg = np.arange(0, 360, 1)            # 共 180 个角度
    theta_target_rad = np.deg2rad(theta_target_deg)

    # 6. 计算对应半径
    r_target = cs(theta_target_rad)                     # shape=(180,)
    r_mean_target = 0.0005
    r_scale = r_target.mean(axis=-1) / r_mean_target
    r_target = r_target / r_scale

    # 7. 极坐标 -> 笛卡尔坐标，再加上 position 偏移
    x = r_target * np.cos(theta_target_rad)
    y = r_target * np.sin(theta_target_rad)

    boundary_pts = np.column_stack((x, y))           # shape=(180, 2)

    return boundary_pts

### ************************************************
### Crop white background from image
def trim_white(filename):
    im   = PIL.Image.open(filename)
    bg   = PIL.Image.new(im.mode, im.size, (255,255,255))
    diff = PIL.ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    cp   = im.crop(bbox)
    cp.save(filename)

def interpolate_to_36(r_key):
    """
    将长度为 4、6、8、12 或 18 的半径数组 r_key（分别对应每 90°、60°、45°、30°、20° 采样）
    插值到每隔 10°（共 36 个角度）的半径，并返回长度为 36 的 ndarray。

    参数
    ----------
    r_key : ndarray, shape (m,)
        输入的原始半径，m 必须是 4、6、8、12 或 18。r_key[i] 对应角度 i*(360/m) 度。

    返回
    ----------
    r36 : ndarray, shape (36,)
        每 10°（0°,10°,20°,...,350°）对应的半径数组。
    """
    # 检查输入长度是否合法
    m = r_key.shape[0]
    if m not in (4, 6, 8, 12, 18):
        raise ValueError("r_key 长度必须是 4、6、8、12 或 18")

    # 1. 原始角度（度），从 0 开始，每步 360/m，取 m 个点
    theta_key_deg = np.arange(0, 360, 360.0 / m)  # 形状 (m,)

    # 2. 在末尾补一个 360 度，对应半径复制第一个
    theta_ext_deg = np.concatenate((theta_key_deg, [360.0]))  # 长度 m+1
    r_ext = np.concatenate((r_key, [r_key[0]]))               # 长度 m+1

    # 3. 转为弧度
    theta_ext_rad = np.deg2rad(theta_ext_deg)

    # 4. 构造周期性三次样条：bc_type='periodic'
    spline = ClampedCubicSpline(theta_ext_rad, r_ext, bc_type='periodic')

    # 5. 目标角度（度），每隔 10° 共 36 个
    theta_tar_deg = np.arange(0, 360, 10)    # [0,10,20,...,350]
    theta_tar_rad = np.deg2rad(theta_tar_deg)

    # 6. 计算对应半径
    r36 = spline(theta_tar_rad)

    return r36

import math
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
def generate_shape_from_keypoints(keypoints):

    a1 = keypoints[0]
    a2 = keypoints[1]
    a3 = keypoints[2]
    a4 = keypoints[3]
    b1 = keypoints[4]
    b2 = keypoints[5]
    b3 = keypoints[6]
    # 假设 pts 是形状为 (N, 2) 的 NumPy 数组，表示按顺序排列的轮廓点
    pts = [
        [0, 0],
        [a1, b1],
        [a1 + a2, b2],
        [a1 + a2 + a3, b3],
        [a1 + a2 + a3 + a4, 0],
        [a1 + a2 + a3, -b3],
        [a1 + a2, -b2],
        [a1, -b1],
        [0, 0],
    ]

    pts = np.array(pts)
    # 将 x 和 y 拆开
    x = pts[:, 0]
    y = pts[:, 1]
    # 1. 按累积弧长参数化 t
    #    先计算相邻点间的距离
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    #    再求累积和，前面补一个 0，使长度仍为 N
    t = np.concatenate(([0], np.cumsum(dists)))
    #    将 t 归一化到 [0, 1]
    t /= t[-1]
    # 2. 调用 splprep 做参数化的 B 样条拟合，这里 s=0 保证通过所有点
    #    k=3 表示三次样条；per=True 表示首尾闭合
    tck, u = splprep([x, y], u=t, s=0, k=3, per=True)
    # 3. 在更细的参数范围上采样，用 splev 计算平滑曲线上的点
    #    这里取 200 个采样点
    u_fine = np.linspace(0, 1, 200)
    x_smooth, y_smooth = splev(u_fine, tck)
    # x_smooth, y_smooth 都是形状为 (M,) 的一维数组，
    # 且假设 x_smooth[0],y_smooth[0] 和 x_smooth[-1],y_smooth[-1] 是同一点（闭合）。
    xi = x_smooth
    yi = y_smooth
    # 将序列向后平移一个位置（末尾补首个），方便计算 xi* y_{i+1} - x_{i+1}*yi
    x_next = np.roll(xi, -1)
    y_next = np.roll(yi, -1)
    # 计算有向面积（注意这是2A，有符号）
    cross = xi * y_next - x_next * yi
    A2    = np.sum(cross)         # 这等于 2 * A_signed
    A     = 0.5 * np.abs(A2)      # 取绝对值得到正面积
    # 计算 Cx, Cy
    Cx = np.sum((xi + x_next) * cross) / (3 * A2)
    Cy = np.sum((yi + y_next) * cross) / (3 * A2)


    area_target = math.pi * (0.0005) ** 2
    scale = math.sqrt(area_target / A)

    x_smooth_scaled = (x_smooth - Cx) * scale
    y_smooth_scaled = (y_smooth - Cy) * scale

    x_scaled = (x - Cx) * scale
    y_scaled = (y - Cy) * scale
    
    return np.column_stack((x_smooth_scaled, y_smooth_scaled))