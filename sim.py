# Generic imports
import os
import sys

# Custom imports
from lbm.src.app.turek import turek
from lbm.src.core.lattice import lattice
from lbm.src.core.momentum     import run_momentum, solve_pressure_poisson, calc_pressure
from lbm.src.core.energy  import solve_temperature_convdiff_seq
from lbm.src.plot.plot     import plot_temperature, plot_pressure
import numpy as np

p0 = 1.2670132e-05 * 1922.6569 * 0.207
t0 = 0.01186329

def calc_score(lattice):
    u_in = lattice.u[:, 0, 0, :]
    u_out = lattice.u[:, 0, -1, :]

    p_in = lattice.p[:, 0, :].mean(axis=-1)
    p_out = lattice.p[:, -1, :].mean(axis=-1)
    delta_p = ((p_in - p_out) / p0).get()

    t_in = (lattice.t[:, 0, :] * u_in).mean(axis=-1)
    t_out = (lattice.t[:, -1, :] * u_out).mean(axis=-1)
    print(t_out - t_in)
    delta_t = ((t_out - t_in) / t0).get()

    mask = ((delta_p < 5.0) & (delta_p > 0.2) & (delta_t < 5.0) & (delta_t > 0.2))
    
    score = np.zeros_like(delta_p)
    score_m = delta_t[mask] / np.power(delta_p[mask], 0.333)
    score[mask] = score_m

    print(delta_p, delta_t, score)
    return score

def do_simulation(datas):
    print(datas)
    app = turek(datas, shape_type="keypoints")
    ltc = lattice(app)
    run_momentum(ltc, app)
    solve_pressure_poisson(ltc)
    # calc_pressure(ltc)
    solve_temperature_convdiff_seq(ltc)

    plot_pressure(ltc)
    plot_temperature(ltc)
    
    score = calc_score(ltc)
    return score

if __name__ == '__main__':
    # Instanciate app
    # {'x0': 2.2696328000899086, 'x1': 0.7414726501490139, 'x2': 1.2735723076320569, 'x3': 2.5962953073550206, 'x4': 1.9581265188611017, 'x5': 2.5968147709349516}
    # datas = [np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]
    # datas = [np.array([2.2696328000899086, 0.7414726501490139, 1.2735723076320569, 2.5962953073550206, 1.9581265188611017, 2.5968147709349516])] # 6参数最终优化结果
    # datas = [np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]
    # {'x0': 1.4877296577060954, 'x1': 0.8581440582214137, 'x2': 0.5482359939861323, 'x3': 0.5, 'x4': 1.3245725892830116}
    # datas = [np.array([1.4877296577060954, 0.8581440582214137, 0.5482359939861323, 0.5, 1.3245725892830116, 0.5, 0.5482359939861323, 0.8581440582214137])]
    # {'x0': 0.9, 'x1': 0.9, 'x2': 0.9, 'x3': 0.6446182196355664, 'x4': 0.1, 'x5': 0.1, 'x6': 0.1336793918336705}
    # datas = [np.array([0.9, 0.9, 0.9, 0.6446182196355664, 0.1, 0.1 #1#, 0.1336793918336705])]
    score = do_simulation(datas)