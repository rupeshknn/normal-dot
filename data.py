import os
import re
import numpy as np
from scipy.constants import e
from numba import njit

from model import ROK_ENERGY_UNIT, FREQUENCY, DELTA
from model import q_capacitance, tunnel_capacitance, conductance

NRG_PATH = f"f1web.ijs.si/~zitko/data/chi/" + "U=0.333_Ec=0.065/"
GAMMA_DICT = np.array(
    re.findall("0.\d{4}", " ".join(os.listdir(NRG_PATH))), dtype=float
)

dmrg_text = lambda data: np.array(
    [
        np.genfromtxt(NRG_PATH + f"Gamma={gamma:.4f}/" + data)[:200]
        for gamma in GAMMA_DICT
    ]
)
optical1_data = dmrg_text("optical1.dat")
n1_data = dmrg_text("n1.dat")
n2_data = dmrg_text("n1e.dat")


@njit
def nrg_data_func(u, gamma):
    idx = np.where(GAMMA_DICT == gamma)[0][0]
    nrg_data = [optical1_data[idx, :, :], n1_data[idx, :, :], n2_data[idx, :, :]]
    return nrg_data


@njit
def analytical_data(gamma, log_g0, global_parameters, v0, fit_step):
    u, alpha, temp, log_n = global_parameters

    v0 = v0 * alpha
    normunit = ROK_ENERGY_UNIT * 1e3 / e

    nu = 1 - v0 / (u * normunit)
    n = 10**log_n
    g0 = 10**log_g0
    if log_g0 == 0.0:
        g0 = 0.0
    w = FREQUENCY

    o1, n1, n2 = nrg_data_func(u, gamma)

    s_mg = np.interp(nu, o1[:, 0], o1[:, 1]) * DELTA
    s_mg = (s_mg + s_mg[::-1]) / 2
    n_g = np.interp(nu, n1[:, 0], n1[:, 1])
    n_e = np.interp(nu, n2[:, 0], n2[:, 1])

    dn_g = (
        np.interp(nu, n1[1:, 0], (n1[1:, 1] - n1[:-1, 1]) / (-0.02 * u))
        / ROK_ENERGY_UNIT
    )
    dn_g = (dn_g + dn_g[::-1]) / 2

    dn_e = (
        np.interp(nu, n2[1:, 0], (n2[1:, 1] - n2[:-1, 1]) / (-0.02 * u))
        / ROK_ENERGY_UNIT
    )
    dn_e = (dn_e + dn_e[::-1]) / 2

    if fit_step == 1:
        q_caps = alpha * alpha * q_capacitance(s_mg, temp, n, dn_e, dn_g) * 1e15
        return q_caps, q_caps, q_caps
    ds_mg = (
        np.interp(nu, o1[1:, 0], (o1[1:, 1] - o1[:-1, 1]) / (-0.02 * u))
        * DELTA
        / ROK_ENERGY_UNIT
    )

    q_caps = alpha * alpha * q_capacitance(s_mg, temp, n, dn_e, dn_g) * 1e15
    t_caps = (
        alpha * alpha * tunnel_capacitance(g0, s_mg, ds_mg, temp, n, n_e, n_g, w) * 1e15
    )
    t_caps = (t_caps + t_caps[::-1]) / 2
    c_total = q_caps + t_caps

    couduc = alpha * alpha * conductance(g0, s_mg, ds_mg, temp, n, n_e, n_g, w) * 1e8
    couduc = (couduc + couduc[::-1]) / 2

    return q_caps, c_total, couduc
