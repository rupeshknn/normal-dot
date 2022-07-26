# pylint: disable=invalid-name
"""Module that loads the simulation and experimental data"""

import os
import re
import numpy as np
from scipy.constants import e
from numba import njit

from model import q_capacitance, tunnel_capacitance, conductance

DELTA = 0.250 * 1e-3 * e
ROK_ENERGY_UNIT = DELTA / 0.166
V_RANGE = 0.605
FREQUENCY = 2 * np.pi * 368 * 1e6
U = 0.333  # in rok units

NRG_PATH = f"data/U={U}_Ec=0.065/"
# U = re.findall("U=[0-9 .]+", NRG_PATH)
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


EXPERIMENTAL_PATH = (
    "data/opening_parity_1/" + "dataset_opening_parity_1_"
)
Rows_to_load = [0, 1, 2]
Gate_Voltages = np.round([-159.8 + 0.1 * idx for idx in range(5, 184, 17)], 1)
EXPERIMENTAL_DATA = np.array(
    [
        np.loadtxt(EXPERIMENTAL_PATH + f"{v_g}.csv", skiprows=1, delimiter=",")[
            :, Rows_to_load
        ]
        for v_g in Gate_Voltages
    ]
)


@njit
def exp_data_func(data_idx):
    idx = np.where(Gate_Voltages == data_idx)[0][0]
    return EXPERIMENTAL_DATA[idx, :, 0:2], EXPERIMENTAL_DATA[idx, :, 0:3:2]


@njit
def nrg_data_func(gamma):
    idx = np.where(GAMMA_DICT == gamma)[0][0]
    nrg_data = [optical1_data[idx, :, :], n1_data[idx, :, :], n2_data[idx, :, :]]
    return nrg_data


@njit
def analytical_data(gamma, log_g0, global_parameters, v0, fit_step):
    alpha, temp, log_n = global_parameters

    v0 = v0 * alpha
    normunit = ROK_ENERGY_UNIT * 1e3 / e

    nu = 1 - v0 / (U * normunit)
    n = 10**log_n
    g0 = 10**log_g0
    if log_g0 == 0.0:
        g0 = 0.0
    w = FREQUENCY

    o1, n1, n2 = nrg_data_func(gamma)

    s_mg = np.interp(nu, o1[:, 0], o1[:, 1]) * DELTA
    s_mg = (s_mg + s_mg[::-1]) / 2
    n_g = np.interp(nu, n1[:, 0], n1[:, 1])
    n_e = np.interp(nu, n2[:, 0], n2[:, 1])

    dn_g = (
        np.interp(nu, n1[1:, 0], (n1[1:, 1] - n1[:-1, 1]) / (-0.02 * U))
        / ROK_ENERGY_UNIT
    )
    dn_g = (dn_g + dn_g[::-1]) / 2

    dn_e = (
        np.interp(nu, n2[1:, 0], (n2[1:, 1] - n2[:-1, 1]) / (-0.02 * U))
        / ROK_ENERGY_UNIT
    )
    dn_e = (dn_e + dn_e[::-1]) / 2

    if fit_step == 1:
        q_caps = alpha * alpha * q_capacitance(s_mg, temp, n, dn_e, dn_g) * 1e15
        return q_caps, q_caps, q_caps
    ds_mg = (
        np.interp(nu, o1[1:, 0], (o1[1:, 1] - o1[:-1, 1]) / (-0.02 * U))
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


@njit
def experimental_data(data_idx, symmetrize):  # filter
    c_data, r_data = exp_data_func(data_idx)

    exp_c = c_data[:, 1]
    exp_g = r_data[:, 1]
    exp_v = (c_data[:, 0] - np.mean(c_data[:, 0])) * 1e3

    filter_bool = (exp_v < V_RANGE) * (-V_RANGE < exp_v)

    exp_c = exp_c[filter_bool] * 1e15
    exp_g = exp_g[filter_bool] * 1e8
    exp_v = exp_v[filter_bool]

    if symmetrize:
        exp_c = (exp_c + exp_c[::-1]) / 2
        exp_g = (exp_g + exp_g[::-1]) / 2

    return exp_v, exp_c, exp_g
