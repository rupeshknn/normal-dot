import numpy as np
from numba import njit
from data import analytical_data, experimental_data, Gate_Voltages, V_RANGE, GAMMA_DICT

# Fitting Step 1
@njit
def grad(f, dx):
    return (f[1:] - f[:-1]) / dx


@njit
def weight_def(gamma, dset, global_parameters, sym):
    cap_thresh = 0.027  # fF
    exp_v, exp_c, _ = experimental_data(dset, sym)
    q_caps, _, _ = analytical_data(gamma, 0.0, global_parameters, exp_v, fit_step=1)
    weight_qc = (
        ((q_caps - exp_c) ** 2)
        / (np.maximum(cap_thresh, (q_caps + exp_c) / 2) ** 2)
        * (1 - np.abs(exp_v) / V_RANGE)
    )
    if dset != Gate_Voltages[-1]:  # dips present
        if grad(grad(q_caps, 0.02), 0.02)[np.argmin(np.abs(grad(q_caps, 0.02)))] < 0:
            return np.sum(weight_qc) + 100

    return np.sum(weight_qc)


@njit
def min_gamma_weight(dset, global_parameters, sym):
    gamma_set = GAMMA_DICT
    weight_set = np.array(
        [weight_def(gamma, dset, global_parameters, sym) for gamma in gamma_set]
    )
    weight_min_idx = np.argmin(weight_set)
    return gamma_set[weight_min_idx], weight_set[weight_min_idx]


@njit
def total_weight(global_parameters, sym):
    weight_set = np.array(
        [min_gamma_weight(v_g, global_parameters, sym)[1] for v_g in Gate_Voltages]
    )
    total_weight = np.sum(weight_set)
    return total_weight
