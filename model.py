import numpy as np
from scipy.constants import e, k
from numba import njit


@njit
def gamma(g0, v0, T, N):
    numerator = np.exp(v0 / (k * T)) + N
    denominator = np.exp(v0 / (k * T)) - 1
    return g0 * numerator / denominator


@njit
def etta(g0, v0, T, N):
    numerator = N * gamma(g0, v0, T, N)
    denominator = 4 * k * T * ((np.cosh(v0 / (2 * k * T))) ** 2)
    return numerator / denominator


@njit
def llambda(v0, T, N):
    numerator = np.exp(v0 / (k * T)) + 1
    denominator = np.exp(v0 / (k * T)) + N
    return numerator / denominator


@njit
def p0(smg, T, N):
    numerator = np.exp(smg / (k * T))
    denominator = np.exp(smg / (k * T)) + N
    return numerator / denominator


@njit
def q_capacitance(smg, T, N, dne, dng):
    return (e**2) * (p0(smg, T, N) * (dne - dng) - dne)


@njit
def tunnel_capacitance(g0, smg, dsmg, T, N, ne, ng, w):
    numerator = (
        (e**2)
        * etta(g0, smg, T, N)
        * (llambda(smg, T, N) ** 2)
        * dsmg
        * (ne - ng)
        * gamma(g0, smg, T, N)
    )
    denominator = gamma(g0, smg, T, N) ** 2 + w**2
    return numerator / denominator


@njit
def total_capacitance(g0, smg, dsmg, T, N, ne, ng, dne, dng, w):
    return tunnel_capacitance(g0, smg, dsmg, T, N, ne, ng, w) + q_capacitance(
        smg, T, N, dne, dng
    )


@njit
def conductance(g0, smg, dsmg, T, N, ne, ng, w):
    numerator = (
        (e**2)
        * etta(g0, smg, T, N)
        * (llambda(smg, T, N) ** 2)
        * dsmg
        * (ne - ng)
        * (w**2)
    )
    denominator = gamma(g0, smg, T, N) ** 2 + w**2
    return numerator / denominator
