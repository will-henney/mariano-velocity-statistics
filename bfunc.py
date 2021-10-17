"""
New improved structure function models 

Will Henney 2021-10-16: Hopefully, these are the final versions

bfunc00s is the basic model for B(r) based on an exponential autocorrelation function
bfunc03s includes the effect of seeing with rms width s0 plus noise
bfunc04s is the same with additional effect of finite box size

See new-model-strucfuncs.{py,ipynb} for more details
"""
import numpy as np


def bfunc00s(r, r0, sig2, m):
    "Simple 3-parameter structure function"
    C = np.exp(-np.log(2) * (r / r0) ** m)
    return 2.0 * sig2 * (1.0 - C)


def seeing_large_scale(s0, r0):
    return np.exp(-s0 / r0)


def seeing_empirical(r, s0, r0, a=1.5):
    """
    Simplified version of empirical fit to B(r) reduction from seeing
    """
    return seeing_large_scale(s0, r0) / (1 + (2 * s0 / r) ** a)


def bfunc03s(r, r0, sig2, m, s0, noise):
    "Structure function with better seeing (scale `s0`) and noise"
    return seeing_empirical(r, s0, r0) * bfunc00s(r, r0, sig2, m) + noise


def finite_box_effect(r0, L, scale=3.6):
    return 1 - np.exp(-L / (scale * r0))


def bfunc04s(r, r0, sig2, m, s0, noise, box_size):
    "Structure function with better seeing (scale `s0`) and noise, plus finite box effect"
    boxeff = finite_box_effect(r0, box_size)
    return (
        # Note that the seeing is unaffected by boxeff
        seeing_empirical(r, s0, r0) * bfunc00s(r, boxeff * r0, boxeff * sig2, m)
        + noise
    )
