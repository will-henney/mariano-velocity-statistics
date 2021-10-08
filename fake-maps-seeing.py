# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Fake maps with fake seeing
#
# + Generate simulated emissivity and velocity cubes
# + Integrate them to get simulated intensity, velocity (and sigma) maps
# + Calculate structure functions and other statistics (PDFs, delta variance, etc)
# + Apply fake seeing to the cubes and investigate the effect on the structure function, etc
# + Also see if there is any difference if we just simulate a velocity map and apply seeing to that

# +
from pathlib import Path
import numpy as np
import json
from astropy.io import fits
import astropy.units as u
from matplotlib import pyplot as plt
import turbustat.statistics as tss
import turbustat.simulator
from turbustat.simulator import make_3dfield, make_extended
import seaborn as sns
import sys
sys.path.append("../muse-strucfunc")
import strucfunc

sns.set_color_codes()
sns.set_context("talk")

# + [markdown] tags=[]
# ## Fake velocity maps only
# -

vmap = make_extended(
    256, powerlaw=4.0, 
    ellip=0.5, theta=45, 
    randomseed=2021_10_07,
)
vmap /= vmap.std()

fig, ax = plt.subplots()
im = ax.imshow(vmap, vmin=-2.5, vmax=2.5, cmap="RdBu_r")
fig.colorbar(im, ax=ax)

# ### Structure function of fake velocity map

sf = strucfunc.strucfunc_numba_parallel(vmap, dlogr=0.05)

sf

fig, ax = plt.subplots()
m = sf["N pairs"] > 0
r = 10**sf["log10 r"][m]
B = sf["Unweighted B(r)"][m]
ax.plot(
    r, 
    B / r, 
    marker=".",
)
ax.set(xscale="log", yscale="log");

# ### Smoothed version of velocity field
#
#

from astropy.convolution import Gaussian2DKernel, convolve_fft

widths = [1, 2, 4, 8, 16, 32]
vmaps = {}
for width in widths:
    kernel = Gaussian2DKernel(x_stddev=width)
    vmaps[width] = convolve_fft(vmap, kernel)

ncols = len(widths)
fig, axgrid = plt.subplots(
    1, ncols, figsize=(10, 5), sharex=True, sharey=True,
)
for ax, width in zip(axgrid, widths):
    im = ax.imshow(
        vmaps[width], 
        origin="lower",
        vmin=-2.5, vmax=2.5,
        cmap="RdBu_r",
    )

sfs = {}
for width in widths:
    sfs[width] = strucfunc.strucfunc_numba_parallel(
        vmaps[width], dlogr=0.05,
    )

fig, ax = plt.subplots()
m = sf["N pairs"] > 0
B0 = sf["Unweighted B(r)"][m]
for width in widths:
    r = 10**sfs[width]["log10 r"][m]
    B = sfs[width]["Unweighted B(r)"][m]
    ax.plot(
        r, 
        B / B0, 
        marker=".",
    )
ax.set(xscale="log", yscale="linear");







# ## Fake emissivity and velocity cubes


