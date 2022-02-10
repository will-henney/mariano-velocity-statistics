# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
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

# # Structure function (and other statistics) for NGC 346 MUSE map

import numpy as np
from astropy.io import fits

import sys
sys.path.append("../muse-strucfunc")
import strucfunc


# Load the moment maps from FITS files

hdu = fits.open("data/NGC346/ngc346-hi-6563-bin01-vmean.fits")
hdu.info()

vha = hdu["DATA"].data.astype("float")

iha = fits.open(
    "data/NGC346/ngc346-hi-6563-bin01-sum.fits"
)["DATA"].data.astype("float")

# Note that we explicitly set the dtype because MPDAF saves the FITS files in 32-bit format to save space, but my strucfunc library requires default floats. 
#
# Now, deal with bad pixels.  Set velocities to the mean and intenisty to zero.

m = ~np.isfinite(iha * vha) | (iha < 0.0)
iha[m] = 0.0
vha[m] = np.nanmean(vha)
iha /= iha.max()

# Make a map of the velocity field.

from matplotlib import pyplot as plt
plt.style.use([
    "seaborn-poster",
])
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    vha - np.nanmean(vha), 
    origin="lower",
    vmin=-15, vmax=15,
    cmap="RdBu_r",
)
fig.colorbar(im).set_label("km/s")
ax.set(
    xlim=[10, 340],
    ylim=[10, 315],
)
ax.set_aspect("equal")


# ## Hα structure function
#
# ### Calculate the structure function

# #### Revisiting 2021-10-15
#
# Trim of the edges of the map.  Use the same boundaries as in the above image.

trim = (slice(10, 340), slice(10, 315))

vha[trim].shape

sf = strucfunc.strucfunc_numba_parallel(vha[trim], wmap=iha[trim], dlogr=0.05)

sf

sig2 = np.var(vha[trim])
sig2

good = (~m) & (iha > 0.001)
sf["Unweighted mean velocity"] = np.mean(vha[good])
sf["Unweighted sigma^2"] = np.var(vha[good])
v0w = sf["Weighted mean velocity"] = np.average(vha, weights=iha)
sf["Weighted sigma^2"] = np.average((vha - v0w)**2, weights=iha)


{k: sf[k] for k in sf if "sigma" in k or "mean" in k}

import astropy.units as u

# Convert pixels to parsecs at distance to SMC:

D = 61.7 * u.kpc
pix_pc = (0.2 * (D / u.pc) * u.au).to(u.pc)
pix_pc

sf["sep, pc"] = 10**sf["log10 r"] * pix_pc.value

# ### Plot the Hα structure function

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_color_codes()
sns.set_context("talk")

# +
fig, ax = plt.subplots(figsize=(5, 5))
rgrid = np.logspace(
    np.log10(sf["sep, pc"][0]),
    np.log10(sf["sep, pc"][-1]),
    200
)
    
ax.plot(
    "sep, pc", "Unweighted B(r)", data=sf,
    linestyle="none",
    marker="o", label="unweighted",
)
ax.plot(
    "sep, pc", "Weighted B(r)", data=sf,
    linestyle="none",
    marker=".", label="weighted",
)
ax.axhline(sig2, linestyle="dashed")

rgrid = np.logspace(-1.2, 1.2)
mm = 0.66
ax.plot(
    rgrid, 25*rgrid**mm,
    color="r", linestyle="dashed",
    label=fr"$s^{{{mm}}}$",
)
mm = 1.05
ax.plot(
    rgrid, 38*rgrid**mm,
    color="m", linestyle="dashed",
    label=fr"$s^{{{mm}}}$",
)

ax.legend()
ax.set(
    xscale="log",
    yscale="log",
    xlabel="Separation, pc",
    ylabel=r"$B(r)$ (km/s)$^2$",
    ylim=[0.5, 1.5e3],
);
# -

# I am using the same y-axis scale as for 30 Dor.  The structure function starts off the same at small separations, but levels off at a much lower value.
#
# I have now fixed the issue with the weighted version.

# ### Save the Hα structure function to JSON file
#
# This is so we can do a better version of the plot later (as in tarantula-strucfunc-plot-ha.pdf)

# +
import json

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


# +
jsonfilename = f"ngc346-strucfunc-ha.json"
with open(jsonfilename, "w") as f:
    json.dump(sf, fp=f, indent=3, cls=MyEncoder)


# -

# ### Hα structure function conclusions
#
# The correlation length is about 1.5 pc. 
#
# The plane-of-sky sigma is 5.65 km/s
#
# These are consistent with the trends of our other sources, given the size and luminosity of NGC 346.

# ## [S II] structure function

# +
isii = fits.open(
    "data/NGC346/ngc346-sii-6716-bin01-sum.fits"
)["DATA"].data.astype("float")
vsii = fits.open(
    "data/NGC346/ngc346-sii-6716-bin01-vmean.fits"
)["DATA"].data.astype("float")


# -

m = ~np.isfinite(isii * vsii) | (isii < 0.0)
isii[m] = 0.0
vsii[m] = np.nanmean(vsii)
isii /= isii.max()

sf_sii = strucfunc.strucfunc_numba_parallel(
    vsii, 
    wmap=isii
)

good = (~m) & (isii > 0.001)
sf_sii["Unweighted mean velocity"] = np.mean(vsii[good])
sf_sii["Unweighted sigma^2"] = np.var(vsii[good])
v0w = sf_sii["Weighted mean velocity"] = np.average(
    vsii, weights=isii
)
sf_sii["Weighted sigma^2"] = np.average(
    (vsii - v0w)**2, 
    weights=isii
)

{k: sf_sii[k] for k in sf_sii 
 if "sigma" in k or "mean" in k}


sf_sii["sep, pc"] = 10**sf_sii["log10 r"] * pix_pc.value

# +
fig, ax = plt.subplots(figsize=(5, 5))
    
ax.plot(
    "sep, pc", "Unweighted B(r)", data=sf_sii,
    linestyle="none",
    marker="o", label="unweighted",
)
ax.plot(
    "sep, pc", "Weighted B(r)", data=sf_sii,
    linestyle="none",
    marker=".", label="weighted",
)
ax.axhline(sf_sii["Unweighted sigma^2"], 
           linestyle="dashed",
          )
ax.legend()
ax.set(
    xscale="log",
    yscale="log",
    xlabel="Separation, pc",
    ylabel=r"$B(r)$ (km/s)$^2$",
    ylim=[0.5, 1.5e3],
);
# -

jsonfilename = f"ngc346-strucfunc-sii.json"
with open(jsonfilename, "w") as f:
    json.dump(sf_sii, fp=f, indent=3, cls=MyEncoder)

# This is a lot flatter than the Hα version.  Lots of this is due to noise at the smallest scale.
#
# We can try and correct that by just subtracting it off:

# +
fig, ax = plt.subplots(figsize=(5, 5))

sep = sf_sii["sep, pc"]

# Assume fraction `frac` of variation at small scales is noise
bu0 = sf_sii["Unweighted B(r)"][0]
bw0 = sf_sii["Weighted B(r)"][0]
frac = 0.92
bu = sf_sii["Unweighted B(r)"] - frac*bu0
bw = sf_sii["Weighted B(r)"] - frac*bw0

ax.plot(
    sep, bu,
    linestyle="none",
    marker="o", label="unweighted",
)
ax.plot(
    sep, bw,
    linestyle="none",
    marker=".", label="weighted",
)
rgrid = np.logspace(-1.2, 1.2)
mm = 0.66
ax.plot(
    rgrid, 25*rgrid**mm,
    color="r", linestyle="dashed",
    label=fr"$s^{{{mm}}}$",
)

sig2_sii = sf_sii["Unweighted sigma^2"] - 0.5*frac*bu[0]
ax.axhline(
    sig2_sii, 
    linestyle="dashed",
)
ax.legend()
ax.set(
    xscale="log",
    yscale="log",
    xlabel="Separation, pc",
    ylabel=r"$B(r)$ (km/s)$^2$",
    ylim=[0.5, 1.5e3],
);
# -

np.sqrt(sig2_sii)

# By subtracting just the right amount of noise, it is possible to make the unweighted [S II] structure function into a perfect straight line! 
#
# (I am assuming a 92% noise contribution to the observed strucfunc at the smallest separation). 
#
# The fact that it keeps climbing at all scales implies that the true correlation length is larger than the FOV of the observations.
#
# *But this also implies that the true $\sigma$ is larger than the value that we measure*

# Also, note that the intermediate part of the structure function is identical between Hα and [S II], with both having a slope of 2/3 and the same normalization: $B(r) = 25\ (\mathrm{km/s})^2$ at 1 pc.
#
# ## An idea about how to improve the structure function fits
#
# 1. What we could do is to incorporate the noise term and the seeing into the structure function model.
#     1. In some cases, the noise level can be determined independently.  For instance, in the MUSE data by propagation of errors in the moment calculations.
# 2. Also, we can include the total $\sigma^2$ and the $\ell_0$ as fitting parameters, rather than determining them directly from the data
#     1. The reason for doing this is to allow a fit where $\ell_0$ is bigger than our maximum separation (although of course neither $\ell_0$ nor $\sigma$ will be well-constrained in this case).
# 3. So we now have five parameters:
#     1. $\sigma$
#     2. $\ell_0$
#     3. $m$
#     4. Seeing FWHM
#     5. Noise level
# 4. We can maybe use MCMC to do the model fitting.  Then we can use priors to constrain some of the parameters. And we can construct the posterior distributions of all the parameters
#     1. In the case of the sparsely sampled observations, such as Hubble V and X, we will probably find severe degeneracies.  For instance between $m$ and the noise and the seeing. 
#     2. But at least we will be able to quantify this


