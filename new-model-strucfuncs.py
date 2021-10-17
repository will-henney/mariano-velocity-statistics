# -*- coding: utf-8 -*-
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

# # New model structure functions
#
# Test out the adjustments to the structure function model that account for the finite box effect. 

# ## The model itself

# The structure function is written in terms of the spatial autocorrelation 
# $$
# B(r) = 2 \sigma^2 \left( 1 - C \right)
# $$
# where the spatial autocorrelation (normalized autocovariance) is modeled as 
# $$
# C(r) = \exp \left[ -\ln 2 \, \left( \frac{r}{r_0} \right)^m \right]
# $$
# where $r_0$ is the coherence length and $m$ is the power law slope at small scales. 
# This replaces the model we had previously:
# $$
# C(r) = \left[ 1 + \left( \frac{r}{r_0} \right)^m \right]^{-1}
# $$
# The advantage of the exponential version is that it makes a shaper transition from the power law growth to the flat portion. 
#
# Note that the model autocorrelation is always positive, whereas in principle it could go between $[-1, 1]$. Negative autocorrelation can be produced by periodic fluctuations of wavelength $\lambda$ – negative peaks occur at $r = \lambda/2, 3\lambda/2, \dots$. 
#
# The integral scale of the turbulence is given by $r_\mathrm{int} = \xi(m) r_0$ where
# $$
# \xi(m) = \frac{ 1} { m \, (\ln 2)^{1/m}} \, \Gamma\left(\frac{1}{m}\right) \sim 1.5
# $$
# The $\xi$ factor falls from 2.3 to 1.1 as $m$ rises from $2/3$ to $5/3$. 

def bfunc00s(r, r0, sig2, m):
    "Simple 3-parameter structure function"
    C = np.exp(-np.log(2) * (r / r0)**m)
    return 2.0 * sig2 * (1.0 - C)


# The seeing is the empirical function that we fitted to the fake data.  After some manipulation, we can reduce the fitting function to:
# $$
# S(r; s_0, r_0) = 
# \frac{ e^{-s_0 / r_0} }
# {1 + (2 s_0 / r)^a}
# $$
# with $a = 3/2$ yielding the best fit.

# +
def seeing_large_scale(s0, r0):
    return np.exp(-s0 / r0) 

def seeing_empirical(r, s0, r0, a=1.5):
    """
    Simplified version of empirical fit to B(r) reduction from seeing
    """
    return seeing_large_scale(s0, r0) / (1 + (2 * s0 / r)**a)



# -

def bfunc03s(r, r0, sig2, m, s0, noise):
    "Structure function with better seeing (scale `s0`) and noise"
    return seeing_empirical(r, s0, r0) * bfunc00s(r, r0, sig2, m) + noise


# Finally, we have the finite-box effect, which depends on the box size $L$ relative to the correlation length:

# +
def finite_box_effect(r0, L, scale=3.6):
    return 1 - np.exp(-L / (scale * r0))

def bfunc04s(r, r0, sig2, m, s0, noise, box_size):
    "Structure function with better seeing (scale `s0`) and noise, plus finite box effect"
    boxeff = finite_box_effect(r0, box_size)
    return seeing_empirical(r, s0, r0) * bfunc00s(r, boxeff*r0, boxeff*sig2, m) + noise


# -

# # Example structure functions

import cmasher as cmr
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_color_codes()
sns.set_context("talk")

# Compare true and apparent structure functions:

# +
fig, axes = plt.subplots(
    2, 4, 
    sharex=True, sharey=True,
    figsize=(16, 8),
)
r = np.logspace(0.0, 2.5)
box_size = 212.0
sig2 = 1.0

for (r0, s0, m, noise, label), ax in zip([
        [20.0, 3.0, 1.0, 0.03, "standard"],
        [20.0, 1.5, 1.0, 0.03, "smaller s0"],
        [20.0, 6.0, 1.0, 0.03, "larger s0"],
        [20.0, 3.0, 1.0, 0.05, "more noise"],
        [10.0, 3.0, 1.0, 0.03, "smaller r0"],
        [100.0, 3.0, 1.0, 0.03, "larger r0"],
        [20.0, 3.0, 0.667, 0.03, "shallow m"],
        [20.0, 3.0, 1.4, 0.03, "steep m"],
], axes.flat):
    true_b = ax.plot(
        r, bfunc00s(r, r0, sig2, m),
        label="true",
    )
    c_true = true_b[0].get_color()
    app_b = ax.plot(
        r, 
        bfunc04s(r, r0, sig2, m, s0, noise, box_size),
        label="apparent",
    )
    c_app = app_b[0].get_color()
    
    ax.axhline(sig2, color=c_true, linestyle="dotted")
    ax.axvline(r0, color=c_true, linestyle="dotted")
    
    
    sig2_app = finite_box_effect(r0, box_size) * seeing_large_scale(s0, r0) * sig2
    rr, bb = app_b[0].get_data()
    r0_app = np.interp(sig2_app, bb, rr)
    
    ax.axhline(sig2_app, color=c_app, linestyle="dotted")
    ax.axvline(r0_app, color=c_app, linestyle="dotted")
    ax.set_title(label)
axes[0, 0].legend()             
axes[-1, 0].set(
    xscale="log",
    yscale="log",
    ylim=[0.04, 2.5],
    xlabel=r"separation, $r$",
    ylabel=r"$B(r)$",
);
# -

# The "true" curve (blue) is the basic model without any instrumental effects.  The "apparent" curve includes the effects of seeing, noise, and finite box.  The true and apparent $r_0$ and $\sigma^2$ are shown by dotted lines.  The $\sigma^2$ is reduced by both effects, but $r_0$ can be increased by seeing or reduced by finite-box.

# ## Apply to real data
#
# Eventually, we need to repeat the lmfit fits.  But for the moment, I will just tune the parameters by hand to get something that fits OK. 
#
# _Actually, it might be worth trying to use `linmix` to do these fits. No - can't do that since linmix is only for fitting straight lines, but we could look at some of the [mcmc extensions of lmfit](https://lmfit.github.io/lmfit-py/examples/example_emcee_Model_interface.html)._

# ### Orion

import json


def values2arrays(d):
    for k in d.keys():
        if type(d[k]) == list:
            d[k] = np.array(d[k])
    return d


with open("orion-strucfunc-ha.json") as f:
    sf = values2arrays(json.load(f))

mask = sf["N pairs"] > 0

import astropy.units as u

Distance_pc = 410
pix_scale_arcsec = 0.534
pc_per_arcsec = (Distance_pc * (u.au / u.pc)).cgs.value
pix_scale_pc = pc_per_arcsec * pix_scale_arcsec
pix_scale_pc

# +
fig, ax = plt.subplots(figsize=(12, 12))


r = 10**sf["log10 r"][mask]
r *= pix_scale_pc
B = sf["Unweighted B(r)"][mask]

# Observed strucfunc
ax.plot(r, B, linestyle="none", marker="o", color="k")

# Model parameters
L = np.sqrt(356 * 512) * pix_scale_pc
s0 = 1.0 * pc_per_arcsec
r0 = 0.07
m = 1.1
sig2 = 13.5
noise = 0.01

# Model strucfunc
rgrid = np.logspace(np.log10(r[0]), np.log10(r[-1]))

true_b = ax.plot(
    rgrid, bfunc00s(rgrid, r0, sig2, m),
    label="true",
)
c_true = true_b[0].get_color()
app_b = ax.plot(
    rgrid, 
    bfunc04s(rgrid, r0, sig2, m, s0, noise, L),
    label="apparent",
)
c_app = app_b[0].get_color()


ax.axvline(L, color="k", linestyle="dotted")
ax.axvline(2 * s0, color="k", linestyle="dotted")
ax.axvline(r0, color="k", linestyle="dashed")
ax.axhline(sig2, color="k", linestyle="dashed")
ax.axvspan(L/2, r[-1], color="k", alpha=0.05, zorder=-1)
ax.set(
    xscale="log",
    yscale="log",
)
sns.despine();
# -

# Blue line is the theoretical structure function, while orange line is the preficted observed one, taking into account the instrumental effects. 
#
# Dashed lines show the "true" $r_0$ and $\sigma^2$.  Dotted lines show $2 s_0$ to left and $L$ to right.  The shaded area shows $r > L/2$ where large-scale gradients and periodicity will be important. 

# For Orion the finite-box effect is very small but the effect of seeing is quite important. 

# ### NGC 346

with open("ngc346-strucfunc-ha.json") as f:
    sf = values2arrays(json.load(f))
mask = sf["N pairs"] > 0

Distance_pc = 61700
pix_scale_arcsec = 0.2
pc_per_arcsec = (Distance_pc * (u.au / u.pc)).cgs.value
pix_scale_pc = pc_per_arcsec * pix_scale_arcsec
pix_scale_pc

# +
fig, ax = plt.subplots(figsize=(12, 12))


r = 10**sf["log10 r"][mask]
r *= pix_scale_pc
B = sf["Unweighted B(r)"][mask]

# Merge first K points
K = 3
r[K] = np.mean(r[:K])
B[K] = np.mean(B[:K])
r = r[K:]
B = B[K:]

# Observed strucfunc
ax.plot(r, B, linestyle="none", marker="o", color="k")

# Model parameters
L = np.sqrt(316 * 305) * pix_scale_pc
s0 = 0.2 * pc_per_arcsec
r0 = 5.0
m = 0.87
sig2 = 65
noise = 1.0

# Model strucfunc
rgrid = np.logspace(np.log10(r[0]), np.log10(r[-1]))

true_b = ax.plot(
    rgrid, bfunc00s(rgrid, r0, sig2, m),
    label="true",
)
c_true = true_b[0].get_color()
app_b = ax.plot(
    rgrid, 
    bfunc04s(rgrid, r0, sig2, m, s0, noise, L),
    label="apparent",
)
c_app = app_b[0].get_color()


ax.axvline(L, color="k", linestyle="dotted")
ax.axvline(2 * s0, color="k", linestyle="dotted")
ax.axvline(r0, color="k", linestyle="dashed")
ax.axhline(sig2, color="k", linestyle="dashed")
ax.axvspan(L/2, r[-1], color="k", alpha=0.05, zorder=-1)
ax.set(
    xscale="log",
    yscale="log",
)
sns.despine();
# -

# So, for NGC 346 we have a smaller seeing effect than with Orion, but we do have a more significant finite box effect with $L / r_0$ of about 3.8. The resultant $\sigma^2$ is about twice what we would naively obtain from the data. 
#
# There is a degeneracy between $r_0$ and $\sigma^2$ that needs to be explored with the fits.

# ### 30 Dor

with open("tarantula-strucfunc-ha.json") as f:
    sf = values2arrays(json.load(f))
mask = sf["N pairs"] > 0

Distance_pc = 50000
pix_scale_arcsec = 0.2
pc_per_arcsec = (Distance_pc * (u.au / u.pc)).cgs.value
pix_scale_pc = pc_per_arcsec * pix_scale_arcsec
pix_scale_pc

# +
fig, ax = plt.subplots(figsize=(12, 12))


r = 10**sf["log10 r"][mask]
r *= pix_scale_pc
B = sf["Unweighted B(r)"][mask]

# Merge first K points
K = 3
r[K] = np.mean(r[:K])
B[K] = np.mean(B[:K])
r = r[K:]
B = B[K:]

# Observed strucfunc
ax.plot(r, B, linestyle="none", marker="o", color="k")

# Model parameters
L = 650 * pix_scale_pc
s0 = 0.3 * pc_per_arcsec
r0 = 3.0
m = 1.0
sig2 = 250
noise = 4.0

# Model strucfunc
rgrid = np.logspace(np.log10(r[0]), np.log10(r[-1]))

true_b = ax.plot(
    rgrid, bfunc00s(rgrid, r0, sig2, m),
    label="true",
)
c_true = true_b[0].get_color()
app_b = ax.plot(
    rgrid, 
    bfunc04s(rgrid, r0, sig2, m, s0, noise, L),
    label="apparent",
)
c_app = app_b[0].get_color()


ax.axvline(L, color="k", linestyle="dotted")
ax.axvline(2 * s0, color="k", linestyle="dotted")
ax.axvline(r0, color="k", linestyle="dashed")
ax.axhline(sig2, color="k", linestyle="dashed")
ax.axvspan(L/2, r[-1], color="k", alpha=0.05, zorder=-1)
ax.set(
    xscale="log",
    yscale="log",
)
sns.despine();
# -

# For 30 Dor, the instrumental effects at both ends (seeing and finite-box) are quite small. On the other hand, there is a large deviation at scales of about 2 pixels, which must be due to the pattern noise. 
#
# What we could do there is to merge the first 3 points, or just the first 2. And then it would be well-fit with the noise term. 

# ### NGC 604

# We will use Javier's structure functions for the distant sources

# +
import pickle
from pathlib import Path

jav_path = Path("../JavierGVastro/PhD.Paper/")

# +
with open(jav_path / "SFresults" / "N604H.pkl", "rb") as f:
    sfdata = pickle.load(f)
with open(jav_path / "VFM" / "N604H.pkl", "rb") as f:
    vfdata = pickle.load(f)

sf = sfdata["SF"]
sf
# -

vfdata["vv"].shape

Distance_pc = 840_000
pix_scale_arcsec = vfdata["pix"]
pc_per_arcsec = (Distance_pc * (u.au / u.pc)).cgs.value
pix_scale_pc = pc_per_arcsec * pix_scale_arcsec
pix_scale_pc

pix_scale_arcsec, pc_per_arcsec

# +
fig, ax = plt.subplots(figsize=(12, 12))


r = 10**sf["log10 r"]
r *= pix_scale_pc
B = sf["Unweighted B(r)"]

# Merge first K points
# K = 3
# r[K] = np.mean(r[:K])
# B[K] = np.mean(B[:K])
# r = r[K:]
# B = B[K:]

# Observed strucfunc
ax.plot(r, B, linestyle="none", marker="o", color="k")

# Model parameters
L = 120 * pix_scale_pc
s0 = 0.45 * pc_per_arcsec
r0 = 10.0
m = 0.8
sig2 = 80
noise = 0.0

# Model strucfunc
rgrid = np.logspace(np.log10(r[0]), np.log10(r[-1]))

true_b = ax.plot(
    rgrid, bfunc00s(rgrid, r0, sig2, m),
    label="true",
)
c_true = true_b[0].get_color()
app_b = ax.plot(
    rgrid, 
    bfunc04s(rgrid, r0, sig2, m, s0, noise, L),
    label="apparent",
)
c_app = app_b[0].get_color()


ax.axvline(L, color="k", linestyle="dotted")
ax.axvline(2 * s0, color="k", linestyle="dotted")
ax.axvline(r0, color="k", linestyle="dashed")
ax.axhline(sig2, color="k", linestyle="dashed")
ax.axvspan(L/2, r[-1], color="k", alpha=0.05, zorder=-1)
ax.set(
    xscale="log",
    yscale="log",
)
sns.despine();
# -

# In NGC 604 it is the seeing that is the most important effect.  The finite-box effect does absolutely nothing, which is expected since the observed field covers the entire object. 
#
# _It would be better if the structure function were more finely sampled - use `dlogr=0.05`_
#

# ### NGC 595 

# +
with open(jav_path / "SFresults" / "N595.pkl", "rb") as f:
    sfdata = pickle.load(f)
with open(jav_path / "VFM" / "N595.pkl", "rb") as f:
    vfdata = pickle.load(f)

sf = sfdata["SF"]
sf
# -

vfdata["vv"].shape

Distance_pc = 840_000
pix_scale_arcsec = vfdata["pix"]
pc_per_arcsec = (Distance_pc * (u.au / u.pc)).cgs.value
pix_scale_pc = pc_per_arcsec * pix_scale_arcsec
pix_scale_pc

# +
fig, ax = plt.subplots(figsize=(12, 12))


r = 10**sf["log10 r"]
r *= pix_scale_pc
B = sf["Unweighted B(r)"]

# Merge first K points
# K = 3
# r[K] = np.mean(r[:K])
# B[K] = np.mean(B[:K])
# r = r[K:]
# B = B[K:]

# Observed strucfunc
ax.plot(r, B, linestyle="none", marker="o", color="k")

# Model parameters
L = 150 * pix_scale_pc
s0 = 0.4 * pc_per_arcsec
r0 = 9.0
m = 0.9
sig2 = 70
noise = 1.0

# Model strucfunc
rgrid = np.logspace(np.log10(r[0]), np.log10(r[-1]))

true_b = ax.plot(
    rgrid, bfunc00s(rgrid, r0, sig2, m),
    label="true",
)
c_true = true_b[0].get_color()
app_b = ax.plot(
    rgrid, 
    bfunc04s(rgrid, r0, sig2, m, s0, noise, L),
    label="apparent",
)
c_app = app_b[0].get_color()


ax.axvline(L, color="k", linestyle="dotted")
ax.axvline(2 * s0, color="k", linestyle="dotted")
ax.axvline(r0, color="k", linestyle="dashed")
ax.axhline(sig2, color="k", linestyle="dashed")
ax.axvspan(L/2, r[-1], color="k", alpha=0.05, zorder=-1)
ax.set(
    xscale="log",
    yscale="log",
)
sns.despine();
# -

# NGC 595 is another one that is dominated by seeing.  
#
# Is it possible that the true spectrum is flat and that the entire downturn is due to the seeing?
# Let's do an experiment:
#

# +
fig, ax = plt.subplots(figsize=(12, 12))


r = 10**sf["log10 r"]
r *= pix_scale_pc
B = sf["Unweighted B(r)"]

# Merge first K points
# K = 3
# r[K] = np.mean(r[:K])
# B[K] = np.mean(B[:K])
# r = r[K:]
# B = B[K:]

# Observed strucfunc
ax.plot(r, B, linestyle="none", marker="o", color="k")

# Model parameters
L = 150 * pix_scale_pc
s0 = 1.3 * pc_per_arcsec
r0 = 20.0
m = 0.0
sig2 = 150
noise = 0.0

# Model strucfunc
rgrid = np.logspace(np.log10(r[0]), np.log10(r[-1]))

true_b = ax.plot(
    rgrid, bfunc00s(rgrid, r0, sig2, m),
    label="true",
)
c_true = true_b[0].get_color()
app_b = ax.plot(
    rgrid, 
    bfunc04s(rgrid, r0, sig2, m, s0, noise, L),
    label="apparent",
)
c_app = app_b[0].get_color()


ax.axvline(L, color="k", linestyle="dotted")
ax.axvline(2 * s0, color="k", linestyle="dotted")
ax.axvline(r0, color="k", linestyle="dashed")
ax.axhline(sig2, color="k", linestyle="dashed")
ax.axvspan(L/2, r[-1], color="k", alpha=0.05, zorder=-1)
ax.set(
    xscale="log",
    yscale="log",
)
sns.despine();
# -

# So the answer is "yes", but we have to use an unrealistically large seeing: $s_0 = 1.3$ arcsec, so FWHM of 3 arcsec. So, this means that the true structure function cannot be flat.


