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
#     display_name: Python 3
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
from astropy.utils.misc import JsonCustomEncoder
import astropy.units as u
import cmasher as cmr
from matplotlib import pyplot as plt
import turbustat.statistics as tss
import turbustat.simulator
from turbustat.simulator import make_3dfield
from turb_utils import make_extended
import seaborn as sns
import sys
sys.path.append("../muse-strucfunc")
import strucfunc

sns.set_color_codes()
sns.set_context("talk")

# + [markdown] tags=[]
# ## Fake velocity maps only
# -

# ### Compare pure power law with tapered power law
#
# The standard `make_extended()` function from `turbustat` gives a power law field (`vmap0`).  I have modified it to use an exponential taper at large scales, which I use to make `vmap` (all other parameters are the same as `vmap0`).

# +
r0 = 32.0
N = 256
m = 1.2
vmap = make_extended(
    N, powerlaw=2.0 + m, 
    ellip=0.5, theta=45, 
    correlation_length=r0,
    randomseed=2021_10_08,
)
signorm = vmap.std()
vmap /= signorm

vmap0 = make_extended(
    N, powerlaw=2.0 + m, 
    ellip=0.5, theta=45, 
    randomseed=2021_10_08,
)
signorm0 = vmap0.std()
vmap0 /= signorm0
signorm, signorm0
# -

fig, (ax, axx) = plt.subplots(
    1, 2, 
    sharey=True,
    figsize=(10, 4),
)
imshow_kwds = dict(origin="lower", vmin=-3, vmax=3, cmap="RdBu_r")
im = ax.imshow(vmap, **imshow_kwds)
im = axx.imshow(vmap0, **imshow_kwds)
fig.colorbar(im, ax=[ax, axx])

# The left-hand figure is the tapered map. The right-hand figure is the pure power law map. 

# Check out the power spectrum of these maps

# +
fftmap = make_extended(
    N, powerlaw=2.0 + m, 
    ellip=0.5, theta=45, 
    correlation_length=r0,
    randomseed=2021_10_08,
    return_fft=True,
)
fftmap0 = make_extended(
    N, powerlaw=2.0 + m, 
    ellip=0.5, theta=45, 
    randomseed=2021_10_08,
    return_fft=True,
)

pmap = np.real(fftmap * np.conj(fftmap))
pmap0 = np.real(fftmap0 * np.conj(fftmap0))
# -

fig, (ax, axx) = plt.subplots(
    1, 2, 
    sharey=True,
    figsize=(12, 4),
)
imshow_kwds = dict(origin="lower", cmap="magma", vmin=0.0, vmax=12.0)
imm = axx.imshow(np.log10(pmap), **imshow_kwds)
im = ax.imshow((pmap / pmap0), cmap="gray_r", vmin=0.0, vmax=1.0)
fig.colorbar(im, ax=ax).set_label("Tapered / untapered")
fig.colorbar(imm, ax=axx).set_label("log10 Power spectrum");

# The left-hand image is the ratio of the two power spectra on a linear inverted scale (white = 0, black = 1).  It drops in the middle (low $k$, large scales) as expected.  The right-hand image is the power spectrum itself on a log scale. 

# ### Structure function of fake velocity map
#
# We compare the structure functions with and without the taper. In both cases, we are using `m = 1.2` and `N = 256`

sf = strucfunc.strucfunc_numba_parallel(vmap, dlogr=0.05)

sf0 = strucfunc.strucfunc_numba_parallel(vmap0, dlogr=0.05)

# +
fig, ax = plt.subplots(
    figsize=(8, 8),
)
mask = sf["N pairs"] > 0
r = 10**sf["log10 r"][mask]
B = sf["Unweighted B(r)"][mask]
B0 = sf0["Unweighted B(r)"][mask]

ax.plot(r, B, marker=".")
ax.plot(r, B0, marker=".")

rgrid = np.logspace(0.0, 2.0)

for scale in 0.02, 0.08:
    ax.plot(rgrid, scale * rgrid**m, color="0.8")

ax.axhline(1.0, color="k", linestyle="dotted")
ax.axvline(r0, color="k", linestyle="dotted")
ax.axvline(N / 2, color="k", linestyle="dashed")
ax.axvline(N / 6, color="k", linestyle="dashed")
ax.set(
    xscale="log", yscale="log",
    ylim=[8e-3, 4],
);
# -

# Note that the untapered version (orange) has a good power law structure function that recovers `m = 1.2` over a wide range of scales.  But the tapered version flattens away from the power law quite quickly. 
#
# Now that we are using a shallower power spectrum, we don't get the correlation length `r0` to come out at exactly the tapering length that we fed in. 
#
# The untapered structure function has a maximum at roughly half the box size and falls for larger scales.  This is likely to be because of the periodic boundary conditions – we will investigate further below. Note that the derived `r0` for the untapered version would be about 1/8 of the box size, but that this is not really a correlation length. 
#
# **So we need a total field of view that is at least 8 times the correlation length in order to be able to recover the latter from the structure function**

# ### Smoothed version of velocity field
#
#

# Now we smooth the velocity field directly using a gaussian of different widths.  This is different from what we did with the fake Orion data, where it was the PPV intensity cuve that we smoothed.
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

# These are the maps with the varying degrees of smoothing.  For the greatest degree of smoothing we are starting to reduce the overall amplitude of the fluctuations.

sfs = {}
for width in widths:
    sfs[width] = strucfunc.strucfunc_numba_parallel(
        vmaps[width], dlogr=0.05,
    )

# +
fig, ax = plt.subplots(
    figsize=(8, 8),
)
mask = sf["N pairs"] > 0
r = 10**sf["log10 r"][mask]
B = sf["Unweighted B(r)"][mask]
B0 = sf0["Unweighted B(r)"][mask]

ax.plot(r, B, marker=".", color="c")
ax.plot(r, B0, marker=".", color="orange")

rgrid = np.logspace(-1.0, 2.5)
for a in np.logspace(-3, -1, 9):
    ax.plot(rgrid, a * rgrid**m, color="0.8")

for width in widths:
    r = 10**sfs[width]["log10 r"][mask]
    B = sfs[width]["Unweighted B(r)"][mask]
    line = ax.plot(r, B)
    B0 = np.interp(2*width, r, B)
    c = line[0].get_color()
    ax.plot(2*width, B0, marker="o", color=c)



ax.axhline(1.0, color="k", linestyle="dotted")
ax.axvline(r0, color="k", linestyle="dotted")
ax.axvline(N / 2, color="k", linestyle="dashed")
ax.axvline(N / 6, color="k", linestyle="dashed")
ax.set(
    xscale="log", yscale="log", 
    xlim=[1.0, N], ylim=[7e-3, 4.0]);


# -

# These are the structure functions after smoothing. The big circle symbols show `2 * s0`. The power law becomes steeper for up to about `4 * s0`.

# ### Ratio of smoothed to unsmoothed
#
# I copy the functions from the Orion notebook.

# +
def ratio_empirical(rad, s0, a=1.0):
    """
    Simple tanh law in semi-log space to fit the seeing
    
    Reduction in B(r) is always 0.5 when r = 2 * s0
    Parameter `a` controls the slope of the transition.
    """
    x = np.log(rad / (2 * s0))
    y = np.tanh(a * x)
    return 0.5 * (1.0 + y)

def bfac(x):
    """
    Across-the board reduction in B(r) for x = s0 / r0
    
    Where s0 is RMS seeing width and r0 is correlation length
    """
    return 1 / (1 + 4*x**2)

def seeing_empirical(r, s0, r0, a=0.75):
    return bfac(s0 / r0) * ratio_empirical(r, s0, a)


# -

fig, ax = plt.subplots()
B0 = sf["Unweighted B(r)"][mask]
for width in widths:
    r = 10**sfs[width]["log10 r"][mask]
    B = sfs[width]["Unweighted B(r)"][mask]
    rat = B / B0
    line = ax.plot(r, rat, marker=".",)
    rat0 = np.interp(2*width, r, rat)
    c = line[0].get_color()
    ax.plot(2*width, rat0, marker="o", color=c)
    # Functional fit
    ax.plot(r, seeing_empirical(r, width, r0, 0.75), color=c, linestyle="dashed")
ax.set(xscale="log", yscale="linear");

# This is the ratio of each smoothed structure function to the original structure function. The curves look remarlably similar to what we obtained for Orion, despite the very different methodology used here.
#
# The main difference is that the curves drop at the largest scales.  This must be another consequence of the periodic boundary conditions: for the largest scales, we go all the way round and get close again. 
#
# The dashed lines are the model fit that worked for Orion.  These fit OK, but not great. 

# ### Non-periodic boundaries
#
# We can make a map that is twice as big and then analyze 1/4 of it. That way, it will not be periodic.  First we do the pure power law.

# +
vmap2x2 = make_extended(
    2 * N, powerlaw=2.0 + m, 
    ellip=0.5, theta=45, 
    randomseed=2021_10_08,
)


def split_square_in_4(arr):
    ny, nx = arr.shape
    assert nx == ny and nx % 2 == 0
    slices = slice(None, nx // 2), slice(nx // 2, None)
    corners = []
    for i, j in [
        (0, 0), (0, 1), (1, 0), (1, 1),
    ]:
        corners.append(arr[slices[i], slices[j]])
    return corners

def normalize(arr):
    return (arr - np.mean(arr)) / np.std(arr)

vms = split_square_in_4(vmap2x2)
vms = [normalize(_) for _ in vms]
# -

fig, axes = plt.subplots(
    2, 2, 
    sharex=True, sharey=True,
    figsize=(8, 8),
)
imshow_kwds = dict(origin="lower", vmin=-3, vmax=3, cmap="RdBu_r")
for vm, ax in zip(vms, axes.flat):
    im = ax.imshow(vm, **imshow_kwds)

sfs_np = [
    strucfunc.strucfunc_numba_parallel(vm, dlogr=0.05)
    for vm in vms
]

# +
fig, ax = plt.subplots(
    figsize=(8, 8),
)

Bs = [_["Unweighted B(r)"][mask] for _ in sfs_np]
Bm = np.mean(np.stack(Bs), axis=0)

for _B in Bs:
    ax.plot(r, _B, marker=".")
ax.plot(r, Bm, linewidth=4, color="k")

rgrid = np.logspace(0.0, 2.0)

for scale in 0.02, 0.08:
    ax.plot(rgrid, scale * rgrid**m, color="0.8")

ax.axhline(1.0, color="k", linestyle="dotted")
ax.axvline(r0, color="k", linestyle="dotted")
ax.axvline(N / 2, color="k", linestyle="dashed")
ax.axvline(N / 6, color="k", linestyle="dashed")
ax.set(
    xscale="log", yscale="log",
    ylim=[8e-3, 4],
);
# -

# This time, the apparent correlation length is 1/6 of the map size, albeit with some variation between the 4 quadrants. This is a bit larger than the 1/8 that we found for the periodic case. 

# Now we look at the tapered power law.

# +
vms_t = split_square_in_4(
    normalize(
        make_extended(
            2 * N, powerlaw=2.0 + m, 
            ellip=0.5, theta=45, 
            correlation_length=r0,
            randomseed=2021_10_08,
        )
    )
)
# vms_t = [normalize(_) for _ in vms_t]

fig, axes = plt.subplots(
    2, 2, 
    sharex=True, sharey=True,
    figsize=(8, 8),
)
imshow_kwds = dict(origin="lower", vmin=-3, vmax=3, cmap="RdBu_r")
for vm, ax in zip(vms_t, axes.flat):
    im = ax.imshow(vm, **imshow_kwds)
# -

sfs_npt = [
    strucfunc.strucfunc_numba_parallel(vm, dlogr=0.05)
    for vm in vms_t
]

# +
fig, ax = plt.subplots(
    figsize=(8, 8),
)

Bs = [_["Unweighted B(r)"][mask] for _ in sfs_npt]
Bm = np.mean(np.stack(Bs), axis=0)

for _B in Bs:
    ax.plot(r, _B, marker=".")
ax.plot(r, Bm, linewidth=4, color="k")

rgrid = np.logspace(0.0, 2.0)

for scale in 0.02, 0.08:
    ax.plot(rgrid, scale * rgrid**m, color="0.8")

ax.axhline(1.0, color="k", linestyle="dotted")
ax.axvline(r0, color="k", linestyle="dotted")
ax.axvline(N / 2, color="k", linestyle="dashed")
ax.axvline(N / 6, color="k", linestyle="dashed")
ax.set(
    xscale="log", yscale="log",
    ylim=[8e-3, 4],
);
# -

# #### Effects of smoothing the non-periodic maps

vmap2x2_t = make_extended(
        2 * N, powerlaw=2.0 + m, 
        ellip=0.5, theta=45, 
        correlation_length=r0,
        randomseed=2021_10_08,
    )
widths = [1, 2, 4, 8, 16, 32]
vmap_nps = {}
for width in widths:
    kernel = Gaussian2DKernel(x_stddev=width)
    vmap_nps[width] = split_square_in_4(
        convolve_fft(normalize(vmap2x2_t), kernel, boundary="wrap")
    )
#   vmap_nps[width] = [convolve_fft(normalize(_), kernel) 
#                      for _ in split_square_in_4(vmap2x2_t)] 

# +
ncols = len(widths) + 1
nrows = 4
fig, axes = plt.subplots(
    nrows, ncols, figsize=(8, 5.1), sharex=True, sharey=True,
)
for j, vm in enumerate(vms_t):
    im = axes[j, 0].imshow(vm, **imshow_kwds)
axes[0, 0].set_title("original")
for i, width in enumerate(widths):
    for j, vm in enumerate(vmap_nps[width]):
        im = axes[j, i + 1].imshow(vm, **imshow_kwds)
    axes[0, i + 1].set_title(fr"$s_0 = {width}$")
    
for ax in axes.flat:
    ax.set(xticks=[], yticks=[])
sns.despine(left=True, bottom=True)
fig.tight_layout(h_pad=0.2, w_pad=0.2)
fig.savefig("fake-seeing-nonp-thumbnails.pdf");


# -

def values2arrays(d):
    for k in d.keys():
        if type(d[k]) == list:
            d[k] = np.array(d[k])
    return d


# Try and load the structure function from JSON files.  If that fails, then recalculate the structure functions and save to JSON files.

use_cached_strucfunc = True

if use_cached_strucfunc:
    sfs_npt_s = {
        width: [
            values2arrays(json.load(open(fn))) 
            for fn in sorted(
                Path(".").glob(
                    f"fake-tapered-nonp-s0-{width:03d}-*-strucfunc.json"
                )
            )
        ]
        for width in widths
    }
else:
    sfs_npt_s = {
        width: [
            strucfunc.strucfunc_numba_parallel(vm, dlogr=0.05)
            for vm in vmap_nps[width]
        ]
        for width in widths
    }
    for width in widths:
        for jj, (_sf, _vm) in enumerate(zip(sfs_npt_s[width], vmap_nps[width])):
            jsonfilename = f"fake-tapered-nonp-s0-{width:03d}-{jj:02d}-strucfunc.json"
            sig2 = _sf["Unweighted sigma^2"] = np.var(_vm)
            B = _sf["Unweighted B(r)"][mask]
            true_r0 = _sf["Apparent r0"] = np.interp(sig2, B[:-4], r[:-4])
            with open(jsonfilename, "w") as f:
                json.dump(_sf, fp=f, indent=3, cls=JsonCustomEncoder)

# +
fig, ax = plt.subplots(
    figsize=(8, 8),
)
ax.plot(r, Bm, marker=".", color="c")
true_r0 = np.interp(1.0, Bm[:-4], r[:-4])

rgrid = np.logspace(-1.0, 2.5)
for a in np.logspace(-3, -1, 9):
    ax.plot(rgrid, a * rgrid**m, color="0.8")

for width in widths:
    B = np.mean(np.stack(
        [_["Unweighted B(r)"][mask] for _ in sfs_npt_s[width]]
    ), axis=0)
    line = ax.plot(r, B)
    B0 = np.interp(2*width, r, B)
    c = line[0].get_color()
    apparent_r0 = np.mean([_["Apparent r0"] for _ in sfs_npt_s[width]])
    ax.plot(2*width, B0, marker="o", color=c)
    ax.plot(apparent_r0, B0, marker="s", color=c)



ax.axhline(1.0, color="k", linestyle="dotted")
ax.axvline(true_r0, color="k", linestyle="dotted")
ax.axvline(N / 6, color="k", linestyle="dashed")
ax.set(
    xscale="log", yscale="log", 
    xlim=[1.0, N], ylim=[7e-3, 4.0]);


# -

def bfac(x):
    """
    Across-the board reduction in B(r) for x = s0 / r0
    
    Where s0 is RMS seeing width and r0 is correlation length
    """
    return np.exp(-x)



# +
fig, ax = plt.subplots(figsize=(8, 5))
rat_maxes = []
callout_r0_widths = [8]
callout_s0_widths = [1]

colors = cmr.take_cmap_colors(
    'cmr.dusk', len(widths), 
    cmap_range=(0.25, 0.95),
)

for width, c in zip(widths, colors):
    B = np.mean(np.stack(
        [_["Unweighted B(r)"][mask] for _ in sfs_npt_s[width]]
    ), axis=0)
    rat = B / Bm
    rat_individs = [
        _this["Unweighted B(r)"][mask] / _B
        for _this, _B in zip(sfs_npt_s[width], Bs)
    ]
    rat_sigma = np.std(np.stack(rat_individs), axis=0)
    rat_maxes.append(np.max(rat))
    line = ax.plot(r, rat, marker=".", color=c)
    ax.fill_between(
        r, rat - rat_sigma, rat + rat_sigma,
        color=c, alpha=0.2, linewidth=0,
    )
    #for _rat in rat_individs:
    #    ax.plot(r, _rat, color=c, alpha=0.2)
    rat0 = np.interp(2*width, r, rat)
    #c = line[0].get_color()
    ax.plot(2*width, rat0, marker="o", ms=15, color=c)
    # Functional fit
    ax.plot(r, seeing_empirical(r, width, true_r0, 0.75), color=c, linestyle="dashed")
    # Plot apparent correlation lengths
    apparent_r0 = np.mean([_["Apparent r0"] for _ in sfs_npt_s[width]])
    ax.plot(apparent_r0, rat0, marker="+", ms=15, mew=5, color="w")
    ax.plot(apparent_r0, rat0, marker="+", ms=15, mew=3, color=c)
    ax.plot(
        [2*width, apparent_r0], [rat0]*2,
        linestyle="dotted", color=c, linewidth=3,
    )
    ax.text(
        400, bfac(width / true_r0), 
        fr"$s_0 = {width}$",
        color=c,
        va="center",
        alpha=1.0,
    )
    if width in callout_r0_widths:
        ax.annotate(
            "apparent $r_0$",
            xy=(apparent_r0, rat0),
            xytext=(20, 25),
            ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color=c, shrinkB=6),
            textcoords="offset points",
            color=c,
        )
    if width in callout_s0_widths:
        ax.annotate(
            r"$2 \times s_0$",
            xy=(2 * width, rat0),
            xytext=(-40, 40),
            ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color=c, shrinkB=8),
            textcoords="offset points",
            color=c,
        )
        
ax.axvline(
    true_r0, ymax=0.93,
    color="k", linestyle="dotted", zorder=-1
)
ax.text(
    true_r0, 1.05,
    fr"true $r_0 = {true_r0:.1f}$",
    color="k",
    ha="center", va="top",
)
ax.set(
    xscale="log", 
    yscale="linear",
    ylim=[-0.03, 1.03],
    xlabel="Separation, $r$, pixels",
    ylabel="Reduction in $B(r)$",
)
sns.despine()
fig.tight_layout()
fig.savefig("fake-seeing-nonp-reduction.pdf");
# -

# Finding a new functional form for `bfunc()`.  Previously I had had $\bigl(1 + 4 (s_0/r_0)^2\bigr)^{-1}$ but now I have $\exp(-s_0 / r_0)$. This works much better for this synthetic case, although it is slightly worse than the first form for Orion.

# The shaded area is now the std of the results from the 4 individual panels.

fig, ax = plt.subplots()
ax.plot(widths, rat_maxes, marker="s")
wgrid = np.logspace(0.0, 1.5, 200)
ax.plot(wgrid, bfac(wgrid / true_r0), linestyle="dashed")
#ax.plot(wgrid, bfac2(wgrid / true_r0, a=1.), linestyle="dashed")
ax.set(
    xlabel="s0",
    ylabel="max B",
    ylim=[0.0, 1.0],
    xscale="log",
)


# ## Effects of the finite map size
#
# We can try reducing the size of the box for a field with a given $r_0$ and see if $r_0$ is affected:

def split4(arrs):
    """Split each 2D array in list into 4 quadrants
    
    Returns new list of all the quadrants from each input array.
    Requires that all axis lengths be even.
    """
    rslt = []
    for arr in arrs:
        for vsplit in np.vsplit(arr, 2):
            rslt.extend(np.hsplit(vsplit, 2))
    return rslt


a = np.arange(16).reshape((4, 4))
split4(split4([a]))


# This function splits up the whole image into sub-images that are smaller (linearly) by `2**niter` and calculates the mean and std of the $\sigma^2$ and $r_0$ of them:

def split4_and_strucfuncs(arrs, niter=1):
    subarrs = arrs.copy()
    for i in range(niter):
        subarrs = split4(subarrs)
    BB = []
    sig2s = []
    r0s = []
    for subarr in subarrs:
        sf = strucfunc.strucfunc_numba_parallel(subarr, dlogr=0.05)
        mask = sf["N pairs"] > 0
        B = sf["Unweighted B(r)"][mask]
        BB.append(B)
        r = 10**sf["log10 r"][mask]
        sig2 = np.var(subarr)
        sig2s.append(sig2)
        try:
            i0 = np.argmax(r[B <= sig2])
            i1 = max(0, i0 - 2)
            i2 = min(len(B) - 1, i0 + 2)
            r0 = np.interp(sig2, B[i1:i2], r[i1:i2])
        except:
            r0 = np.nan
        r0s.append(r0)
    r0s = np.array(r0s)
    mgood = np.isfinite(r0s)
    Bmean = np.mean(np.stack(BB), axis=0)
    Bsig = np.std(np.stack(BB), axis=0)
    # calculate quartiles
    Bp25 = np.percentile(np.stack(BB), 25, axis=0)
    Bp75 = np.percentile(np.stack(BB), 75, axis=0)
    Bp05 = np.percentile(np.stack(BB), 5, axis=0)
    Bp95 = np.percentile(np.stack(BB), 95, axis=0)
    return {
        #"subarrs": subarrs,
        "r": r,
        "Bmean": Bmean,
        "Bsig": Bsig,
        "Bp05": Bp05,
        "Bp25": Bp25,
        "Bp75": Bp75,
        "Bp95": Bp95,
        "sig2mean": np.mean(sig2s),
        "sig2sig": np.std(sig2s),
        "sig2_25": np.percentile(sig2s, 25),
        "sig2_75": np.percentile(sig2s, 75),
        "r0mean": np.nanmean(r0s),
        "r0sig": np.nanstd(r0s),
        "r0_25": np.percentile(r0s[mgood], 25),
        "r0_75": np.percentile(r0s[mgood], 75),
    }


splits = {}
for k in [0, 1, 2, 3, 4, 5]:
    splits[2**k] = split4_and_strucfuncs(vms_t, k)

wdata = {
    "n": [], "r0mean": [], "r0sig": [],
    "sig2mean": [], "sig2sig": [],
}
for ii in splits.keys():
    wdata["n"].append(256 / ii)
    for v1 in "r0", "sig2":
        for v2 in "mean", "sig":
            v = v1 + v2
            wdata[v].append(splits[ii][v])
wdata = values2arrays(wdata)
wdata["r0mean"] /= true_r0
wdata["r0sig"] /= true_r0

bcolors = cmr.take_cmap_colors(
    'cmr.flamingo', len(splits.values()), 
    cmap_range=(0.05, 0.75),
)

# +
fig, ax = plt.subplots(
    figsize=(8, 8),
)

whitebox = dict(color="white", alpha=1.0, pad=0.0)
L = 256
N = 2
for split, bcolor in zip(splits.values(), bcolors):
    line = ax.plot(split["r"], split["Bmean"], marker=".", color=bcolor)
    c = line[0].get_color()
    #ax.fill_between(
    #    split["r"], 
    #    split["Bp05"], 
    #    split["Bp95"],
    #    color=c, alpha=0.1, linewidth=0, zorder=-1,
    #)
    ax.fill_between(
        split["r"], 
        split["Bp25"], 
        split["Bp75"],
        color=c, alpha=0.2, linewidth=0, zorder=-1,
    )
    x, y = split["r"][-4], split["Bp75"][-4] * 1.3
    ax.text(
        x, y, (rf"$N = {N**2}$" "\n" rf"$L = {L}$"), 
        color=c, ha="center", va="bottom",
        fontsize="x-small",
        bbox=whitebox,
    )
    L //= 2
    N *= 2
#ax.plot(r, Bm, linewidth=4, color="k")

rgrid = np.logspace(0.0, 2.0)

for scale in 0.025, 0.1:
    ax.plot(rgrid, scale * rgrid**m, linestyle="dotted", color="k")

ax.axhline(1.0, color="k", linestyle="dotted")
ax.axvline(true_r0, color="k", linestyle="dotted")

# plot the range of derived r0 and sigma for smallest L
ax.axhline(split["sig2mean"], color=c, linestyle="dotted")
ax.axhspan(
    split["sig2_25"],
    split["sig2_75"],
    color=c,
    alpha=0.2,
)
ax.axvline(split["r0mean"], color=c, linestyle="dotted")
ax.axvspan(
    split["r0_25"],
    split["r0_75"],
    color=c,
    alpha=0.2,
)

ax.text(
    true_r0, 0.04, 
    rf"$r_0 = {true_r0:.1f}$",
    color="k", fontsize="x-small", ha="center", va="bottom",
    bbox=whitebox,
)
ax.set(
    xscale="log", yscale="log",
    ylim=[0.03, 4.9],
    xlabel=r"Separation, $r$, pixels",
    ylabel=r"$B(r) \, / \, \sigma^2$",
)
ax.set_aspect("equal")
sns.despine()
fig.tight_layout()
fig.savefig("fake-finite-box-strucfunc.pdf");
# -

# We have now changed this plot to use the 5, 25, 75, 95 centiles of the distributions. This is more realistic than using the mean +/- sigma, although it turns out to not make much difference.
#
# I have also plotted the mean and quartiles of the r0 and sigma for the smallest box `L=8`, which gives $r_0 \approx L/4$

# +
NN = 512
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for ax, N in zip(axes.flat, [2, 4, 8, 16]):
    ax.imshow(normalize(vmap2x2_t), **imshow_kwds)
    ax.set(xticks=[], yticks=[])
    L = NN // N
    for i in range(N + 1):
        ax.axhline(i * L, color="w")
        ax.axvline(i * L, color="w")
    ax.text(
        0, NN,
        (rf"$N = {N**2}$" "\n" rf"$L = {L}$"), 
        color="k", ha="left", va="top",
        fontsize="small",
        bbox=whitebox,
    )

sns.despine(left=True, bottom=True)
fig.tight_layout(pad=0, h_pad=0.0, w_pad=0.0)
fig.savefig("fake-finite-box-images.pdf");
# -

N, m, r0

# +
N = 256
vms_r16 = split_square_in_4(
    normalize(
        make_extended(
            2 * N, powerlaw=2.0 + m, 
            ellip=0.5, theta=45, 
            correlation_length=16,
            randomseed=2021_10_08,
        )
    )
)

splits_r16 = {}
for k in [0, 1, 2, 3, 4, 5]:
    splits_r16[2**k] = split4_and_strucfuncs(vms_r16, k)

# +
fig, ax = plt.subplots(
    figsize=(8, 4),
)

for split in splits_r16.values():
    line = ax.plot(split["r"], split["Bmean"], marker=".")
    c = line[0].get_color()
    ax.fill_between(
        split["r"], 
        split["Bmean"] - split["Bsig"], 
        split["Bmean"] + split["Bsig"],
        color=c, alpha=0.1, linewidth=0, zorder=-1,
    )
#ax.plot(r, Bm, linewidth=4, color="k")

rgrid = np.logspace(0.0, 2.0)

for scale in 0.05, 0.2:
    ax.plot(rgrid, scale * rgrid**m, linestyle="dotted", color="k")

_split = splits_r16[1]
true_r0_r16 = np.interp(1.0, _split["Bmean"][:-4], _split["r"][:-4])

ax.axhline(1.0, color="k", linestyle="dotted")
ax.axvline(true_r0_r16, color="k", linestyle="dotted")
ax.set(
    xscale="log", yscale="log",
    ylim=[0.02, 4],
)
sns.despine()
fig.tight_layout();
# -

wdata_r16 = {
    "n": [], "r0mean": [], "r0sig": [],
    "sig2mean": [], "sig2sig": [],
}
for ii in splits_r16.keys():
    wdata_r16["n"].append(256 / ii)
    for v1 in "r0", "sig2":
        for v2 in "mean", "sig":
            v = v1 + v2
            wdata_r16[v].append(splits_r16[ii][v])
wdata_r16 = values2arrays(wdata_r16)
wdata_r16["r0mean"] /= 0.5 * true_r0
wdata_r16["r0sig"] /= 0.5 * true_r0

# +
vms_m15 = split_square_in_4(
    normalize(
        make_extended(
            2 * N, powerlaw=2.0 + 1.5, 
            ellip=0.5, theta=45, 
            correlation_length=16,
            randomseed=2021_10_08,
        )
    )
)

splits_m15 = {}
for k in [0, 1, 2, 3, 4, 5]:
    splits_m15[2**k] = split4_and_strucfuncs(vms_m15, k)

# +
fig, ax = plt.subplots(
    figsize=(8, 4),
)

for split in splits_m15.values():
    line = ax.plot(split["r"], split["Bmean"], marker=".")
    c = line[0].get_color()
    ax.fill_between(
        split["r"], 
        split["Bmean"] - split["Bsig"], 
        split["Bmean"] + split["Bsig"],
        color=c, alpha=0.1, linewidth=0, zorder=-1,
    )
#ax.plot(r, Bm, linewidth=4, color="k")

rgrid = np.logspace(0.0, 2.0)

for scale in 0.025, 0.1:
    ax.plot(rgrid, scale * rgrid**1.5, linestyle="dotted", color="k")

_split = splits_m15[1]
true_r0_m15 = np.interp(1.0, _split["Bmean"][:-4], _split["r"][:-4])

ax.axhline(1.0, color="k", linestyle="dotted")
ax.axvline(true_r0_m15, color="k", linestyle="dotted")
ax.set(
    xscale="log", yscale="log",
    ylim=[0.02, 4],
)
sns.despine()
fig.tight_layout();
# -

wdata_m15 = {
    "n": [], "r0mean": [], "r0sig": [],
    "sig2mean": [], "sig2sig": [],
}
for ii in splits_m15.keys():
    wdata_m15["n"].append(256 / ii)
    for v1 in "r0", "sig2":
        for v2 in "mean", "sig":
            v = v1 + v2
            wdata_m15[v].append(splits_m15[ii][v])
wdata_m15 = values2arrays(wdata_m15)
wdata_m15["r0mean"] /= true_r0_m15
wdata_m15["r0sig"] /= true_r0_m15

# +
vms_m07 = split_square_in_4(
    normalize(
        make_extended(
            2 * N, powerlaw=2.0 + 0.666, 
            ellip=0.5, theta=45, 
            correlation_length=16,
            randomseed=2021_10_08,
        )
    )
)

splits_m07 = {}
for k in [0, 1, 2, 3, 4, 5]:
    splits_m07[2**k] = split4_and_strucfuncs(vms_m07, k)

# +
fig, ax = plt.subplots(
    figsize=(8, 4),
)

for split in splits_m07.values():
    line = ax.plot(split["r"], split["Bmean"], marker=".")
    c = line[0].get_color()
    ax.fill_between(
        split["r"], 
        split["Bmean"] - split["Bsig"], 
        split["Bmean"] + split["Bsig"],
        color=c, alpha=0.1, linewidth=0, zorder=-1,
    )
#ax.plot(r, Bm, linewidth=4, color="k")

rgrid = np.logspace(0.0, 2.0)

for scale in 0.15, 0.6:
    ax.plot(rgrid, scale * rgrid**0.666, linestyle="dotted", color="k")

_split = splits_m07[1]
true_r0_m07 = np.interp(1.0, _split["Bmean"][:-4], _split["r"][:-4])

ax.axhline(1.0, color="k", linestyle="dotted")
ax.axvline(true_r0_m07, color="k", linestyle="dotted")
ax.set(
    xscale="log", yscale="log",
    ylim=[0.02, 4],
)
sns.despine()
fig.tight_layout();
# -

wdata_m07 = {
    "n": [], "r0mean": [], "r0sig": [],
    "sig2mean": [], "sig2sig": [],
}
for ii in splits_m07.keys():
    wdata_m07["n"].append(256 / ii)
    for v1 in "r0", "sig2":
        for v2 in "mean", "sig":
            v = v1 + v2
            wdata_m07[v].append(splits_m07[ii][v])
wdata_m07 = values2arrays(wdata_m07)
wdata_m07["r0mean"] /= true_r0_m07
wdata_m07["r0sig"] /= true_r0_m07


# Empirical fit to the finite-box effect.  This turns out to be very simple:

def finite_box_effect(L_over_r0, scale=3.6):
    return 1 - np.exp(-L_over_r0 / scale)


# +
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    sharex=True, sharey=True,
    figsize=(8, 8),
)
for data, corr_length in [
    [wdata_r16, true_r0_r16],
    [wdata, true_r0], 
    [wdata_m15, true_r0_m15], 
    [wdata_m07, true_r0_m07], 
]:
    x = data["n"] / corr_length
    ax1.plot(x, data["r0mean"], linestyle="none", marker="o")
    ax1.fill_between(
        x,
        data["r0mean"] - data["r0sig"],
        data["r0mean"] + data["r0sig"],
        alpha=0.15, lw=0, zorder=-1,
    )
    ax2.plot(x, data["sig2mean"], linestyle="none", marker="o")
    ax2.fill_between(
        x,
        data["sig2mean"] - data["sig2sig"],
        data["sig2mean"] + data["sig2sig"],
        alpha=0.15, lw=0, zorder=-1,
    )
xgrid = np.logspace(-0.5, 1.7, 200)
for ax in [ax1, ax2]:
    ax.axvline(1.0, color="k", linestyle="dotted")
    ax.axhline(1.0, color="k", linestyle="dotted")
    ax.plot(
        xgrid, 
        finite_box_effect(xgrid, scale=3.6), 
        color="k", linestyle="dashed",
    )

    
ax1.set(
    ylabel=r"Apparent $r_0$ / true $r_0$",
)
ax2.set(
    xscale="log",
    #yscale="log",
    #xlim=[1, 300],
    #ylim=[0, None],
    xlabel=r"Box size / correlation length: $L\, /\, r_0$",
    ylabel=r"Apparent $\sigma^2$ / true $\sigma^2$",
)
sns.despine()
fig.tight_layout()
fig.savefig("fake-finite-box-effect.pdf");
# -

# We find that the same function: $1 - \exp(-x / 3.6)$ is an adequate fit to both the $r_0$ and the $\sigma^2$ behaviors.

# ## Fake emissivity and velocity cubes
#
# There is not such a rush to do these now, given that the pure velocity maps worked so well.


