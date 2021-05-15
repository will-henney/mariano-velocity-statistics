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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Analysis of Tarantula surface brightness fluctuations

from pathlib import Path
import numpy as np
from astropy.io import fits
import astropy.units as u
from matplotlib import pyplot as plt
import turbustat.statistics as tss

# Load moment data for Ha line.

fitsfilename = {
    "ha": "GAUS_Ha6562.8_060_Will.fits",
    "nii": "GAUS_NII6583.45_060_Will.fits",
}
datadir = Path("data/Tarantula/MUSE_R136toWill")
hdulist = fits.open(datadir / fitsfilename["ha"])
hdulist.info()


sb = hdulist[1].data

# ## PDF and spatial power spectrum

# ### Find PDF of Hα surface brightness
#
# Normalize the surface brightness map by its mean. 

hdulist[1].data /= np.nanmean(hdulist[1].data)

# First, we try the unweighted PDF.

pdf_mom0 = tss.PDF(hdulist[1], min_val=0.0, bins=None)

plt.figure(figsize=(12, 6))
pdf_mom0.run(verbose=True)

# Despite the appearances, this is a log-normal PDF (dashed line) that is fitted to the observations (blue solid line). The reason that it looks a bit strange in the left-hand plot is that they are using uniform bin sizes on a linear scale. This gives a $x^{-1}$ term in the PDF of $x$, which skews the sistribution to the left.  Also they are plotting it on a linear scale in $x$, but a logarithmic scale in the PDF ($y$-axis).  This is probably because star-formation people are most interested in the high-density end of the PDF, which is what this emphasizes. 
#
# The cumulatitive distribution function plot (right hand panel) is better suited to our purposes. Looking at the fit, it is clearly not optimised to the peak of the PDF (CDF of 0.5).  We could do a better "fit" to this part by using the median and the inter-quartile range to uniquely determine the $\mu$ and $\sigma$, respectively, of the log-normal. This would move the $\mu$ slightly to the right and possibly reduce the $\sigma$ slightly.
#
# Anyway, using the fitted PDF, the peak is at 0.72 and the $\sigma$ is 1.1. 

# Now I look at the scipy.stats implementation of the log-normal PDF.  The docs are very confusing, but I think I have sorted it out.

import seaborn as sns
sns.set_color_codes()
sns.set_context("talk")

from scipy.stats import lognorm

LN = lognorm(s=1.0, scale=np.exp(1.0))

x = np.logspace(-2.0, 2.0, 300)
fig, ax = plt.subplots()
ax.plot(x, LN.pdf(x))
ax.set(xscale="log")

# Now I replot the PDF myself, using the empirical PDF and the model fit that turbustat has fuond.

fig, ax = plt.subplots()
s, scale = pdf_mom0.model_params
LN = lognorm(s=s, scale=scale)
x = pdf_mom0.bins
ax.plot(x, x*pdf_mom0.pdf)
ax.plot(x, x*LN.pdf(x))
ax.set(
    xlabel="intensity",
    ylabel="PDF",
    xscale="log",
    ylim=[0, None],
);

pdf_mom0.model_params

# I have multiplied the PDF by $x$ to put it in the form per uniform logarithmic interval.  This is the only way that it actually looks like a Gaussian. 

fig, ax = plt.subplots()
ax.plot(x, x*pdf_mom0.pdf)
ax.plot(x, x*LN.pdf(x))
ax.set(
    xlabel="intensity",
    ylabel="PDF",
    xscale="log",
    yscale="log",
    ylim=[1e-3, 1.0],
);

# Then I plot the same thing on a log-log scale.  When you compare this with molecular hydrogen column density PDFs, such as in Dib:2020m, it doesn't look very impressive.  They have an excellent fit to a log-normal PDF on the low side, down to about 1e-5 in the histogram, whereas we have a much worse fit. 

# #### Weighted PDF using turbustats
#
# Next, we fit the weighted PDF, using the brightness itself as weight.  So the result is no longer the fraction of the POS area that has each intensity.  But instead is the fraction of the total flux that has that intensity.

wpdf_mom0 = tss.PDF(hdulist[1], min_val=0.0, bins=np.logspace(-2.4, 1.6, 50), weights=hdulist[1].data)
plt.figure(figsize=(12, 6))
wpdf_mom0.run(verbose=True)

# And plot the weighted pdf on a log scale as above:

fig, ax = plt.subplots()
s, scale = wpdf_mom0.model_params
LN = lognorm(s=s, scale=scale)
x = wpdf_mom0.bins
ax.plot(x, x*wpdf_mom0.pdf)
ax.plot(x, x*LN.pdf(x))
ax.set(
    xlabel="intensity",
    ylabel="PDF",
    xscale="log",
    ylim=[0, None],
);

wpdf_mom0.model_params

# This is a suprising appearance for the weighted version.  It has a larger width than the unweighted one.  **I don't understand what turbustats means by weighted in this context, so we can't rely on this**

# #### Calculate the PDF by other means

# First, look at the weighted PDF using seaborn:

m = np.isfinite(sb) & (sb > 0.0)
sns.histplot(x=np.log(sb[m]), kde=False, weights=sb[m].astype(float), bins=100)

# So that is very different from the turbustats version, and looks more convincing. Now, we check this by doing the same thing by hand with `np.histogram`:

H, edges = np.histogram(np.log(sb[m]), weights=sb[m], bins=100, range=[-4.0, 2.5], density=True)

fig, ax = plt.subplots()
centers = 0.5*(edges[:-1] + edges[1:])
ax.plot(centers, H)
LN = lognorm(s=0.75,scale=1.8)
ax.plot(centers, np.exp(centers)*LN.pdf(np.exp(centers)))
ax.set(
    xlabel="$\ln (S/S_0)$",
    ylabel="PDF",
#    yscale="log",
#    ylim=[1e-3, 1.0],
)


# That is exactly the same as the seaborn version, and makes perfect sense: the distribution is pulled up at the high brightness because of the brightness weighting.  Note that the log-normal fit is by eye.  
#
# Now we look at the CDF:

# +
cdf = np.cumsum(H)*(centers[1] - centers[0])
fit = np.exp(centers)*LN.pdf(np.exp(centers))
cdf_fit = np.cumsum(fit)*(centers[1] - centers[0])

fig, ax = plt.subplots()
ax.plot(centers, cdf)
ax.plot(centers, cdf_fit)
ax.set(
    xlabel="$\ln (S/S_0)$",
    ylabel="CDF",
#    yscale="log",
#    ylim=[1e-3, 1.0],
)
# -

# So that looks reasonable.  The log-normal fit is not very realistic, since it doesn't account for the skewness of the distribution.  The observed distribution has an excess at low brightness and a deficit at high brightness, as compared with the fit. 

fig, ax = plt.subplots()
ax.plot(centers, cdf/(1 - cdf))
ax.plot(centers, cdf_fit/(1 - cdf_fit))
ax.set(
    xlabel="$\ln (S/S_0)$",
    ylabel="CDF / (1 $-$ CDF)",
    yscale="log",
    ylim=[1e-3/3, 3e3],
)

# This is an experiment with plotting $\mathrm{CDF}/(1 - \mathrm{CDF})$ on a log scale, which gives better visibility to the low and high-intensity wings and emphasizes the disagreement with the log-normal model. We have to be careful with interpreting the CDF graphs, since the sense of a deviation flips when you go past the half-way point.  The blue line is above the orange line at high and low intensities, but this only indicates an excess at low intensities.  At high intensities it indicates a *deficit* (observations below model). 

# #### Conclusions on the weighted PDF
#
# The width of the PDF is $\sigma \approx 0.75$, which is 50% larger than we found in Orion. 

# ### Spatial power spectrum of Hα surface brightness

pspec = tss.PowerSpectrum(hdulist[1])

pspec.run(verbose=True)

plt.figure(figsize=(14, 8))
pspec.plot_fit(show_2D=True)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(pspec.freqs, pspec.freqs**3 * pspec.ps1D, ".")
ax.set(
    xscale="log", yscale="log",
)



plt.figure(figsize=(14, 8))
pspec.run(
    verbose=True, 
    apodize_kernel="tukey", alpha=0.3,
    fit_2D=False, 
    low_cut=0.003/ u.pix,
    high_cut=0.2/ u.pix,
    fit_kwargs={"brk": 0.02 / u.pix, "log_break": False},
)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(pspec.freqs, pspec.freqs**3 * pspec.ps1D, ".")
ax.set(
    xscale="log", yscale="log",
)

pspec_v = tss.PowerSpectrum(hdulist[2])

plt.figure(figsize=(14, 8))
pspec_v.run(
    verbose=True, 
    apodize_kernel="splitcosinebell", alpha=0.5, beta=0.2,
    fit_2D=False, 
    low_cut=0.008/ u.pix,
    high_cut=0.1/ u.pix,
    #fit_kwargs={"brk": 0.02 / u.pix, "log_break": False},
)

fig, ax = plt.subplots(figsize=(12, 6))
pspec_v.freqs[0] = np.nan
ax.plot(pspec_v.freqs, pspec_v.freqs**3 * pspec_v.ps1D, ".", color="r")
ax.set(
    xscale="log", yscale="log",
)

hdulist_n2 = fits.open(datadir / fitsfilename["nii"])

m = hdulist_n2[1].data > 6e4
hdulist_n2[1].data[m] = np.median(hdulist_n2[1].data)

pspec_n2 = tss.PowerSpectrum(hdulist_n2[1])

plt.figure(figsize=(14, 8))
pspec_n2.run(
    verbose=True, 
    apodize_kernel="tukey", alpha=0.3,
    fit_2D=False, 
    low_cut=0.006/ u.pix,
    high_cut=0.1/ u.pix,
    fit_kwargs={"brk": 0.02 / u.pix, "log_break": False},
)

fig, ax = plt.subplots(figsize=(12, 6))
pspec_n2.freqs[0] = np.nan
ax.plot(pspec_n2.freqs, pspec_n2.freqs**3 * pspec_n2.ps1D, ".", color="g")
ax.set(
    xscale="log", yscale="log",
)

# ## Delta variance of the intensity and velocity maps
#
# In principle, the delta variance is quite similar to the structure function in the sense that it is measuring the variance as a function of scale.  It is generally applied to column densities, but there is nothing stopping us applying it to the velocity field as well.  
#
# Acording to the case stiudies in the turbustats documentation, it deals with edge effects in non-periodic data sets much better than the power law does.

dvar = tss.DeltaVariance(hdulist[1])

plt.figure(figsize=(14, 8))
dvar.run(verbose=True, boundary="fill", xlow=3*u.pix, xhigh=100*u.pix, brk=30*u.pix)

plt.figure(figsize=(14, 8))
dvar.plot_fit()

# This looks very similar to the structure function!
#
# Now try it on the velocities.  We have to tidy up the data first, since it contains NaNs, which turbustat doesn't like. Replace all bad pixels with the median velocity:

vmed = np.nanmedian(hdulist[2].data)
m = np.isfinite(hdulist[2].data)
hdulist[2].data[~m] = vmed

dvar_v = tss.DeltaVariance(hdulist[2])

plt.figure(figsize=(14, 8))
dvar_v.run(verbose=True, boundary="fill", xlow=4*u.pix, xhigh=100*u.pix, brk=30*u.pix)

# Now, we will try normalizing everything:

vmean = np.mean(hdulist[2].data)
vsig = np.std(hdulist[2].data)
print(vmean, vsig)
dv = (hdulist[2].data - vmean) / vsig

ln_S = np.log(hdulist[1].data)
ln_S[~np.isfinite(ln_S)] = np.nanmedian(ln_S)
ln_S_mean = np.mean(ln_S)
ln_S_sig = np.std(ln_S)
print(ln_S_mean, ln_S_sig)
dlnS = (ln_S - ln_S_mean)/ln_S_sig

dvar_dlnS = tss.DeltaVariance(dlnS, header=hdulist[1].header)

dvar_dv = tss.DeltaVariance(dv, header=hdulist[2].header)

dvar_dlnS.compute_deltavar(boundary="fill")

dvar_dv.compute_deltavar(boundary="fill")


# Compare the 

def bfunc(r, r0, sig2, m):
    "Theoretical structure function"
    C = 1.0 / (1.0 + (r/r0)**m)
    return 2.0*sig2*(1 - C)



# +
fig, ax = plt.subplots(figsize=(10, 8))
NORM = 4*np.pi
PIX = 0.05 # pixel scale in pc
#ax.plot(dvar_dlnS.lags, NORM*dvar_dlnS.delta_var, "o", label="ln S")
ax.plot(PIX*1.02*dvar_dv.lags, NORM*dvar_dv.delta_var, "o", label="V")
ax.plot(PIX*0.98*dvar.lags, NORM*dvar.delta_var, "o", label="S")
ax.axhline(1.0, lw=0.5, ls="--", color="k")
ax.axhline(0.5, lw=0.5, ls="--", color="k")

x = np.logspace(-1.0, 1.3, 200)
ax.plot(x, bfunc(x, r0=2.1, sig2=0.58, m=1.4))

ax.legend()
ax.set(
    xscale="log",
    yscale="log",
    xlabel="Lag, pc",
    ylabel=r"$\Delta$-variance $\times 4\pi$",
);
# -

# Look at the PDF of velocity differences for different lags.  However, this is going to run into problems round the edges since the data isn't periodic

# +
vmap = dv

dv_001 = vmap - np.roll(vmap, (4, 4), axis=(0, 1))
m = (dv_001 == 0.0) | (np.abs(dv_001) > 8.0)
dv_001[m] = np.nan

dv_010 = vmap - np.roll(vmap, (16, 16), axis=(0, 1))
m = (dv_010 == 0.0) | (np.abs(dv_010) > 8.0)
dv_010[m] = np.nan

dv_100 = vmap - np.roll(vmap, (64, 64), axis=(0, 1))
m = (dv_100 == 0.0) | (np.abs(dv_100) > 8.0)
dv_100[m] = np.nan

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12,4))
kwds = dict(
    origin="lower", 
    vmin=0.0, 
    vmax=3.0, 
    cmap="gray_r"
)
axes[0].imshow(np.abs(dv_001), **kwds)
axes[1].imshow(np.abs(dv_010), **kwds)
axes[2].imshow(np.abs(dv_100), **kwds)

# +
lag = 4
ndv1 = dv_001[lag:-lag, lag:-lag].ravel()
sig1 = ndv1.std()
ndv1 /= sig1
label1 = fr"$\ell = {lag}$ pix, $\sigma = {sig1:.2f} \sigma_0$"

lag = 16
ndv2 = dv_010[lag:-lag, lag:-lag].ravel()
sig2 = ndv2.std()
ndv2 /= sig2
label2 = fr"$\ell = {lag}$ pix, $\sigma = {sig2:.2f} \sigma_0$"


lag = 64
ndv3 = dv_100[lag:-lag, lag:-lag].ravel()
sig3 = ndv3.std()
ndv3 /= sig3
label3 = fr"$\ell = {lag}$ pix, $\sigma = {sig3:.2f} \sigma_0$"

xmin, xmax = -10.0, 10.0
xx = np.linspace(xmin, xmax, 200)
# Gaussian profile
gg = np.exp(-0.5 * xx**2) / np.sqrt(np.pi)

fig, ax = plt.subplots(figsize=(12,8))
sns.histplot(
    {
        label3: ndv3, 
        label2: ndv2, 
        label1: ndv1, 
    },
    bins=100,
    stat="density",
    multiple="stack",
    common_norm=False,
    common_bins=True,
    ax=ax,
    palette="tab10",
    alpha=0.7,
)
ax.plot(xx, gg)
ax.set(
    xlim=[xmin, xmax],
    ylim=[1e-5, 1.0],
    yscale="log",
    xlabel="$\Delta v / \sigma$",
);
# -

# So this is now plotted against the velocity difference normalised by the sigma at each lag. This boosts the wings of the low-lag results, since it is easier to get to 5-sigma when sigma is small!

import pandas as pd

ser = pd.Series(data=dv.ravel(), name="dv")

ser.describe()

np.percentile(dv, (0.001, 0.01, 0.1, 1.0, 99.0, 99.9, 99.99, 99.999))

sig1, np.percentile(ndv1, (0.001, 0.1, 99.9, 99.999))

sig2, np.percentile(ndv2, (0.001, 0.1, 99.9, 99.999))

sig3, np.percentile(ndv3, (0.001, 0.1, 99.9, 99.999))

# We want to improve this by using all the lags of a certain scale size. 

import itertools


def find_all_lags(s0, d_log_s0=0.15, only_positive_x=True):
    """Find all the integer 2D pixel lags (i, j) with approx length `s0`
    
    Optional parameter `d_log_s0` is the base-10 logarithmic interval around `s0`
    If `only_positive_x` is True, then only lags with j >= 0 are returned
    """
    smin = s0 * 10**(-d_log_s0/2)
    smax = s0 * 10**(d_log_s0/2)
    imax = int(smax) + 1
    yrange = range(-imax, imax)
    xrange = range(imax) if only_positive_x else range(-imax, imax)
    lags = filter(
        lambda x: smin <= np.hypot(*x) < smax, 
        itertools.product(yrange, xrange)
    )
    return list(lags)


find_all_lags(3)

find_all_lags(1)

find_all_lags(2)

# So that seems to work fine for small separations

len(find_all_lags(4)), len(find_all_lags(16)), len(find_all_lags(64))

# For larger separations, the number of lags increases significantly.  We could maybe take a random sample of them. 

fig, ax = plt.subplots()
width = {
    1: 0.5, 4: 0.25, 16: 0.15
}
for s0 in [1, 4, 16][::-1]:
    pts = find_all_lags(s0, d_log_s0=width[s0], only_positive_x=False)
    _y, _x = zip(*pts)
    ax.scatter(_x, _y, marker=".", ec=None, alpha=0.7)
ax.set_aspect("equal")
sns.despine()

# We need to put the NaNs back into the velocity array.

havmap = fits.open(datadir / fitsfilename["ha"])[2].data

havmap.mean()

x = 4
x is not None and 3 > x

# +
import random

def make_dv_maps(vmap, s0, max_lags=None, d_log_s0=0.15):
    """
    Calculate a series of velocity difference maps for lags of length `s0`
    """
    lags = find_all_lags(s0, d_log_s0=d_log_s0, only_positive_x=False)
    
    # If we have more lags than desired, then choose a random sample of them
    if max_lags is not None and len(lags) > max_lags:
        lags = random.sample(lags, max_lags)
        
    dv_maps = []
    for i, j in lags:
        # difference with 2D lag of (y, x) = (i, j)
        dv = vmap - np.roll(vmap, (i, j), axis=(0, 1))
        # determine invalid border due to wrap-around
        islice = slice(None, i) if i >= 0 else slice(i, None)
        jslice = slice(None, j) if j >= 0 else slice(j, None)
        # blank out the border pixels
        dv[islice, :] = np.nan
        dv[:, jslice] = np.nan
        # save the difference map
        dv_maps.append(dv)
    # return a 3D stack of maps
    return np.stack(dv_maps)


# -

dvmaps = make_dv_maps(havmap, 16.0)
dvmaps.shape

np.nanmean(dvmaps[0, :, :])

advmap = np.nanmedian(np.abs(dvmaps), axis=0)

np.nanmean(advmap)

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10,10))
kwds = dict(
    origin="lower", 
    vmin=0.0, 
    vmax=40.0, 
    cmap="turbo"
)
im = ax.imshow(advmap, **kwds)
c = fig.colorbar(im, ax=ax)
#ax.imshow(np.nanmean(np.abs(dvmaps), axis=0), **kwds)

hdulist[2].data.mean()

m = np.isfinite(dvmaps)
dvmaps[m].shape

dvmaps[m].min()

H, edges = np.histogram(dvmaps[m], range=[-150, 150], bins=100)

H

centers = 0.5*(edges[:-1] + edges[1:])
fig, ax = plt.subplots(1, 1)
ax.plot(centers, H)
ax.set(
    yscale="log",
)

dvmaps004 = make_dv_maps(havmap, 4.0, d_log_s0=0.25)
dvmaps004.shape

advmap004 = np.nanmedian(np.abs(dvmaps004), axis=0)
np.nanmean(advmap004)

dvmaps001 = make_dv_maps(havmap, 1.0, d_log_s0=0.5)
dvmaps001.shape

advmap001 = np.nanmedian(np.abs(dvmaps001), axis=0)
np.nanmean(advmap001)

dvmaps064 = make_dv_maps(havmap, 64.0, max_lags=400)
dvmaps064.shape

advmap064 = np.nanmedian(np.abs(dvmaps064), axis=0)
np.nanmean(advmap064)

sigmap001 = np.nanstd(dvmaps001)
sigmap004 = np.nanstd(dvmaps004)
sigmap016 = np.nanstd(dvmaps)
sigmap064 = np.nanstd(dvmaps064)
sigmap001, sigmap004, sigmap016, sigmap064

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14,12))
kwds = dict(
    origin="lower", 
    vmin=0.0, 
    vmax=40.0, 
    cmap="turbo"
)
im = axes[0, 0].imshow(advmap001, **kwds)
im = axes[0, 1].imshow(advmap004 - 0.8*advmap001, **kwds)
im = axes[1, 0].imshow(advmap - 0.8*advmap001, **kwds)
im = axes[1, 1].imshow(advmap064 - 0.8*advmap001, **kwds)
c = fig.colorbar(im, ax=axes)

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14,12))
kwds = dict(
    origin="lower", 
    vmin=0.0, 
    vmax=4.0, 
    cmap="turbo"
)
im = axes[0, 0].imshow(advmap001 / sigmap001, **kwds)
im = axes[0, 1].imshow((advmap004 - 0.8*advmap001) / sigmap004, **kwds)
im = axes[1, 0].imshow((advmap - 0.8*advmap001) / sigmap016, **kwds)
im = axes[1, 1].imshow((advmap064 - 0.8*advmap001) / sigmap064, **kwds)
c = fig.colorbar(im, ax=axes)
fig.savefig("multi-shear-ha.pdf")

hists = {}
for s0, dvstack in (1, dvmaps001), (4, dvmaps004), (16, dvmaps), (64, dvmaps064):
    m = np.isfinite(dvstack)
    H, edges = np.histogram(dvstack[m], range=[-60, 60], bins=200)
    hists[s0] = H

centers = 0.5*(edges[:-1] + edges[1:])
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for s0 in hists.keys():
    H = hists[s0]
    ax.plot(centers, H/H.sum())
ax.set(
    yscale="log",
    ylim=[1e-4, 1.0],
    xlim=[-60, 60],
)

# Now, normalize by the sigma at each scale

xmin, xmax = -6.0, 6.0
scaled_hists = {}
for s0, dvstack in (1, dvmaps001), (4, dvmaps004), (16, dvmaps), (64, dvmaps064):
    m = np.isfinite(dvstack)
    sig = np.std(dvstack[m])
    H, edges = np.histogram(
        dvstack[m]/sig, 
        range=[xmin, xmax], 
        bins=200,
        density=True,
    )
    scaled_hists[s0] = H

centers = 0.5*(edges[:-1] + edges[1:])

# Fit Gaussian and Lorentzian

# +
from astropy.modeling import models, fitting

fitter = fitting.LevMarLSQFitter()
# Fit to core of s0=4 histogram
H = scaled_hists[4]
mcore = H >= 0.3*H.max()
lfit = fitter(models.Lorentz1D(), centers[mcore], H[mcore])
gfit = fitter(models.Gaussian1D(), centers[mcore], H[mcore])
# -

gfit, lfit

fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 8))
for s0 in scaled_hists.keys():
    if s0 == 1: continue
    H = scaled_hists[s0]
    label = f"lag = {s0}"
    ax1.plot(centers, H, label=label)
    ax2.plot(centers, H, label=label)
gauss = np.exp(-0.5*centers**2) / np.sqrt(2*np.pi)
for ax in ax1, ax2:
    ax.plot(centers, gfit(centers), 
            color="k", ls="--", label="Gauss")
    ax.plot(centers, lfit(centers), 
            color="k", ls=":", label="Lorentz")
    ax.plot(centers, np.exp(-1.7*np.abs(centers)), 
            color="k", ls="-.", label="Exponential")
ax1.set(
    xlim=[-3, 3],
    ylim=[0.0, 0.6],
)
ax2.legend(ncol=3)
ax2.set(
    yscale="log",
    xlim=[xmin, xmax],
    ylim=[1e-4, 30.0],
    xlabel = "$\Delta v / \sigma$",
)
fig.tight_layout(h_pad=0.3);

# Thus shows that the profile is approximately exponential in a large part of the wings, out to $\pm 4$, but then curves up a bit. 
#
# Now plot the same on log-log scale, which shows that the far wings go ax about $|x|^{-4}$, so the sigma is still finite.

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for s0 in scaled_hists.keys():
    if s0 == 1: continue
    H = scaled_hists[s0]
    ax.plot(centers, H)
ax.plot(centers, gfit(centers), color="k", ls="--")
ax.plot(centers, lfit(centers), color="k", ls=":")
ax.plot(centers, centers**-4, color="k", lw=1)
ax.plot(centers, np.exp(-1.7*np.abs(centers)), color="k", ls="-.", lw=1.0)
ax.set(
    xscale="log",
    yscale="log",
    xlim=[0.3, None],
    ylim=[1e-4, 1],
    xlabel = "$\Delta v / \sigma$",
);



# ## Using generated red-noise to simulate two layers

from turbustat.simulator import make_extended
img = make_extended(512, powerlaw=3.67, ellip=0.5, theta=45, randomseed=3)
# Now shuffle so the peak is near the centre
#img = np.roll(img, (128, -30), (0, 1))  
img -= img.min()
img2 = make_extended(512, powerlaw=3.67, ellip=0.5, theta=135, randomseed=99)
img2 -= img2.min()
imap_2l = img**2 + img2**2
plt.figure(figsize=(10, 10))
plt.imshow(imap_2l, origin='lower')  
plt.colorbar()  

vmap_2l = (img**2 - img2**2)/(img**2 + img2**2)
plt.figure(figsize=(10, 10))
plt.imshow(vmap_2l, origin='lower', cmap="coolwarm")  
plt.colorbar()  

tss.PDF(fits.PrimaryHDU(imap_2l), min_val=0.0, bins=None).run(verbose=True)

vmap = vmap_2l
dv_001 = vmap - np.roll(vmap, (4, -4), axis=(0, 1))
dv_010 = vmap - np.roll(vmap, (16, -16), axis=(0, 1))
dv_100 = vmap - np.roll(vmap, (128, -128), axis=(0, 1))
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12,4))
kwds = dict(
    origin="lower", 
    vmin=0.0, 
    vmax=1.0, 
    cmap="gray_r"
)
axes[0].imshow(np.abs(dv_001), **kwds)
axes[1].imshow(np.abs(dv_010), **kwds)
axes[2].imshow(np.abs(dv_100), **kwds)

fig, ax = plt.subplots(figsize=(12,8))
sns.histplot(
    {
        "lag: (4, 4)": dv_001.ravel(), 
        "lag: (16, 16)": dv_010.ravel(), 
        "lag: (64, 64)": dv_100.ravel(), 
    },
    bins=100,
    stat="density",
    common_norm=False,
    common_bins=True,
    ax=ax,
)
ax.set( 
    #xlim=[-2.0, 2.0],
    yscale="log"
);

# This looks very similar to the observations! Very fat tails to the distributions.

# ## Is delta variance affected by projection smoothing?
#
# Projection smoothing operates at scales smaller than the line-of-sight depth through a turbulent region and means that the 2D structure function slope becomes steeper than the 3D structure function slope.  So it becomes $\beta - 3$ instead of $\beta - 2$. Where $\beta$ is the slope of the power spectrum: $P(k) \propto k^{-\beta}$.  This is the notation of Ossenkopf:2006a. 
#
# We want to check that his really produces a change in the slope of the structure function, and also to check whether we see anything similar in the delta-variance.  From what everyone says, the delta-variance should mot suffer from the same problem, but I am not sure I beleive it.  
#
# Also, we can check the difference between steep and shallow spectra.  Ossenpof:2006a imply that it is only for shallow spectra that the autocorrelation function is a poer law.  

# We can investigate this using the 3D fBM functions of turbustats.

from turbustat.simulator import make_3dfield

threeD_field = make_3dfield(256, powerlaw=2.5)

deep_vmap = np.mean(threeD_field, axis=0)
koffset = 64
shallow_vmap = np.mean(threeD_field[koffset:32 + koffset, :, :], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(deep_vmap, vmin=-1.5, vmax=1.5, cmap="seismic")
axes[1].imshow(shallow_vmap, vmin=-1.5, vmax=1.5, cmap="seismic")

dvar_deep = tss.DeltaVariance(fits.PrimaryHDU(deep_vmap))

dvar_shallow = tss.DeltaVariance(fits.PrimaryHDU(shallow_vmap))

dvar_deep.run(verbose=True)

dvar_shallow.run(verbose=True)

# ### Statistics of sigmas and how it varies with power law steepness

deep_vmap.std()

shallow_vmap.std()

threeD_field[:, 0, 0].std()

deep_sig_los = threeD_field[:, :, :].std(axis=0)
shallow_sig_los = threeD_field[koffset:32 + koffset, :, :].std(axis=0)

deep_sig_los.mean(), deep_sig_los.std()

shallow_sig_los.mean(), shallow_sig_los.std()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].imshow(shallow_sig_los, vmin=0.0, vmax=1.2)
c0 = fig.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(deep_sig_los, vmin=0.0, vmax=1.2)
c1 = fig.colorbar(im1, ax=axes[1])

# So, this is interesting.  For the shallow velocity spectrum, $k=2$, the LOS sigma does not show much variation, either in the thick or the thin volume. 
#
# Whereas, with the steep velocity spectrum, $k=4$, we see a broad PDF of the LOS sigma
#
# **We need to sort out the terminology better.  We can use deep-vs-shallow for LOS thickness, or we can use steep-vs-shallow for power spectrum slope, but we shouldn't use shallow for both**

sns.histplot(
    {"thin": shallow_sig_los.ravel(), "thick": deep_sig_los.ravel()}
)
plt.gca().set(xlim=[0.0, 2.0], yscale="log");

# This is the histogram of the LOS sigmas, which for the shallow spectrum becomes narrow, with a mean of just less than 1, especially for the thick cloud.  
#
# When we plot it on a log axis, we see that the distributions are slightly skew, especially for the thin slice, and the tails are a bit fat. 

sns.histplot(
    {
        "thin": shallow_vmap.ravel(), 
        "thick": deep_vmap.ravel(), 
        #"3D": threeD_field.ravel(),
    },
    stat="density",
    common_norm=False,
)
plt.gca().set(xlim=[-2.5, 2.5], yscale="log");

# And this is a histogram of the centroid velocities, compared with the 3D velocities in green.  For the shallow spectrum, the distribution is narrower than that of the full cube. 
#
# There is no evidence of skewness or of fat tails.

# ### Structure function from simulated maps

import sys
sys.path.append("../muse-strucfunc")

import strucfunc

sf = strucfunc.strucfunc_numba_parallel(shallow_vmap)
sf

sig0 = shallow_vmap.std()
fig, ax = plt.subplots()
ax.plot("log10 r", "Unweighted B(r)", "o", data=sf)
ax.axhline(sig0**2)
ax.axhline(2*sig0**2)
ax.set(
    yscale="log",
)

sfd = strucfunc.strucfunc_numba_parallel(deep_vmap)
sfd

sig0 = deep_vmap.std()
fig, ax = plt.subplots()
ax.plot(10**sfd["log10 r"], sfd["Unweighted B(r)"], "o", label="thick")
ax.plot(10**sf["log10 r"], sf["Unweighted B(r)"], "o", label="thin")
ax.axhline(sig0**2)
ax.axhline(2*sig0**2)
xx = np.logspace(0.0, 2.0)
yy = 0.01 * xx**0.5
ax.plot(xx, yy)
ax.legend()
ax.set(
    xscale="log",
    yscale="log",
)

# ### Look at PDF of velocity differences at different lags
#
# We can use `np.roll` to quickly look at velocity differences at different lags. 

dv_001 = deep_vmap - np.roll(deep_vmap, (-1, 1), axis=(0, 1))
dv_010 = deep_vmap - np.roll(deep_vmap, (10, 10), axis=(0, 1))
dv_100 = deep_vmap - np.roll(deep_vmap, (100, 100), axis=(0, 1))
fig, ax = plt.subplots(figsize=(12,8))
sns.histplot(
    {
        "lag: (1, 1)": dv_001.ravel(), 
        "lag: (10, 10)": dv_010.ravel(), 
        "lag: (100, 100)": dv_100.ravel(), 
    },
    stat="density",
    common_norm=False,
    ax=ax,
)
ax.set(
    xlim=[-2.0, 2.0],
    yscale="log"
);

dv_001 = shallow_vmap - np.roll(shallow_vmap, (1, 1), axis=(0, 1))
dv_010 = shallow_vmap - np.roll(shallow_vmap, (10, 10), axis=(0, 1))
dv_100 = shallow_vmap - np.roll(shallow_vmap, (100, 100), axis=(0, 1))
fig, ax = plt.subplots(figsize=(12,8))
sns.histplot(
    {
        "lag: (1, 1)": dv_001.ravel(), 
        "lag: (10, 10)": dv_010.ravel(), 
        "lag: (100, 100)": dv_100.ravel(), 
    },
    stat="density",
    common_norm=False,
    ax=ax,
)
ax.set(
    xlim=[-3.0, 3.0], 
    yscale="log"
);


