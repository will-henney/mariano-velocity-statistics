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

# ## Using generated red-noise to simulate two layers

from turbustat.simulator import make_extended
img = make_extended(1024, powerlaw=4.0, ellip=0.5, theta=45, randomseed=3)
# Now shuffle so the peak is near the centre
#img = np.roll(img, (128, -30), (0, 1))  
img -= img.min()
img2 = make_extended(1024, powerlaw=4.0, ellip=0.5, theta=135, randomseed=99)
img2 -= img2.min()
plt.figure(figsize=(10, 10))
plt.imshow(img**2 + img2**2, origin='lower')  
plt.colorbar()  

plt.figure(figsize=(10, 10))
plt.imshow((img**2 - img2**2)/(img**2 + img2**2), origin='lower', cmap="coolwarm")  
plt.colorbar()  

tss.PDF(fits.PrimaryHDU(img**2 + img2**2), min_val=0.0, bins=None).run(verbose=True)

# ## Is delta variance affected by projection smoothing?
#
# Projection smoothing operates at scales smaller than the line-of-sight depth through a turbulent region and means that the 2D structure function slope becomes steeper than the 3D structure function slope.  So it becomes $\beta - 3$ instead of $\beta - 2$. Where $\beta$ is the slope of the power spectrum: $P(k) \propto k^{-\beta}$.  This is the notation of Ossenkopf:2006a. 
#
# We want to check that his really produces a change in the slope of the structure function, and also to check whether we see anything similar in the delta-variance.  From what everyone says, the delta-variance should mot suffer from the same problem, but I am not sure I beleive it.  
#
# Also, we can check the difference between steep and shallow spectra.  Ossenpof:2006a imply that it is only for shallow spectra that the autocorrelation function is a poer law.  

# We can investigate this using the 3D fBM functions of turbustats.

from turbustat.simulator import make_3dfield

threeD_field = make_3dfield(128, powerlaw=2.5)

deep_vmap = np.mean(threeD_field, axis=0)
koffset = 64
shallow_vmap = np.mean(threeD_field[koffset:16 + koffset, :, :], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(deep_vmap)
axes[1].imshow(shallow_vmap)

dvar_deep = tss.DeltaVariance(fits.PrimaryHDU(deep_vmap))

dvar_shallow = tss.DeltaVariance(fits.PrimaryHDU(shallow_vmap))

dvar_deep.run(verbose=True)

dvar_shallow.run(verbose=True)

# ### Statistics of sigmas and how it varies with power law steepness

deep_vmap.std()

shallow_vmap.std()

threeD_field[:, 0, 0].std()

deep_sig_los = threeD_field[:, :, :].std(axis=0)
shallow_sig_los = threeD_field[koffset:16 + koffset, :, :].std(axis=0)

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
plt.gca().set(xlim=[0.0, 1.4]);

# This is the histogram of the LOS sigmas, which for the shallow spectrum becomes narrow, with a mean of just less than 1, especially for the thick cloud.  

sns.histplot(
    {
        "thin": shallow_vmap.ravel(), 
        "thick": deep_vmap.ravel(), 
        "3D": threeD_field.ravel(),
    },
    stat="density",
    common_norm=False,
)
plt.gca().set(xlim=[-2.5, 2.5]);

# And this is a histogram of the centroid velocities, compared with the 3D velocities in green.  For the shallow spectrum, the distribution is narrower than that of the full cube. 

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


