import sys
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import json
import numpy as np
from astropy.io import fits
sys.path.append("../muse-strucfunc")
import strucfunc

try:
    LINEID = sys.argv[1]
except:
    LINEID = "ha"

try:
    METHOD = sys.argv[2]
except:
    METHOD = "standard"

USE_COLDEN = "colden" in METHOD
USE_DEPTH = "depth" in METHOD

fitsfilename = {
    "ha": "GAUS_Ha6562.8_060_Will.fits",
    "nii": "GAUS_NII6583.45_060_Will.fits",
}
wav0 = {"ha": 6562.8, "nii": 6583.45}
atm_wt = {"ha": 1.0, "nii": 14.0}
fs_var = {"ha": 10.233, "nii": 0.0}
# Assume 1e4 K for thermal broadening
thermal_var = 82.5 / atm_wt[LINEID]

datadir = Path("data/Tarantula/MUSE_R136toWill")

hdulist = fits.open(datadir / fitsfilename[LINEID])

n = None
sb = hdulist[1].data[:n, :n].astype(np.float64)
vv = hdulist[2].data[:n, :n].astype(np.float64)
ss = hdulist[3].data[:n, :n].astype(np.float64)

# optionally use column density, instead of surface brightness
if USE_COLDEN:
    dd = fits.open(datadir / "Density.fits")["DATA"].data[:n, :n].astype(np.float64)
    sb /= dd
if USE_DEPTH:
    dd = fits.open(datadir / "Density.fits")["DATA"].data[:n, :n].astype(np.float64)
    sb /= dd**2


# Convert sigma to km/s
ss *= 3e5 / wav0[LINEID]

# Subtract instrumental width and thermal width 
ss = np.sqrt(ss**2 - 48.0**2 - fs_var[LINEID] - thermal_var)

# Replace spurious values in the arrays
m = ~np.isfinite(sb*vv*ss) | (sb < 0.0)
if LINEID == "nii":
    # Remove bad patch from the [N II] map
    m = m | (sb > 6e4) 


m = ~m                          # invert mask

# additional mask for bright pixels
# BRIGHT_THRESHOLD = 0.1*np.max(sb[m])
BRIGHT_THRESHOLD = np.median(sb[m])
mb = sb > BRIGHT_THRESHOLD

# Brightness-weighted average sigma
AV_SIG = np.average(ss[m], weights=sb[m])

NBIN = 100
BMAX = np.max(1.2*sb[m])
BMIN = BMAX / 1000.0
if USE_COLDEN:
    BMAX = 5*BRIGHT_THRESHOLD
    BMIN = BMAX / 100.0
if USE_DEPTH:
    BMAX = 10*BRIGHT_THRESHOLD
    BMIN = BMAX / 500.0
SMIN, SMAX = 0.0, 90.0
VMIN, VMAX = 220.0, 330.0
GAMMA = 1.5

vlabel = "Centroid velocity, km/s"
slabel = "RMS line width, km/s"
blabel = "log10(Surface brightness)"
if USE_COLDEN:
    blabel = "log10(Column density)"
if USE_DEPTH:
    blabel = "log10(LOS depth)"

fig, axes = plt.subplots(2, 2)

linestyle = dict(lw=0.7, ls="--", color="r", alpha=0.5)

# I - sigma
xmin, xmax = np.log10(BMIN), np.log10(BMAX)
ymin, ymax = SMIN, SMAX
H, xedges, yedges = np.histogram2d(
    np.log10(sb[m]), ss[m], 
    bins=[NBIN, NBIN],
    range=[[xmin, xmax], [ymin, ymax]],
)
axes[0, 0].imshow(
    (H.T)**(1.0/GAMMA), 
    extent=[xmin, xmax, ymin, ymax], 
    interpolation='none', aspect='auto', 
    origin='lower', cmap=plt.cm.gray_r,
)
# Show brightness thereshold
axes[0, 0].axvline(np.log10(BRIGHT_THRESHOLD), **linestyle)
# Show average sigma
axes[0, 0].axhline(AV_SIG, **linestyle)
axes[0, 0].set(
    xlabel=blabel,
    ylabel=slabel,
    xlim=[xmin, xmax],
    ylim=[ymin, ymax],
)

# I - V
xmin, xmax = np.log10(BMIN), np.log10(BMAX)
ymin, ymax = VMIN, VMAX
H, xedges, yedges = np.histogram2d(
    np.log10(sb[m]), vv[m], 
    bins=[NBIN, NBIN],
    range=[[xmin, xmax], [ymin, ymax]],
)
axes[1, 0].imshow(
    (H.T)**(1.0/GAMMA), 
    extent=[xmin, xmax, ymin, ymax], 
    interpolation='none', aspect='auto', 
    origin='lower', cmap=plt.cm.gray_r,
)
# Show brightness thereshold
axes[1, 0].axvline(np.log10(BRIGHT_THRESHOLD), **linestyle)
axes[1, 0].set(
    xlabel=blabel,
    ylabel=vlabel,
    xlim=[xmin, xmax],
    ylim=[ymin, ymax],
)

# V - sigma
xmin, xmax = VMIN, VMAX
ymin, ymax = SMIN, SMAX
H, xedges, yedges = np.histogram2d(
    vv[m & (~mb)], ss[m & (~mb)], 
    bins=[NBIN, NBIN],
    range=[[xmin, xmax], [ymin, ymax]],
)
# Show average sigma
axes[0, 1].axhline(AV_SIG, **linestyle)
axes[0, 1].imshow(
    (H.T)**(1.0/GAMMA), 
    extent=[xmin, xmax, ymin, ymax], 
    interpolation='none', aspect='auto', 
    origin='lower', cmap=plt.cm.gray_r,
)
axes[0, 1].set(
    xlabel=vlabel,
    ylabel=slabel,
    xlim=[xmin, xmax],
    ylim=[ymin, ymax],
)


# V - sigma but bright only
xmin, xmax = VMIN, VMAX
ymin, ymax = SMIN, SMAX
H, xedges, yedges = np.histogram2d(
    vv[m & mb], ss[m & mb], 
    bins=[NBIN, NBIN],
    range=[[xmin, xmax], [ymin, ymax]],
)
axes[1, 1].imshow(
    (H.T)**(1.0/GAMMA), 
    extent=[xmin, xmax, ymin, ymax], 
    interpolation='none', aspect='auto', 
    origin='lower', cmap=plt.cm.gray_r,
)
# Show average sigma
axes[1, 1].axhline(AV_SIG, **linestyle)
axes[1, 1].set(
    xlabel=vlabel,
    ylabel=slabel,
    xlim=[xmin, xmax],
    ylim=[ymin, ymax],
)

fig.tight_layout()

plotfile = f"tarantula-I-sigma-hist-{LINEID}.png"
if USE_COLDEN:
    plotfile = plotfile.replace(".", "-colden.")
if USE_DEPTH:
    plotfile = plotfile.replace(".", "-depth.")

fig.savefig(plotfile, dpi=200)

print(plotfile, end="")
