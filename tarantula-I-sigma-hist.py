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

fitsfilename = {
    "ha": "GAUS_Ha6562.8_060_Will.fits",
    "nii": "GAUS_NII6583.45_060_Will.fits",
}
wav0 = {"ha": 6562.8, "nii": 6583.45}

datadir = Path("data/Tarantula/MUSE_R136toWill")

hdulist = fits.open(datadir / fitsfilename[LINEID])

n = None
sb = hdulist[1].data[:n, :n].astype(np.float64)
vv = hdulist[2].data[:n, :n].astype(np.float64)
ss = hdulist[3].data[:n, :n].astype(np.float64)

# Convert sigma to km/s
ss *= 3e5 / wav0[LINEID]

# Replace spurious values in the arrays
m = ~np.isfinite(sb*vv*ss) | (sb < 0.0)
if LINEID == "nii":
    # Remove bad patch from the [N II] map
    m = m | (sb > 6e4) 

m = ~m                          # invert mask
fig, ax = plt.subplots()

NBIN = 50
BMAX = np.max(sb[m])
BMIN = BMAX / 1000.0
SMIN, SMAX = 40.0, 100.0
GAMMA = 1.5
xmin, xmax = np.log10(BMIN), np.log10(BMAX)
ymin, ymax = SMIN, SMAX
H, xedges, yedges = np.histogram2d(np.log10(sb[m]), ss[m], 
                                   bins=[NBIN, NBIN],
                                   range=[[xmin, xmax], [ymin, ymax]],
                                   )

ax.imshow((H.T)**(1.0/GAMMA), 
          extent=[xmin, xmax, ymin, ymax], 
          interpolation='none', aspect='auto', 
          origin='lower', cmap=plt.cm.gray_r)


ax.set(
    xlabel='Log line brightness: ' + LINEID,
    ylabel= 'Line width: ' + LINEID,
    xlim=[xmin, xmax],
    ylim=[ymin, ymax],
)
fig.tight_layout()

plotfile = f"tarantula-I-sigma-hist-{LINEID}.png"
fig.savefig(plotfile, dpi=200)

print(plotfile, end="")
