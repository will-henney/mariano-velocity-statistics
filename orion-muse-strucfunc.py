import sys
import json
import numpy as np
from astropy.utils.misc import JsonCustomEncoder
from astropy.io import fits
from pathlib import Path
import astropy.units as u


sys.path.append(str(Path("~/Dropbox/muse-strucfunc").expanduser()))
import strucfunc

try:
    ITYPE = sys.argv[1]
except IndexError:
    sys.exit(f"Usage: {sys.argv[0]} low|mid|high")


ihdu = fits.open(f"muse-linesum-{ITYPE}.fits")[0]
vhdu = fits.open(f"muse-vmean-{ITYPE}.fits")[0]

iha = ihdu.data.astype("float")
vha = vhdu.data.astype("float")
iha /= np.nanmax(iha)

sf = strucfunc.strucfunc_numba_parallel(
    vha,
    wmap=iha,
    dlogr=0.05,
)

good = iha > 0.001
sf["Unweighted mean velocity"] = np.mean(vha[good])
sig2 = sf["Unweighted sigma^2"] = np.var(vha[good])
v0w = sf["Weighted mean velocity"] = np.average(vha, weights=iha)
sf["Weighted sigma^2"] = np.average((vha - v0w) ** 2, weights=iha)

D = 0.410 * u.kpc
pix_arcsec = 0.2
pix_pc = (pix_arcsec * (D / u.pc) * u.au).to(u.pc)

sf["sep, pc"] = 10 ** sf["log10 r"] * pix_pc.value

sf["r0, pc"] = np.interp(
    sf["Unweighted sigma^2"],
    sf["Unweighted B(r)"],
    sf["sep, pc"],
)
sf["r0, pc"]

jsonfilename = f"orion-muse-{ITYPE}-strucfunc.json"
with open(jsonfilename, "w") as f:
    json.dump(sf, fp=f, indent=3, cls=JsonCustomEncoder)
