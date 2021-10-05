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
    PREFIX = sys.argv[1]
    SUFFIX = sys.argv[2]
except IndexError:
    sys.exit(f"Usage: {sys.argv[0]} PREFIX SUFFIX")

ihdu = fits.open(f"{PREFIX}-flux-{SUFFIX}.fits")[0]
vhdu = fits.open(f"{PREFIX}-vmean-{SUFFIX}.fits")[0]

iha = ihdu.data.astype("float")
vha = vhdu.data.astype("float")

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
pix_arcsec = 0.534
pix_pc = (pix_arcsec * (D / u.pc) * u.au).to(u.pc)

sf["sep, pc"] = 10 ** sf["log10 r"] * pix_pc.value

sf["r0, pc"] = np.interp(
    sf["Unweighted sigma^2"],
    sf["Unweighted B(r)"],
    sf["sep, pc"],
)
sf["r0, pc"]

jsonfilename = f"{PREFIX}-strucfunc-{SUFFIX}.json"
with open(jsonfilename, "w") as f:
    json.dump(sf, fp=f, indent=3, cls=JsonCustomEncoder)
