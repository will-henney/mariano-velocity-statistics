import sys
from pathlib import Path
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
datadir = Path("data/Tarantula/MUSE_R136toWill")

hdulist = fits.open(datadir / fitsfilename[LINEID])

n = None
sb = hdulist[1].data[:n, :n].astype(np.float64)
vv = hdulist[2].data[:n, :n].astype(np.float64)
ss = hdulist[3].data[:n, :n].astype(np.float64)

# Replace spurious values in the arrays
m = ~np.isfinite(sb*vv*ss) | (sb < 0.0)
if LINEID == "nii":
    # Remove bad patch from the [N II] map
    m = m | (sb > 6e4) 
sb[m] = 0.0
vv[m] = np.nanmean(vv)
sb /= sb.max()


rslt = strucfunc.strucfunc_numba_parallel(vv, wmap=sb, dlogr=0.05)

good = (~m) & (sb > 0.001)
rslt["Unweighted mean velocity"] = np.mean(vv[good])
rslt["Unweighted sigma^2"] = np.var(vv[good])
v0w = rslt["Weighted mean velocity"] = np.average(vv, weights=sb)
rslt["Weighted sigma^2"] = np.average((vv - v0w)**2, weights=sb)


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

jsonfilename = f"tarantula-strucfunc-{LINEID}.json"
with open(jsonfilename, "w") as f:
    json.dump(rslt, fp=f, indent=3, cls=MyEncoder)
print(jsonfilename, end="")
