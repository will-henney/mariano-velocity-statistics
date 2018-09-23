# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext_format_version: '1.2'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.3
# ---

# All attempts to load from the URL have failed.  I will try to download the data locally first
#

from astropy.table import Table
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# %matplotlib inline
sns.set_color_codes()

damiani_tab1_file = "data/J_A+A_591_A74_table1.dat.fits"


tab = Table.read(damiani_tab1_file)

tab

df = tab.to_pandas()

df.describe()

fig, [axb, axr, axd] = plt.subplots(3, 1, sharex=True)
axb.hist(tab["HaRVb"], label='Blue comp')
axr.hist(tab["HaRVr"], color='r', label='Red comp')
axd.hist(tab["HaRVr"] - tab["HaRVb"], color='g', label='Delta')
for ax in axb, axr, axd:
    ax.legend()
axd.set(xlabel='Velocity')

# Add a column with red–blue velocity difference, $dV$

df = df.assign(Ha_dV=df['HaRVr'] - df['HaRVb'])

# Add a column that is true when $dV < 15$. 

df = df.assign(Ha_close=(df['Ha_dV'] < 18.0).astype('S5') )

# Add a column with the log of the red/blue intensity ratio.

df = df.assign(Ha_rb_ratio=np.log10(df['HaNr']/df['HaNb']))

sns.pairplot(df, 
             vars=["HaRVb", "HaNb", "Hasigmab", "Ha_dV", "RAdeg", "DEdeg"], 
             diag_kind='hist', hue="Ha_close", 
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none'),
             diag_kws=dict(bins=20),
            )

# Note that the small red-blue velocity differences of $dV < 15$ are correlated with the following:
#
# * Less negative $V_b$
# * Larger spread in component width $\sigma_b$
# * Concentration in northern declinations, $\delta > -59.7$, with $\alpha \approx 160.1 \pm 0.2$.  This is the region just N of Tr 14
#
# It might be worth filtering these out, since the separation into blue and red components may not be reliable

# + {"scrolled": true}
mask = df['Hasigmar'] > 35.0
df = df[~mask]
df
# -

df.dropna(inplace=True)

sns.pairplot(df, 
             vars=["HaRVr", "HaNr", "Hasigmar", "Ha_dV", "RAdeg", "DEdeg"], 
             diag_kind='hist', hue="Ha_close",
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none'),
             diag_kws=dict(bins=20),
            )

sns.pairplot(df, 
             vars=["HaRVb", "HaRVr", "Hasigmab", "Hasigmar", "Ha_rb_ratio", "Ha_dV"], 
             diag_kind='hist', hue="Ha_close", kind='scatter',
             plot_kws=dict(alpha=0.5, s=4, edgecolor='none'),
             palette='Set2',
             diag_kws=dict(bins=20),
            )

# So, it definitely looks like the orange points (low $dV$) should be removed.  Note the strange shape of the $\sigma_b$--$\sigma_r$ correlation: points either have wide blue, or wide red, but not both. This might well be a degeneracy in the fitting. 

# ### $V$ and $\sigma$ correlations with red/blue flux ratio
#
# Looking at the penultimate column, velocities of both components are negatively correlated with $F_r/F_b$, especially once the orange points are removed.  This is real, I think. 

ax = sns.scatterplot('HaRVb', 'HaRVr', data=df, 
                     hue='Ha_rb_ratio', size='Hasigmab', sizes=(0, 200),
                     style='Ha_close',  
                     palette='coolwarm')
ax.set(xlim=[-45, 5], ylim=[-15, 35])
ax.set_aspect('equal')
plt.gcf().set_size_inches(8, 8)

# So the previous plot shows the $V_b$--$V_r$ correlation, with color showing $F_r/F_b$ and size showing $\sigma_b$.  It is clear that as $\langle V \rangle$ increases (redder):
#
# 1. $F_b$ becomes higher than $F_r$ (`Ha_rb_ratio`, which is the logarithm, becomes increasingly negative).
# 2. $\sigma_b$ increases
#
# This means that it is always the component closer to systemic ($-10$) that is the brighter
#

# ## Combining the red and blue components

# Implement equations A7, A8, A9 of García-Díaz et al. (2008) for the moments from the sum of two components:
# $$
# F = F_1 + F_2 \ ;
# \quad\quad V = \frac{F_1 V_1 + F_2 V_2}{F} \ ;
# \quad\quad \sigma^2 = \frac{F_1 \sigma_1^2 + F_2 \sigma_2^2}{F}
# + \frac{F_1 F_2 (V_1 - V_2)^2}{F^2}
# $$

def combine_moments(f1, v1, s1, f2, v2, s2, return_skew=False):
    """Find combined flux, mean velocity, and sigma for two components 
    with fluxes `f1` and `f2`, velocities `v1` and `v2`, and sigmas `s1` and `s2`. 
    Returns tuple of the combined moments: `f`, `v`, `s`."""
    f = f1 + f2
    v = (v1*f1 + v2*f2)/f
    ss = (s1*s1*f1 + s2*s2*f2)/f
    ss += f1*f2*(v1 - v2)**2 / f**2
    s = np.sqrt(ss)
    if return_skew:
        p1 = f1/f
        p2 = f2/f
        skew = p1*p2*(v1 - v2)*((1 - 2*p1)*(v1 - v2)**2 + 3*(s1**2 - s2**2))
        skew /= (p1*(p2*(v1 - v2)**2 + s1**2 - s2**2) + s2**2)**1.5
#        vmode = np.where(f1 > f2, v1, v2)
#        mskew = (v - vmode)/s
        return f, v, s, skew
    else:
        return f, v, s
    

fHa, vHa, sHa, gHa = combine_moments(
    df.HaNr, df.HaRVr, df.Hasigmar, 
    df.HaNb, df.HaRVb, df.Hasigmab,
    return_skew=True
)

dfHa = pd.DataFrame(
    {'log_F': np.log10(fHa), 
     'V_mean': vHa, 
     'sigma': sHa, 
     'skew': gHa,
     'R_B': df.Ha_rb_ratio,
     'dV': df.Ha_dV,
     'close': df.Ha_close,
    }
).dropna()

dfHa.describe()

dfHa.corr(method="spearman")

sns.pairplot(dfHa, 
             vars=["log_F", "V_mean", "sigma", "dV", "R_B", "skew"],
             plot_kws=dict(alpha=0.5, s=6, edgecolor='none'), 
             diag_kind='hist', hue="close", palette="magma_r",
            )

# Some conclusions from this:
#
# 1. The $\sigma$ of the combined line is dominated by the difference between the components, except for in the low-splitting sub-sample.  You can see this in the `dV`-`sigma` correlation graph for the red points. 

# ### Repeat the above, but for He I line.  
#
# We calculate the red–blue ratio `R_B` and velocity difference `dV` from this line itself.

fHe, vHe, sHe = combine_moments(df.HeINr, df.HeIRVr, df.HeIsigmar, df.HeINb, df.HeIRVb, df.HeIsigmab)
dfHe = pd.DataFrame(
    {'log_F': np.log10(fHe), 
     'V_mean': vHe, 
     'sigma': sHe, 
     'R_B': np.log10(df.HeINr/df.HeINb), 
     'dV': df.HeIRVr - df.HeIRVb,
     'close': pd.Categorical((df.HeIRVr - df.HeIRVb < 18.0).astype('S5')),
    }
).dropna()

# Even though we did `.dropna()`, we still have some `Inf`s in the red–blue ratio.  Also, we trim the outliers from that ratio and from $\sigma$. 

dfHe = dfHe[np.isfinite(dfHe.R_B) & (dfHe.sigma < 35.0) & (np.abs(dfHe.R_B) < 1.3)]

sns.pairplot(dfHe,
             vars=["log_F", "V_mean", "sigma", "dV", "R_B"],
             plot_kws=dict(alpha=0.5, s=10, edgecolor='none'), 
             diag_kind='hist', hue="close", palette="viridis_r",
            )

# ### And repeat again, but for the [N II] line 
#

fNii, vNii, sNii = combine_moments(
    df['[NII]Nr'], df['[NII]RVr'], df['[NII]sigmar'],
    df['[NII]Nb'], df['[NII]RVb'], df['[NII]sigmab'],
)
dfNii = pd.DataFrame(
    {'log_F': np.log10(fNii), 
     'V_mean': vNii, 
     'sigma': sNii, 
     'R_B': np.log10(df['[NII]Nr']/df['[NII]Nb']), 
     'dV': df['[NII]RVr'] - df['[NII]RVb'],
     'close': pd.Categorical((df['[NII]RVr'] - df['[NII]RVb'] < 18.0).astype('S5')),
    }
).dropna()
dfNii.describe()

dfNii = dfNii[np.isfinite(dfNii.R_B) & (dfNii.dV > 0.0) & (np.abs(dfNii.R_B) < 1.3)]

sns.pairplot(dfNii,
             vars=["log_F", "V_mean", "sigma", "dV", "R_B"],
             plot_kws=dict(alpha=0.5, s=20, edgecolor='none'), 
             diag_kind='hist', hue="close", palette="rainbow_r",
            )

# ### And finally for the [S II] lines

# +
fSii1, vSii1, sSii1, gSii1 = combine_moments(
    df['[SII]1Nr'], df['[SII]1RVr'], df['[SII]1sigmar'],
    df['[SII]1Nb'], df['[SII]1RVb'], df['[SII]1sigmab'],
    return_skew=True,
)
dfSii1 = pd.DataFrame(
    {'log_F': np.log10(fSii1), 
     'V_mean': vSii1, 
     'sigma': sSii1, 
     'skew': gSii1,
     'R_B': np.log10(df['[SII]1Nr']/df['[SII]1Nb']), 
     'dV': df['[SII]1RVr'] - df['[SII]1RVb'],
     'close': pd.Categorical((df['[SII]1RVr'] - df['[SII]1RVb'] < 18.0).astype('S5')),
    }
).dropna()

fSii2, vSii2, sSii2, gSii2 = combine_moments(
    df['[SII]2Nr'], df['[SII]2RVr'], df['[SII]2sigmar'],
    df['[SII]2Nb'], df['[SII]2RVb'], df['[SII]2sigmab'],
    return_skew=True,

)
dfSii2 = pd.DataFrame(
    {'log_F': np.log10(fSii2), 
     'V_mean': vSii2, 
     'sigma': sSii2, 
     'skew': gSii2,
     'R_B': np.log10(df['[SII]2Nr']/df['[SII]2Nb']), 
     'dV': df['[SII]2RVr'] - df['[SII]2RVb'],
     'close': pd.Categorical((df['[SII]2RVr'] - df['[SII]2RVb'] < 18.0).astype('S5')),
    }
).dropna()


# -

dfSii1 = dfSii1[np.isfinite(dfSii1.R_B) & (dfSii1.dV > -5) & (dfSii1.dV < 55) & (np.abs(dfSii1.R_B) < 1.3)]
dfSii2 = dfSii2[np.isfinite(dfSii2.R_B) & (dfSii2.dV > -5) & (dfSii2.dV < 55) & (np.abs(dfSii2.R_B) < 1.3)]

dfSii1.corr()

dfSii2.corr()

sns.pairplot(dfSii1,
             vars=["log_F", "V_mean", "sigma", "dV", "R_B", "skew"],
             plot_kws=dict(alpha=0.5, s=20, edgecolor='none'), 
             diag_kind='hist', hue="close",
            )

sns.pairplot(dfSii2,
             vars=["log_F", "V_mean", "sigma", "dV", "R_B", "skew"],
             plot_kws=dict(alpha=0.5, s=20, edgecolor='none'), 
             diag_kind='hist', hue="close",
            )

# ### Skewness estimates
#
# Skewness is the third central moment: $\gamma = \langle (X - \mu)^3 / \sigma^3 \rangle$.  If we treat the two components as $\delta$-functions, then this gives:
# $$
# \gamma = \frac{F_1 (V_1 - V)^3 + F_2 (V_2 - V)^3}{F \sigma^3}
# $$
# This is implemented above, but it varies too strongly due to the $V^3$ dependence.  Instead I will try the *mode skewness*: 
# $$
# \gamma = \frac{\text{mean} - \text{mode}}
# {\text{standard deviation}} \ , 
# $$
# in which the mode is $V_1$ or $V_2$, depending on whether $F_1 > F_2$ or not.

# Now I have a better version.  From this [stack overflow question](https://math.stackexchange.com/questions/346451/skewness-of-mixture-density) we have the following, defining $p_1 = F_1/F$, $p_2 = F_2/F$:
# $$
# \gamma = \frac{
# p_1 p_2 (V_1 - V_2) \left[ (1 - 2 p_1) (V_1 - V_2)^2 + 3 (\sigma_1^2 - \sigma_2^2) \right]
# }{
# \left\{ p_1 \left[ p_2 (V_1 - V_2)^2 + \sigma_1^2 - \sigma_2^2\right] + \sigma_2^2 \right\}^{3/2}
# }
# $$

# So I have now ditched the mskew - it is useless.

sns.pairplot(dfHa, 
             vars=["V_mean", "sigma", "dV", "R_B", "skew"],
             plot_kws=dict(alpha=0.5, s=6, edgecolor='none'), 
             diag_kind='hist', hue="close", palette="magma_r",
            )

# So, the conclusion is that skewness is slightly superior to red–blue flux ratio for predicting the mean velocity.  This is most apparent for H$\alpha$, where we have Spearman correlation coeficients $r(\gamma, V) = -0.87$ as against $r(F_r/F_b, V) = 0.64$.  In the case of [S II], the difference is smaller.

# ## [S II] 6731, 6717 intercomparison

dfSii = pd.DataFrame(
    {'log_F': np.log10(fSii1 + fSii2), 
     'R12': fSii1/fSii2,
     'dV12': vSii1 - vSii2, 
     'V': (fSii1*vSii1 + fSii2*vSii2)/(fSii1 + fSii2),
     'sigma': np.sqrt((fSii1*sSii1**2 + fSii2*sSii2**2)/(fSii1 + fSii2)),
     'sigma12': sSii1/sSii2,
     'skew': gSii2,
     'close': pd.Categorical((df['[SII]2RVr'] - df['[SII]2RVb'] < 18.0).astype('S5')),
    }
).dropna()

dfSii = dfSii[(dfSii.sigma12 > 0.9) & (dfSii.sigma12 < 1.1)]

dfSii.describe()

dfSii.corr(method='pearson')

sns.pairplot(dfSii, 
             plot_kws=dict(alpha=0.5, s=6, edgecolor='none'), 
             diag_kind='hist', hue="close",
            )

# There is very little correlation with anything, except for doublet ratio with flux.  This has $r = -0.67$, suggesting that just under half ($r^2 = 0.45$) of the variation in flux can be ascribed to variation in density.  The rest could be a combination of temperature, extinction, and layer thickness variations.  Plus, observational errors in the ratio as well.

# Other weaker correlations are:
#
# * $r^2 \approx 0.1$: doublet ratio versus mean velocity (negative), total $\sigma$ (positive), doublet $\sigma$ ratio (negative), skewness (positive)
# * $r^2 \approx 0.06$: skewness versus doublet $\sigma$ ratio (negative, and mainly for negative skewness) and doublet velocity difference (positive).  

# Finally, the doublet velocity difference is marginally significantly non-zero: $0.6 \pm 0.5$.  This presumably is related to the differing densities in the blue and red layers.

# *We should also do it for the blue and red components separately*


