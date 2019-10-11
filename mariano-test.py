# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
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

mask = df['Hasigmar'] > 35.0
df = df[~mask]
df

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
     'RAdeg': df.RAdeg,
     'DEdeg': df.DEdeg,
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
     'RAdeg': df.RAdeg,
     'DEdeg': df.DEdeg,
     'F_Ha': df.HaNr + df.HaNb,
     'F_Nii': df['[NII]Nr'] + df['[NII]Nr'],
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

# ## Correlations between different lines

# # Finally, the structure function

# ## Test of cartesian product of dataframes
#
# As long as the number of points is not too large, we should be able to do all the work within pandas.  The first thing to do is to make a multi-indexed dataframe that has all the pairs of points.  To make it efficient, we will just use Ra, Dec, and combined Ha velocity.

# I use a technique of joining on a temporary column, which I call `_key`, that is set to a constant (`1`) so that merging two copies of the length-$N$ dataframe produces an outer product with $N \times N$ rows, containing all pairwise combinations of the rows from the two copies. I use the `suffixes` argument to add `_` to the columns from the second copy.  This is inspired by [this blog post](https://mkonrad.net/2016/04/16/cross-join--cartesian-product-between-pandas-dataframes.html) and [this SO discussion](https://stackoverflow.com/questions/13269890/cartesian-product-in-pandas).

df1 = pd.DataFrame(
    {'RA': df.RAdeg, 'DE': df.DEdeg, 'V': vHa, '_key': 1}
)


df1.describe()

df2 = df1.copy()

# Now do the Cartesian product and drop the temporary column.

pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))

pairs.info()

# Add new columns for differences in coordinates (convert to arcsec) and velocities.

pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)

# Only keep rows with $\Delta > 0$ in RA, so that we cut out the repeated points. 

upairs = pairs[(pairs.dRA > 0.0)]

upairs.head()

upairs.describe()

upairs.corr()

mask = (upairs.log_s > 0.0) & (upairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='dV', data=upairs[mask], alpha=0.1, s=1, edgecolor='none')
ax.fig.set_size_inches(12, 12)

mask = (upairs.log_s > 0.0) & (upairs.log_dV2 > -3)
ax = sns.jointplot(x='log_s', y='log_dV2', data=upairs[mask], alpha=0.1, s=1, edgecolor='none')
ax.fig.set_size_inches(12, 12)

# That is now looking more promising.  The distributions look narrower at smaller separations.

# Introduce separation classes at intervals of 0.5 dex:

upairs.loc[:, 's_class'] = pd.Categorical((2*upairs.log_s + 0.5).astype('int'), ordered=True)

# Merge the bottom two separation classes, since there aren't enough points to go round

upairs.s_class[upairs.s_class == 0] = 1

# Now look at the stats for each separation class.  The bottom one should be empty.

upairs[upairs.s_class == 1]

for j in range(7):
    print()
    print("s_class =", j)
    print(upairs[upairs.s_class == j][['dV2', 'log_s']].describe())

sig2 = upairs.dV2.mean()
sig2a = 2*np.var(df1.V)
fig, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
for sclass, ax in zip(range(1, 7), axes):
    b2mean = np.mean(upairs.dV2[upairs.s_class == sclass])
    b2std = np.std(upairs.dV2[upairs.s_class == sclass])
    b2mean2 = np.mean(upairs.log_dV2[upairs.s_class == sclass])
    n = np.sum(upairs.s_class == sclass)
    b2sem = b2std/np.sqrt(n)
    smean = np.mean(10**upairs.log_s[upairs.s_class == sclass])
    label = f"$s = {smean:.1f}''$"
    label += f", $N = {n}$"
    label += fr", $b^2 = {b2mean:.1f} \pm {b2sem:.1f}$"
    sns.distplot(upairs.log_dV2[upairs.s_class == sclass], 
                 norm_hist=True, kde=False, ax=ax,
                 label=label, bins=20, hist_kws=dict(range=[-3.0, 3.0])
                )
    ax.plot([np.log10(b2mean)], [0.2], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.0])
    ax.legend(loc='upper left')
sns.despine()

print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')

ngroup = 500
groups = np.arange(len(upairs)) // ngroup
table = upairs[['s', 'dV2']].sort_values('s').groupby(groups).describe()


fig, ax = plt.subplots(figsize=(8, 6))
s = table[('s', 'mean')]
e_s = table[('s', 'std')]
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')]/np.sqrt(ng - 1)
#ax.plot(s, b2, 'o')
ax.axhline(sig2a, ls=':')
ax.axhline(0.5*sig2a, ls=':')
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', alpha=0.2)
sgrid = np.logspace(1.0, 3.0)
ax.plot(sgrid, 1.3*sgrid**(2/3), color="k", lw=0.5)
ax.set(xscale='log', yscale='log', 
       xlim=[10.0, 2000.0], ylim=[6.0, 100.0],
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None

# The decorrelation scale is about 80 arcsec if we define it as where $b^2 = \sigma^2$. Between 80 and 150 arcsec the structure frunction flattens as it approaches the asymptotic value of $2 \sigma^2$. 

table

# Try to repeat the above using robust statistics.  We change mean to median.  However, this reduces the values and means that at large scales we tend to $sigma^2$ instead of $2 \sigma^2$.  It also eliminates the break at 80 arcsec. 

fig, ax = plt.subplots(figsize=(8, 6))
s = table[('s', '50%')]
e_s = table[('s', '75%')] - table[('s', '25%')]
b2 = table[('dV2', '50%')]
ng = table[('dV2', 'count')]
e_b2_plus = (table[('dV2', '75%')] - table[('dV2', '50%')])/np.sqrt(ng - 1)
e_b2_minus = (table[('dV2', '50%')] - table[('dV2', '25%')])/np.sqrt(ng - 1)
e_b2 = 3*np.stack((e_b2_minus, e_b2_plus))
#ax.plot(s, b2, 'o', color="r")
ax.axhline(sig2a, ls=':')
ax.axhline(0.5*sig2a, ls=':')
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', color="r", alpha=0.2)
sgrid = np.logspace(1.0, 3.0)
ax.plot(sgrid, 0.5*sgrid**(2/3), color="k")
ax.set(xscale='log', yscale='log', 
       xlim=[10.0, 2000.0], ylim=[2.0, 35.0],
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None

# ### Repeat structure function but for components
#
# Start with the blue component

dfm = df[(df.Ha_dV >= 18.0) & (df.HaRVb > -40.0) & (df.HaRVb < -10.0)]
df1 = pd.DataFrame(
    {'RA': dfm.RAdeg, 'DE': dfm.DEdeg, 'V': dfm.HaRVb, '_key': 1}
)
df2 = df1.copy()
pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))
pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)
pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]
pairs = pairs[np.isfinite(pairs.log_dV2)].dropna()
pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)
pairs.s_class[pairs.s_class == 0] = 2
pairs.groupby('s_class')[['s', 'dV2']].describe()

sig2 = pairs.dV2.mean()
sig2a = 2*np.var(df1.V)
fig, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
for sclass, ax in zip(range(1, 7), axes):
    b2mean = np.mean(pairs.dV2[pairs.s_class == sclass])
    b2std = np.std(pairs.dV2[pairs.s_class == sclass])
    b2mean2 = np.mean(pairs.log_dV2[pairs.s_class == sclass])
    n = np.sum(pairs.s_class == sclass)
    b2sem = b2std/np.sqrt(n)
    smean = np.mean(10**pairs.log_s[pairs.s_class == sclass])
    label = f"$s = {smean:.1f}''$"
    label += f", $N = {n}$"
    label += fr", $b^2 = {b2mean:.1f} \pm {b2sem:.1f}$"
    sns.distplot(pairs.log_dV2[pairs.s_class == sclass], 
                 norm_hist=False, kde=False, ax=ax,
                 label=label, bins=20, hist_kws=dict(range=[-3.0, 3.3], color='c')
                )
    ymax = ax.get_ybound()[1]
    ax.plot([np.log10(b2mean)], [0.2*ymax], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2*ymax]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.axvline(np.log10(0.5*sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.3])
    ax.legend(loc='upper left')
sns.despine()

smasks = [
    pairs.s_class <= 2,
    pairs.s_class == 3,
    pairs.s_class == 4,
    pairs.s_class >= 5
]
titles = ['10 arcsec', '40 arcsec', '2 arcmin', '10 arcmin']
for smask, title in zip(smasks, titles):
#    ax.scatter('V', 'V_', data=pairs[smask], marker='.', alpha=1.0, s=0.02)
    sns.jointplot('VV_mean', 'dV', data=pairs[smask], kind='hex')
sns.despine()

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
for smask, ax, title in zip(smasks, axes.flat, titles):
    ax.scatter('RA', 'RA_', data=pairs[smask], marker='.', alpha=1.0, s=0.02)
    ax.set_aspect('equal')
    ax.set_title(title)
sns.despine()

print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')

ngroup = 200
groups = np.arange(len(pairs)) // ngroup
table = pairs[['s', 'dV2']].sort_values('s').groupby(groups).describe()
fig, ax = plt.subplots(figsize=(8, 6))
s = table[('s', 'mean')]
e_s = table[('s', 'std')]
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')]/np.sqrt(ng - 1)
#ax.plot(s, b2, 'o')
ax.axhline(sig2a, ls=':')
ax.axhline(0.5*sig2a, ls=':')
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', color='c', alpha=0.4)
ax.set(xscale='log', yscale='log', 
       xlim=[10.0, 2000.0], ylim=[6.0, 150.0],
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None

# OK, pretty similar.  Now for the red component.

dfm = df[df.Ha_dV >= 18.0]
df1 = pd.DataFrame(
    {'RA': dfm.RAdeg, 'DE': dfm.DEdeg, 'V': dfm.HaRVr, '_key': 1}
)
df2 = df1.copy()
pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))
pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)
pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]
pairs = pairs[np.isfinite(pairs.log_dV2)].dropna()
pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)
pairs.s_class[pairs.s_class == 0] = 1
pairs.groupby('s_class')[['s', 'dV2']].describe()

sig2 = pairs.dV2.mean()
sig2a = 2*np.var(df1.V)
fig, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
for sclass, ax in zip(range(1, 7), axes):
    b2mean = np.mean(pairs.dV2[pairs.s_class == sclass])
    b2std = np.std(pairs.dV2[pairs.s_class == sclass])
    b2mean2 = np.mean(pairs.log_dV2[pairs.s_class == sclass])
    n = np.sum(pairs.s_class == sclass)
    b2sem = b2std/np.sqrt(n)
    smean = np.mean(10**pairs.log_s[pairs.s_class == sclass])
    label = f"$s = {smean:.1f}''$"
    label += f", $N = {n}$"
    label += fr", $b^2 = {b2mean:.1f} \pm {b2sem:.1f}$"
    sns.distplot(pairs.log_dV2[pairs.s_class == sclass], 
                 norm_hist=True, kde=False, ax=ax,
                 label=label, bins=20, hist_kws=dict(range=[-3.0, 3.3], color='r')
                )
    ymax = ax.get_ybound()[1]
    ax.plot([np.log10(b2mean)], [0.2*ymax], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2*ymax]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.axvline(np.log10(0.5*sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.3])
    ax.legend(loc='upper left')
sns.despine()

print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')

ngroup = 500
groups = np.arange(len(pairs)) // ngroup
table = pairs[['s', 'dV2']].sort_values('s').groupby(groups).describe()
fig, ax = plt.subplots(figsize=(8, 6))
s = table[('s', 'mean')]
e_s = table[('s', 'std')]
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')]/np.sqrt(ng - 1)
#ax.plot(s, b2, 'o')
ax.axhline(sig2a, ls=':')
ax.axhline(0.5*sig2a, ls=':')
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', color='r', alpha=0.4)
ax.set(xscale='log', yscale='log', 
       xlim=[10.0, 2000.0], ylim=[6.0, 150.0],
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None

# Try [S II] lines.

dfm = df[df['[SII]1RVr'] - df['[SII]1RVb'] >= 18.0]
df1 = pd.DataFrame(
    {'RA': dfm.RAdeg, 'DE': dfm.DEdeg, 'V': dfm['[SII]1RVb'], '_key': 1}
).dropna()
df2 = df1.copy()
pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))
pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)
pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]
pairs = pairs[np.isfinite(pairs.log_dV2)].dropna()
pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)
pairs.s_class[pairs.s_class == 0] = 1
pairs.groupby('s_class')[['s', 'dV2']].describe()

sig2 = pairs.dV2.mean()
sig2a = 2*np.var(df1.V)
fig, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
for sclass, ax in zip(range(1, 7), axes):
    b2mean = np.mean(pairs.dV2[pairs.s_class == sclass])
    b2std = np.std(pairs.dV2[pairs.s_class == sclass])
    b2mean2 = np.mean(pairs.log_dV2[pairs.s_class == sclass])
    n = np.sum(pairs.s_class == sclass)
    b2sem = b2std/np.sqrt(n)
    smean = np.mean(10**pairs.log_s[pairs.s_class == sclass])
    label = f"$s = {smean:.1f}''$"
    label += f", $N = {n}$"
    label += fr", $b^2 = {b2mean:.1f} \pm {b2sem:.1f}$"
    sns.distplot(pairs.log_dV2[pairs.s_class == sclass], 
                 norm_hist=False, kde=False, ax=ax,
                 label=label, bins=20, hist_kws=dict(range=[-3.0, 3.3], color='c')
                )
    ymax = ax.get_ybound()[1]
    ax.plot([np.log10(b2mean)], [0.2*ymax], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2*ymax]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.axvline(np.log10(0.5*sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.3])
    ax.legend(loc='upper left')
sns.despine()

print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')

ngroup = 500
groups = np.arange(len(pairs)) // ngroup
table = pairs[['s', 'dV2']].sort_values('s').groupby(groups).describe()
fig, ax = plt.subplots(figsize=(8, 6))
s = table[('s', 'mean')]
e_s = table[('s', 'std')]
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')]/np.sqrt(ng - 1)
#ax.plot(s, b2, 'o')
ax.axhline(sig2a, ls=':')
ax.axhline(0.5*sig2a, ls=':')
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', color='c', alpha=0.4)
ax.set(xscale='log', yscale='log', 
       xlim=[10.0, 2000.0], ylim=[6.0, 150.0],
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None

dfm = df[df['[SII]1RVr'] - df['[SII]1RVb'] >= 18.0]
df1 = pd.DataFrame(
    {'RA': dfm.RAdeg, 'DE': dfm.DEdeg, 'V': dfm['[SII]1RVr'], '_key': 1}
).dropna()
df2 = df1.copy()
pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))
pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)
pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]
pairs = pairs[np.isfinite(pairs.log_dV2)].dropna()
pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)
pairs.s_class[pairs.s_class == 0] = 1
pairs.groupby('s_class')[['s', 'dV2']].describe()

sig2 = pairs.dV2.mean()
sig2a = 2*np.var(df1.V)
fig, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
for sclass, ax in zip(range(1, 7), axes):
    b2mean = np.mean(pairs.dV2[pairs.s_class == sclass])
    b2std = np.std(pairs.dV2[pairs.s_class == sclass])
    b2mean2 = np.mean(pairs.log_dV2[pairs.s_class == sclass])
    n = np.sum(pairs.s_class == sclass)
    b2sem = b2std/np.sqrt(n)
    smean = np.mean(10**pairs.log_s[pairs.s_class == sclass])
    label = f"$s = {smean:.1f}''$"
    label += f", $N = {n}$"
    label += fr", $b^2 = {b2mean:.1f} \pm {b2sem:.1f}$"
    sns.distplot(pairs.log_dV2[pairs.s_class == sclass], 
                 norm_hist=True, kde=False, ax=ax,
                 label=label, bins=20, hist_kws=dict(range=[-3.0, 3.3], color='r')
                )
    ymax = ax.get_ybound()[1]
    ax.plot([np.log10(b2mean)], [0.2*ymax], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2*ymax]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.axvline(np.log10(0.5*sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.3])
    ax.legend(loc='upper left')
sns.despine()

ngroup = 500
groups = np.arange(len(pairs)) // ngroup
table = pairs[['s', 'dV2']].sort_values('s').groupby(groups).describe()
fig, ax = plt.subplots(figsize=(8, 6))
s = table[('s', 'mean')]
e_s = table[('s', 'std')]
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')]/np.sqrt(ng - 1)
#ax.plot(s, b2, 'o')
ax.axhline(sig2a, ls=':')
ax.axhline(0.5*sig2a, ls=':')
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', color='r', alpha=0.4)
ax.set(xscale='log', yscale='log', 
       xlim=[10.0, 2000.0], ylim=[6.0, 150.0],
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None

# Try He I line

dfm = df[(df['HeIRVr'] - df['HeIRVb'] >= 18.0) & (df.HeIRVb > -70.0)]
df1 = pd.DataFrame(
    {'RA': dfm.RAdeg, 'DE': dfm.DEdeg, 'V': dfm['HeIRVb'], '_key': 1}
).dropna()
df2 = df1.copy()
pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))
pairs.loc[:, 'dDE'] = 3600*(pairs.DE - pairs.DE_)
pairs.loc[:, 'dRA'] = 3600*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
pairs.loc[:, 'log_s'] = np.log10(pairs.s)
pairs.loc[:, 'dV'] = pairs.V - pairs.V_
pairs.loc[:, 'dV2'] = pairs.dV**2
pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)
pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]
pairs = pairs[np.isfinite(pairs.log_dV2)].dropna()
pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)
pairs.s_class[(pairs.s_class == 1) | (pairs.s_class == 0)] = 2
pairs.groupby('s_class')[['s', 'dV2']].describe()

sig2 = pairs.dV2.mean()
sig2a = 2*np.var(df1.V)
fig, axes = plt.subplots(5, 1, figsize=(10, 12.5), sharex=True)
for sclass, ax in zip(range(2, 7), axes):
    b2mean = np.mean(pairs.dV2[pairs.s_class == sclass])
    b2std = np.std(pairs.dV2[pairs.s_class == sclass])
    b2mean2 = np.mean(pairs.log_dV2[pairs.s_class == sclass])
    n = np.sum(pairs.s_class == sclass)
    b2sem = b2std/np.sqrt(n)
    smean = np.mean(10**pairs.log_s[pairs.s_class == sclass])
    label = f"$s = {smean:.1f}''$"
    label += f", $N = {n}$"
    label += fr", $b^2 = {b2mean:.1f} \pm {b2sem:.1f}$"
    sns.distplot(pairs.log_dV2[pairs.s_class == sclass], 
                 norm_hist=False, kde=False, ax=ax,
                 label=label, bins=20, hist_kws=dict(range=[-3.0, 3.3], color='c')
                )
    ymax = ax.get_ybound()[1]
    ax.plot([np.log10(b2mean)], [0.2*ymax], 'o', color='k')
    ax.plot([np.log10(b2mean - b2sem), np.log10(b2mean + b2sem)], [0.2*ymax]*2, lw=3, color='k')
    ax.axvline(np.log10(sig2a), color='k', ls=':')
    ax.axvline(np.log10(0.5*sig2a), color='k', ls=':')
    ax.set(xlim=[-3.0, 3.3])
    ax.legend(loc='upper left')
sns.despine()

print(f'Dotted line is 2 x sigma^2 = {sig2a:.2f}')

ngroup = 200
groups = np.arange(len(pairs)) // ngroup
table = pairs[['s', 'dV2']].sort_values('s').groupby(groups).describe()
table[[('s', 'mean'), ('s', 'std'), ('dV2', 'mean'), ('dV2', 'std'), ('dV2', 'count')]]

fig, ax = plt.subplots(figsize=(8, 6))
s = table[('s', 'mean')]
e_s = table[('s', 'std')]
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')]/np.sqrt(ng - 1)
#ax.plot(s, b2, 'o')
ax.axhline(sig2a, ls=':')
ax.axhline(0.5*sig2a, ls=':')
ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o')
ax.set(xscale='log', yscale='log', 
       xlim=[10.0, 2000.0], ylim=[9.0, 150.0],
       xlabel='separation, arcsec',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None

# ## Maps of the velocities

points_of_interest = {
    "eta Car": [161.26517, -59.684425],
    "Tr 14": [160.98911, -59.547698],
    "WR 25": [161.0433, -59.719735],
    "Finger": [161.13133, -59.664035],
}
def mark_points(ax):
    for label, c in points_of_interest.items():
        ax.plot(c[0], c[1], marker='+', markersize='12', color='k')

with sns.axes_style("darkgrid"):
    fig, [axr, axb] = plt.subplots(1, 2, figsize=(18, 8))
    scat = axr.scatter(df.RAdeg, df.DEdeg, 
                      s=40*(np.log10(df.HaNr/df.HaNb) + 1.3), 
                      c=df.HaRVr, cmap='RdBu_r',
                      vmin=-55, vmax=35, 
                     )
    scat = axb.scatter(df.RAdeg, df.DEdeg, 
                      s=40*(np.log10(df.HaNb/df.HaNr) + 1.3), 
                      c=df.HaRVb, cmap='RdBu_r',
                      vmin=-55, vmax=35,
                     )
#    scat2 = ax.scatter(df.RAdeg, df.DEdeg, 
#                      s=50*(np.log10(df.HaNr) - 3), 
#                      c=df.HaRVr, cmap='RdBu_r',
#                      vmin=-55, vmax=35, marker='+',
#                     )
    fig.colorbar(scat, ax=[axr, axb])
    mark_points(axr)
    mark_points(axb)
    axr.invert_xaxis()
    axr.set_aspect(2.0)
    axb.invert_xaxis()
    axb.set_aspect(2.0)  
    axr.set_title('H alpha red layer velocity')
    axb.set_title('H alpha blue layer velocity')    

# Here, the symbol size is proportional to the relative brightness of that layer. We can see linear structures, which clearly show the correlation identified above: velocity of each layer is anticorrelated with its relative brightness.

with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(12, 12))
    scat = ax.scatter(dfHa.RAdeg, dfHa.DEdeg, s=8*(dfHa.sigma - 12), c=dfHa.V_mean, cmap='RdBu_r')
    mark_points(ax)
    fig.colorbar(scat, ax=ax).set_label("$V$")
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title("H alpha mean velocity")

with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(12, 12))
    scat = ax.scatter(df.RAdeg, df.DEdeg, s=50*(np.log10(df.HaNr/df.HaNb) + 1.0), c=df.HaRVr - df.HaRVb, cmap='viridis')
    fig.colorbar(scat, ax=ax).set_label("$V_r - V_b$")
    mark_points(ax)   
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title("H alpha red–blue layer velocity difference")

with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(12, 12))
    scat = ax.scatter(df.RAdeg, df.DEdeg, s=100, c=df.HaNb, cmap='gray_r', vmin=0.0, vmax=4e5)
    fig.colorbar(scat, ax=ax)
    mark_points(ax)
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title('H alpha blue layer brightness')

with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(12, 12))
    scat = ax.scatter(df.RAdeg, df.DEdeg, s=100, c=df.HaNr, cmap='gray_r', vmin=0.0, vmax=4e5)
    fig.colorbar(scat, ax=ax)
    mark_points(ax)
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title('H alpha red layer brightness')

def eden(R):
    """Approximate sii electron density from R=6717/6731"""
    RR = 1.0/R
    return 2489*(RR - 0.6971) / (2.3380 - RR)

with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(12, 12))
    scat = ax.scatter(dfSii.RAdeg, dfSii.DEdeg, s=100, c=eden(dfSii.R12), cmap='gray_r', vmin=0.0, vmax=1000.0)
    fig.colorbar(scat, ax=ax).set_label('$n_e$')
    mark_points(ax)
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title('[S II] density')

with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(12, 12))
    scat = ax.scatter(dfSii.RAdeg, dfSii.DEdeg, s=100, 
                      c=dfSii.F_Ha/eden(dfSii.R12)**2, 
                      vmin=0.0, vmax=50.0, cmap='inferno_r')
    fig.colorbar(scat, ax=ax).set_label('$H$')
    mark_points(ax)
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title('Effective layer thickness')

dfSN = dfSii[dfSii.F_Nii > 0.0]
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(12, 12))
    scat = ax.scatter(dfSN.RAdeg, dfSN.DEdeg, s=100, 
                      c=dfSN.F_Nii/dfSN.F_Ha, 
                      cmap='hot_r')
    fig.colorbar(scat, ax=ax).set_label('$6583 / 6563$')
    mark_points(ax)
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title('[N II] / H alpha ratio')





pairs.s_class.cat.categories

(pairs.s_class < 5).sum()

(pairs.s_class >= 6).sum()

(pairs.s_class == 5).sum()


