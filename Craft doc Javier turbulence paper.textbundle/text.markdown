# Javier turbulence paper


Compare structure functions from different regions.


- Orion Nebula from the Sac paper
	- Larger scale Orion from Haenel:1987a
- M33 regions from Javier thesis
- 30 Doradus from Melnick?
	- Melnick data is suspect - Castro:2018a is better
- M8 from us?
- Carina from us?
- Anywhere else?
	- N44 and N180 in LMC from McLeod:2019a – *These would probably be best left for a follow-up study*
	- Figure 7.6 of Javier’s thesis has a lot of curves for different regions, but I am not sure where these are all from.

Correlation length should represent scale of dominant energy injection. If this is longer for giant regions than for Orion, then that means that the energy injection mechanism is different.  
  
But also look for scaling with stromgren radius, say. And also look for scale of brightness fluctuations.

## Relation to the L–σ relation


Here is a comparison between our provisional results and those of [Moiseev:2015a](https://ui.adsabs.harvard.edu/abs/2015MNRAS.449.3568M/abstract) for dwarf galaxies. 

![image](assets/37ED2AD0-2F48-4459-9E4C-AE29A36286B3.png)

So, our results all fall below their curve, **except for 30 Dor**.  So why the difference?


1. Is this just the difference between POS σ and LOS σ that we see in Orion?
2. Is it because of metallicity – presumably dwarf galaxies have low metallicity, and 30 Dor in the LMC is the only object from our sample that is in a dwarf galaxy
3. Is it an age effect?  Carina and 30 Dor are similar luminosities and sizes.  So is it just that Carina is yet to have an SN explosions? (η Car will be the first …)

### Recent papers that cite Moiseev et al


##### [Kohandel:2020a](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.1250K)


![image](assets/F751F115-64EB-48F7-86D3-370DA0EE9EC9.png)

They calculate synthetic observations of LOS σ from simulations of a Lyman break galaxy.  They claim that the main energy source for driving the velocity dispersion is galaxy mergers and accretion, whereas stellar feedback is unimportant.  Here is the part of the introduction where they talk about previous studies of this:

![image](assets/39E428AB-EF79-48CC-A443-036021110C75.png)

Note that this is for Epoch of Reionization, so very high redshift (z ~ 10?).  Observing it would require very high spatial resolution (0.005 arcsec for 30 pc).  So, we shouldn’t worry about it too much.

##### [Krumholz:2016a](https://ui.adsabs.harvard.edu/abs/2016MNRAS.458.1671K) and  [Yu:2019a](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.4463Y/abstract) and [Varidel:2020a](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.2265V/abstract)


	*Is Turbulence in the Interstellar Medium Driven by Feedback or Gravity? An Observational Test – Mark R. Krumholz1,2,3⋆ and Blakesley Burkhart4†*

This is mainly concerned with more massive systems than we are studying.  That is, entire galaxies with a high star-formation rate (for MW it is about 3 Msun/yr). 

![image](assets/54C88D92-76E7-4CBC-BA56-A1ED14A39BD6.png)

Another similar paper is [Yu:2019a](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.4463Y), which gives more observations, but doesn’t advance the theory at all.  

***Note that the star-formation rate is derived from Hα luminosity***

![](assets/3CD014CC-0FC1-4756-B104-7FC88956458F.png)


Citations for this are Kennicut (1998) with Chabrier (2003) for the IMF. 
> So, based on the above equation, our regions range from 0.00005 Msun/yr (Orion) up to 0.013 Msun/yr (30 Dor), which is much less than the range studied in these papers. 


![image](assets/47C26204-95FE-4CCB-A4B9-D7BB788CC4E6.png)

![image](assets/718A0615-EEC9-450D-AF16-ADC7503220E6.png)

Above shows figures from Yu:2019a and Varidel:2020a, which show the σ versus other quantities.  I have added our points to the σ versus SFR plots:


- Compared with the Yu sample, our points seem to lie on the same general trend, which looks consistent with σ⁴ ~ L, but at much smaller luminosity (or SFR)
- The Varidel sample is much flatter, so that are points seem to follow a steeper trend than theirs.
- Varidel also consider the trend with Σ(SFR) – star formation surface denity Msun/yr/kpc²
	- Of course the Σ(SFR) is very *large* for our regions, since the surface area is small, which means we are mainly off to the right of this graph, and the comparison is not very meaningful.

##### [Zaragoza-Cardiel:2014a](https://ui.adsabs.harvard.edu/abs/2014MNRAS.445.1412Z/abstract)


![image](assets/EA7269A7-DF43-4C91-AFD9-6EDFB40DBE07.png)

These regions are much closer to our own in luminosity and velocity dispersion.  It is a useful sample, since it covers the entire pair of galaxies, so it it probably less biased then the selection that we have. 

Taken as a whole, their sample does not show a clear L–σ relation.  They have a fraction of low-luminosity regions with a high velocity dispersion.

![image](assets/B252DE14-CB70-469D-BD3C-A39906637A33.png)

They interpret this as a change between regimes, accprding to the mass of the region.  Only the high-mass region (M(H II) > 1e5 Msun) are expected to show the relation.
> Mass of H II will be proportional to L / n 

----

## Individual regions


### The Lagoon Nebula – M8

- This is something Javier did a ipynb analysis on, following the structure of what I had done previously for Carina.
- So that would make it a good candidate for including in a paper, since it is already work that he has done.
- Correlation and Anti-correlation scale:	
	- Correlation scale is about 200 arcsec (just over 1 pc)	
	- For 300 to 1000 arcsec is uncorrelated (flat structure function)
	- For 1000 to 3000 arcsec, the structure function climbs again. Looking at the maps, this is anti-correlated, with a sinusoidal variation with a period of about 0.5 deg = 1800 arcsec	
	- Orientation of the axis of this variation is roughly E–W (PA = 120 degrees)
	- You get nearly 2 whole spatial periods across the nebula
	- Distance is 1250 pc, so 1800 arcsec is 6.25 pc	
	- Smallest scale accessible from observations is about 20 arcsec or 0.1 pc.  
		- So this means it would be impossible to see variations on as small scales as the correlation length in Orion

![image](assets/FF01D427-1090-4869-8683-6F5A32E91A7E.png)

Above is taken from Javier’s jupyter notebook ([read only version](https://nbviewer.jupyter.org/github/will-henney/CSHR/blob/master/lagoon-will.ipynb)).  It looks like the total σ includes some contribution from the 5–10 pc scale fluctuations, so it would better to normalize by only the fluctuations on scales < 1000 arcsec (3 pc).  This would come out automatically if we were to use the functional fitting methodology described below. 

Note that magnitude of σ² is a bit *smaller* than in Orion: 6 km²/s² as opposed to 9.4 for Ha in Orion.  This is for the first decorrelation at a few hundred arcsec. The total σ² is 7.5, and b² reaches about 50 for the largest separation, which is due to a large scale gradient of about 7 km/s over the full RA span of the nebula. 

We should recalculate the structure function for directions along the minor and major axis of the nebula. since I think that the large scale structure is only along the major axis.  It appears that the large-scale pattern is related to the dark extinction filament that crosses the nebula – the more redshifted emission is associated with the filament, so it is like it is a champagne flow that has an obstacle in the middle that stops some of it. There is also an older Haenel paper that shows the same thing (this is in the same PDF as the Haenel Orion paper). 

### Carina


This also has a large-scale oscillatory pattern in the velocities. The period is about 0.1 degrees and the orientation is about PA=135 degrees. Converting this to physical scale @ 2 kpc gives 3.6 pc

Correlation length (strucfunc = 1) is at about 50 arcsec. At D = 2 kpc. this means 0.5 pc as the physical correlation length

![image](assets/04E8D5A1-2F90-4477-997D-28E9516270C2.png)

Above figure is taken from my jupyter notebook.  (should be 50 arcsecond, not minute). 

### Comparison with Orion


From Arthur et al (2016) we have a correlation length of order 20 to 30 arcsec, which corresponds to the 2nd-order velocity structure function (normalized) reaching unity (NOT TWO). This is a physical scale of about 0.05 pc I have added red lines on the following figure from that paper to make this more clear. 

![image](assets/6187A4FD-C7CC-4BE4-8C2F-010FAA955E01.png)

The POS velocity dispersion is about σ = 3 km/s for Hα and [O ɪɪɪ].  This is much smaller than the LOS velocity dispersion from the line widths.  Even when correcting for the dust scattering, this gives σ = 6 km/s. 

Note that the [N ɪɪ]  has a smaller POS dispersion of 2.4 km/s *and* a smaller autocorrelation length of 0.03 pc, but the *unnormalized* structure function would line up perfectly with the higher ionization lines.  I am not sure what this is telling us.  The [N ɪɪ] results also show spatial inhomogeneity beyween the N and S sides of the nebula, which is not seen so much in the other lines. 

### Tarantula (30 Doradus)


Melnick:2019a say that the structure function is entirely flat on scales from 3 to 200 pc – *but can this really be right?*

![image](assets/E49B5D2F-DACC-4702-8D7C-0F044B294756.png)

For a start, it needs to be checked against results of Castro:2018a who clearly show (Fig 10 and 11) that the dominant scale of the velocity fluctuations is 10 to 20 arcsec.  This corresponds to physical scales at 50 kpc of 2.5 pc to 5 pc.  At smaller scales than this, the fluctuations are much smaller.  Size of these maps is 120 arcsec = 30 pc. 

![image](assets/D4F42FFB-AC08-41CA-AD1B-41672B95AED4.png)

So it might just be that Melnick+ are insensitive to the smallest scales, and therefore do not see the rising part of the structure function. 

Also, they seem to have removed the contribution of expanding shells before doing this analysis, which does seem unwise.  One interesting thing that they find is that the plane-of-sky sigma of velocity centroids falls very sharply with radius:

![image](assets/008D5BF9-D7AF-438C-AF46-05BDA4AAA4FD.png)

The velocity dispersion is 15 km/s in the central core (< 50 pc) and then falls rapidly to 6 km/s in the outskirts (R > 100 pc). The Castro observations are for a FOV that is 120x120 arcsec, which is 30x30 pc, so it is all within the central high-dispersion section.

They think that the motions might be dominated by gravity, for which they need a mass bigger than about 3e5 Msun.  They claim that the total mass of gas and stars is about this.  *If that is the case, then you would expect that the stars should have the same velocity dispersion as the gas. **Is this the case?***

> ### New calculations of the structure functions from the MUSE data for 30 Dor 


> I have the velocity moments for Ha and [N ɪɪ] in the `[mariano-velocity-statistics](https://github.com/will-henney/mariano-velocity-statistics)` project.  If we do use this data, we should add acknowledgment: 

> *We are grateful to Norberto Castro Rodríguez for providing maps of emission line velocity moments for 30 Doradus derived from MUSE-VLT observations.*


![image](assets/09E32BB7-2705-41AE-B867-58BEA95E48B9.png)

![image](assets/0BD3CD63-F9E5-47C1-9CBF-633E402794CB.png)

Second-order velocity structure functions for 30 Doradus derived from MUSE maps of the central 120x120 arcsec region (650 x 650 pixels, or 30 x 30 pc) for (a) Hα 6563 Å and (b) [N II] 6584 Å. 
> So we see a clear decline in the structure function at small scales, as expected.  Main conclusions from this first calculation are as follows:

> - Melnick dispersions are a bit higher than ours (I am comparing with their single-gaussian results – upper panel of Fig 15 – since that is most directly comparable).
> - We see a significant slope from 2 to 10 pc over a range that overlaps with Melnick, but they have the structure function being very flat over that range. 
> - The correlation length is about 3 pc (*note that 1 arcsec = 0.24 pc at distance to LMC*), which is much bigger than Orion (0.05 pc), but significantly smaller than NGC 604 (30 pc).
> - The total σ² is 252 km²/s², which is similar to NGC 604, but a bit higher. This is much higher than Carina (18 km²/s²). It will be interesting to look for high-luminosity Galactic regions that might bridge the gap between these
> - The power index *m*2D is about 1.22, which is very similar to Orion.  All other things being equal, we might expect that the index would be steeper in systems with higher σ² since the δ terms should become more important when the turbulence is supersonic. 
> - For the flux-weighted structure function 

### NGC 595 and 604 from Javier thesis


NGC 604 has correlation length of about 60 pc and sigma of about 10 km/s  
NGC 595 ***doesn***’***t seem to be in the thesis***

![image](assets/0FAEEC0B-3374-48F3-904D-747844663B4B.png)

The above figure is from Javier’s thesis, Fig 7.5.  I added red dashed lines to show the variance and correlation length.  The POS velocity dispersion is σ² = 10 km/s – much higher than in the Galactic regions, and nearly as high as in 30 Dor. 

## Methodology

- Use a uniform methodology for the structure function fitting. Use the functional form from Scalo that I used before
	- This doesn’t allow for negative autocorrelation (see below), but it should be ok if we exclude the largest scales
- We find 3 numbers from the structure function. 
	- An autocorrelation scale. 
	- A POS dispersion sigma. 
	- A power law index.
- The last of those doesn’t mean much, probably. But the other two would make for a nice graph. And then correlation with Luminosity and with size of region.
- What is the right scale to apply the analysis? Does it make sense to analyze sub regions of large nebulae.

### Variations with density


We can define a Stromgren scale. Spherical case will be R ~ (Q / n^2)^(1/3) where n is the ionized density (not the original density of the moloeecular cloud).

### Autocorrelation length


Justification why it occurs at S(l) = 1


- This would seem to mean autocorrelation function of C(l) = 0.5.  
- Aha, this follows from the assumed form of the autocorrelation:

![](assets/83B30862-86E3-4CD7-9A46-48DB70F2C0C2.png)

	- where m is the 2-d power index of the fluctuations
	- So by definition C(ℓ) = 0.5 when ℓ = ℓ₀ 

Here is an example from our Orion paper

![image](assets/400E7C7A-18C5-467C-937B-0EA94989757E.png)

However, sometimes the autocorrelation function goes negative.  For instance, this will happen when there is any sinusoidal variation in velocity with period L: the autocorrelation will be strongly negative on scales around 0.5 L.  And also via aliasing at 1.5 L, 2.5 L, etc.  

One way that this could be dealt with by modeling the large-scale motions and removing them first. 

### Origin of velocity structure


Melnick:2019a suggest that much of the velocity shell structure that they see in 30 Doradus is due to the *cluster wind* rather than the winds due to individual stars.  The main argument seems to be that the shells are not centered on particular stars. 

### Effects of spatial averaging


An important question to address is how σ(POS) is affected by averaging out the small scales.  As long as we resolve the correlation length, then the small scales have relatively small amplitude, so we won’t lose much power by averaging them out.    However, if the structure function is flat all the way down to our resolution limit, then the missing smaller scales may have significant power.  

We can look at this empirically by deliberately doing spatial averaging into larger pixels before calculating the dispersion. 

## Fluctuations of intensity


For Orion we did a VCA analysis (comparing power spectrum of thick and thin velocity slices), which will not necessarily be feasible for the other regions because we don’t have high enough resolution spectra.  *Although it should actually be **easier** for the more massive regions because the lines are broader, which might even allow VCA analysis with intermediate resolution spectroscopy such as MUSE.* 

**But** we can still do the thick slice analysis, which is probably the most useful part anyway.  By combining HST imaging at the small scale with wider-field ground-based imaging we should be able to get the power spectrum of intensity fluctuations over a vast range of scales. 

For some resgions, such as Orion and 30 Dor, we have good enough reddening maps that we can de-redden and remove the effect of foreground extinction. 

One useful application of this would be to show where finite angular resolution and seeing become  a factor.  This is important to address because Melnick criticized the TAURUS data for NGC 640 in this regard, and this has implications for Javier’s thesis results.  What we can do to address it is to calculate P(k) the TAURUS maps and compare this with the *ground truth* from HST images, to see at which scale they begin to deviate. 

## A cautionary tale concerning velocity fluctuations


![image](assets/AB3512C8-AA0A-4211-BBF8-3ACB3EB8E21A.png)

The above is from a survey of molecular clouds in the inner Galaxy from [Schuller:2017a](https://ui.adsabs.harvard.edu/abs/2017A%26A...601A.124S/abstract).  The colors represent different cloud complexes in 3 different spiral arms, which are at very different velocities.  Because the clouds are clumpy, we see a patchwork structure of different clouds on the plane of the sky, which gives apparent velocity fluctuations of order 50 km/s on the size scale of the clouds.  But **there are no real velocity fluctuations of that magnitude on those scales!**  There are density fluctuations, which combine with the velocity gradient from Galactic rotation. 

An important thing to bear in mind is:
> 	*The plane-of-sky fluctuations in mean velocity that we measure with the structure function may not be due to a fluctuating velocity field.  Instead, it is possible that the velocity field is completely ordered (for example, a linear gradient along line of sight) but there are emissivity fluctuations along the line-of-sight instead.*


This is somewhat related to what I tried to explain (not very successfuly) in the Orion paper: 

![image](assets/79A757D6-6190-4F7C-8B20-A2EEDACE8D54.png)

In Orion, I was trying to explain why the POS σ is less than the velocity width (LOS σ).  In the case of the Galactic molecular clouds, these σ would be about the same because the fluctuations in density are very large. 

Strangely, this is kind of the opposite of the velocity crowding effect of Lazarian and Pogosyan.  That is saying that:
> 	O*bserved structure in thin velocity slices is not necessarily real **density** structure, since it might be produced by velocity fluctuations in a constant density medium (velocity caustics)*


Or, in other words, that “velocity fluctuations can be mistaken for density fluctuations **[Lazarian]**”, whereas we are saying that “density fluctuations can be mistaken for velocity fluctuations **[Will]**”.  Possibly, both are true!

Note that the Lazarian view is controversial, when applied to structure visible in H I narrow velocity slices.  See [Yuen:2019a](https://ui.adsabs.harvard.edu/abs/2019arXiv190403173Y/abstract) and [Clark:2019a](https://ui.adsabs.harvard.edu/abs/2019ApJ...874..171C/abstract) and [Yuen:2020a](https://ui.adsabs.harvard.edu/abs/2020arXiv201215776Y/abstract)

## Relation with other fields

- Turbulence in the H I gas in the Galaxy.  For instance, [Kalberla:2019a](https://ui.adsabs.harvard.edu/abs/2019A&A...627A.112K)
- Turbulence in the Reynolds layer of WIM, [Chepurnov:2010a](https://ui.adsabs.harvard.edu/abs/2010ApJ...710..853C)
	- Uses Ha data from WHAM (intensity fluctuations, not velocity)
	- Finds that it is consistent with a big power law in electron density fluctuatiions from scales of 100 pc down to 1000 km

### Application of other techniques


This is not for first paper, but might be worthwhile exploring afterwards


- Velocity Decomposition Algorithm (VDA) – Yuen:2020a
	- Supposedly allows one to separate out the velocity fluctuations from the intensity fluctuations
	- Works even when the velocity dispersion is subsonic 
