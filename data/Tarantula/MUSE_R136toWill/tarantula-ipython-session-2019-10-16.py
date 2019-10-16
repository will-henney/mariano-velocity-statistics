>>> from astropy.io import fits
>>> hdulist = fits.open("GAUS_Ha6562.8_060_Will.fits")
>>> from astropy.coordinates import SkyCoord
>>> vhdu = hdulist[2]
>>> from astropy.wcs
>>> from astropy.wcs import WCS
>>> w = WCS(vhdu)
>>> w
WCS Keywords

Number of WCS axes: 3
CTYPE : 'RA---TAN'  'DEC--TAN'  'AWAV'  
CRVAL : 84.688309  -69.105970125  4.5999873046875003e-07  
CRPIX : 220.92793852  213.660785628  1.0  
CD1_1 CD1_2 CD1_3  : -5.52742276729e-05  -9.66418424553e-08  0.0  
CD2_1 CD2_2 CD2_3  : -4.91681964084e-07  5.53153613478e-05  0.0  
CD3_1 CD3_2 CD3_3  : 0.0  0.0  1.25e-10  
NAXIS : 650  650
>>> w = w.celestial
>>> w
WCS Keywords

Number of WCS axes: 2
CTYPE : 'RA---TAN'  'DEC--TAN'  
CRVAL : 84.688309  -69.105970125  
CRPIX : 220.92793852  213.660785628  
CD1_1 CD1_2  : -5.52742276729e-05  -9.66418424553e-08  
CD2_1 CD2_2  : -4.91681964084e-07  5.53153613478e-05  
NAXIS : 650  650
>>> w.wcs
       flag: 137
      naxis: 2
      crpix: 0x7fa20c7d80a0
               220.93       213.66    
         pc: 0x7fa20c7c0980
    pc[0][]:  -5.5274e-05  -9.6642e-08
    pc[1][]:  -4.9168e-07   5.5315e-05
      cdelt: 0x7fa20c7e89f0
               1.0000       1.0000    
      crval: 0x7fa20c721830
               84.688      -69.106    
      cunit: 0x7fa20df2eb70
             "deg"
             "deg"
      ctype: 0x7fa20df2f5c0
             "RA---TAN"
             "DEC--TAN"
    lonpole: 180.000000
    latpole: -69.105970
    restfrq: 0.000000
    restwav: 0.000000
        npv: 1
     npvmax: 1
         pv: 0x7fa20c798270
               2   1   0.0000    
        nps: 0
     npsmax: 0
         ps: 0x0
         cd: 0x7fa20c7bbb00
    cd[0][]:  -5.5274e-05  -9.6642e-08
    cd[1][]:  -4.9168e-07   5.5315e-05
      crota: 0x7fa20c7e37c0
               0.0000       0.0000    
     altlin: 2
     velref: 0
        alt: ' '
     colnum: 0
      colax: 0x7fa20c7f6540
                 0      0
      cname: 0x7fa20df48230
             UNDEFINED
             UNDEFINED
      crder: 0x7fa20c7e66a0
               UNDEFINED    UNDEFINED
      csyer: 0x7fa20c7ec770
              1.5103e-05   5.7677e-06
      czphs: 0x7fa20c7d1eb0
               UNDEFINED    UNDEFINED
      cperi: 0x7fa20c7ce6e0
               UNDEFINED    UNDEFINED
    wcsname: UNDEFINED
    timesys: UNDEFINED
    trefpos: UNDEFINED
    trefdir: UNDEFINED
    plephem: UNDEFINED
   timeunit: UNDEFINED
    dateref: UNDEFINED
     mjdref:        UNDEFINED       UNDEFINED
   timeoffs: UNDEFINED
    dateobs: UNDEFINED
    datebeg: UNDEFINED
    dateavg: UNDEFINED
    dateend: UNDEFINED
     mjdobs: UNDEFINED
     mjdbeg: UNDEFINED
     mjdavg: UNDEFINED
     mjdend: UNDEFINED
     jepoch: UNDEFINED
     bepoch: UNDEFINED
     tstart: UNDEFINED
      tstop: UNDEFINED
    xposure: UNDEFINED
    telapse: UNDEFINED
    timsyer: UNDEFINED
    timrder: UNDEFINED
    timedel: UNDEFINED
   timepixr: UNDEFINED
     obsgeo:        UNDEFINED       UNDEFINED       UNDEFINED
                    UNDEFINED       UNDEFINED       UNDEFINED
   obsorbit: UNDEFINED
    radesys: "FK5"
    equinox:   2000.000000000
    specsys: UNDEFINED
    ssysobs: UNDEFINED
    velosys: UNDEFINED
    zsource: UNDEFINED
    ssyssrc: UNDEFINED
    velangl: UNDEFINED
       ntab: 0
        tab: 0x0
       nwtb: 0
        wtb: 0x0
      types: 0x7fa20c7f4150
            2200 2201
     lngtyp: "RA"
     lattyp: "DEC"
        lng: 0
        lat: 1
       spec: -1
   cubeface: -1
        err: 0x0
        lin: (see below)
        cel: (see below)
        spc: (see below)
     m_flag: 137
    m_naxis: 2
    m_crpix: 0x7fa20c7d80a0  (= crpix)
       m_pc: 0x7fa20c7c0980  (= pc)
    m_cdelt: 0x7fa20c7e89f0  (= cdelt)
    m_crval: 0x7fa20c721830  (= crval)
    m_cunit: 0x7fa20df2eb70  (= cunit)
    m_ctype: 0x7fa20df2f5c0  (= ctype)
       m_pv: 0x7fa20c798270  (= pv)
       m_ps: 0x0  (= ps)
       m_cd: 0x7fa20c7bbb00  (= cd)
    m_crota: 0x7fa20c7e37c0  (= crota)

    m_colax: 0x7fa20c7f6540  (= colax)
    m_cname: 0x7fa20df48230  (= cname)
    m_crder: 0x7fa20c7e66a0  (= crder)
    m_csyer: 0x7fa20c7ec770  (= csyer)
    m_czphs: 0x7fa20c7d1eb0  (= czphs)
    m_cperi: 0x7fa20c7ce6e0  (= cperi)
      m_tab: 0x0  (= tab)
      m_wtb: 0x0  (= wtb)

   lin.*
       flag: 137
      naxis: 2
      crpix: 0x7fa20c7d80a0
               220.93       213.66    
         pc: 0x7fa20c7c0980
    pc[0][]:  -5.5274e-05  -9.6642e-08
    pc[1][]:  -4.9168e-07   5.5315e-05
      cdelt: 0x7fa20c7e89f0
               1.0000       1.0000    
     dispre: 0x0
     disseq: 0x0
piximg[0][]:  -5.5274e-05  -9.6642e-08
piximg[1][]:  -4.9168e-07   5.5315e-05
imgpix[0][]:  -18091.      -31.607    
imgpix[1][]:  -160.81       18078.    
    i_naxis: 2
      unity: 0
     affine: 1
     simple: 0
        err: 0x0
     tmpcrd: 0x7fa20c7cf520
     m_flag: 0
    m_naxis: 0
    m_crpix: 0x0
       m_pc: 0x0
    m_cdelt: 0x0
   m_dispre: 0x0
   m_disseq: 0x0

   cel.*
      flag: 137
     offset: 0
       phi0:  0.000000
     theta0: 90.000000
        ref:   84.688      -69.106       180.00      -69.106    
        prj: (see below)
      euler:   84.688       159.11       180.00      -0.93424      0.35664   
    latpreq: 0 (not required)
     isolat: 0
        err: 0x0

   prj.*
       flag: 103
       code: "TAN"
         r0: 57.295780
         pv: (not used)
       phi0:  0.000000
     theta0: 90.000000
     bounds: 7

       name: "gnomonic"
   category: 1 (zenithal)
    pvrange: 0
  simplezen: 1
  equiareal: 0
  conformal: 0
     global: 0
  divergent: 1
         x0: 0.000000
         y0: 0.000000
        err: 0x0
        w[]:   0.0000       0.0000       0.0000       0.0000       0.0000    
               0.0000       0.0000       0.0000       0.0000       0.0000    
          m: 0
          n: 0
     prjx2s: 0x0115d95270
     prjs2x: 0x0115d957d0

   spc.*
       flag: 0
       type: "    "
       code: "   "
      crval: UNDEFINED
    restfrq: 0.000000
    restwav: 0.000000
         pv: (not used)
          w:   0.0000       0.0000       0.0000      (remainder unused)
    isGrism: 0
        err: 0x0
     spxX2P: 0x0
     spxP2S: 0x0
     spxS2P: 0x0
     spxP2X: 0x0
>>> w.pixel_to_world
<bound method HighLevelWCSMixin.pixel_to_world of WCS Keywords

Number of WCS axes: 2
CTYPE : 'RA---TAN'  'DEC--TAN'  
CRVAL : 84.688309  -69.105970125  
CRPIX : 220.92793852  213.660785628  
CD1_1 CD1_2  : -5.52742276729e-05  -9.66418424553e-08  
CD2_1 CD2_2  : -4.91681964084e-07  5.53153613478e-05  
NAXIS : 650  650>
>>> w.pixel_to_world?
>>> c = w.pixel_to_world
>>> c = w.pixel_to_world([0, 0])
>>> c = w.pixel_to_world([0], [0])
>>> c
<SkyCoord (FK5: equinox=2000.0): (ra, dec) in deg
    [(84.72247053, -69.11762201)]>
>>> c = w.pixel_to_world([0, 1, 3, 4, 5], [0, 0, 0, 0, 0])
>>> c
<SkyCoord (FK5: equinox=2000.0): (ra, dec) in deg
    [(84.72247053, -69.11762201), (84.72231546, -69.11762253),
     (84.72200533, -69.11762357), (84.72185026, -69.1176241 ),
     (84.72169519, -69.11762462)]>
>>> c[0]
<SkyCoord (FK5: equinox=2000.0): (ra, dec) in deg
    (84.72247053, -69.11762201)>
>>> c[1]
<SkyCoord (FK5: equinox=2000.0): (ra, dec) in deg
    (84.72231546, -69.11762253)>
>>> c[0].separation(c[1])
<Angle 5.52764109e-05 deg>
>>> import astropy.units as u
>>> c[0].separation(c[1]).to(u.arcsec)
<Angle 0.19899508 arcsec>
>>> X, Y = np.meshgrid( np.arange(w.naxis[0]), np.arange(w.naxis[1]))
>>> import numpy as np
>>> X, Y = np.meshgrid( np.arange(w.naxis[0]), np.arange(w.naxis[1]))
>>> xi = np.arange(w.naxis[0])
>>> w.naxis
2
>>> ny, nx = vhdu.shape
>>> ny
650
>>> X, Y = np.meshgrid( np.arange(nx), np.arange(ny))
>>> X
array([[  0,   1,   2, ..., 647, 648, 649],
       [  0,   1,   2, ..., 647, 648, 649],
       [  0,   1,   2, ..., 647, 648, 649],
       ...,
       [  0,   1,   2, ..., 647, 648, 649],
       [  0,   1,   2, ..., 647, 648, 649],
       [  0,   1,   2, ..., 647, 648, 649]])
>>> Y
array([[  0,   0,   0, ...,   0,   0,   0],
       [  1,   1,   1, ...,   1,   1,   1],
       [  2,   2,   2, ...,   2,   2,   2],
       ...,
       [647, 647, 647, ..., 647, 647, 647],
       [648, 648, 648, ..., 648, 648, 648],
       [649, 649, 649, ..., 649, 649, 649]])
>>> c = w.pixel_to_world(X, Y)
>>> c
<SkyCoord (FK5: equinox=2000.0): (ra, dec) in deg
    [[(84.72247053, -69.11762201), (84.72231546, -69.11762253),
      (84.72216039, -69.11762305), ..., (84.62214034, -69.11793079),
      (84.62198527, -69.11793122), (84.6218302 , -69.11793166)],
     [(84.72247017, -69.11756669), (84.7223151 , -69.11756721),
      (84.72216004, -69.11756774), ..., (84.62214024, -69.11787548),
      (84.62198516, -69.11787591), (84.62183009, -69.11787634)],
     [(84.72246981, -69.11751138), (84.72231475, -69.1175119 ),
      (84.72215968, -69.11751242), ..., (84.62214013, -69.11782016),
      (84.62198506, -69.11782059), (84.62182999, -69.11782103)],
     ...,
     [(84.72223956, -69.08183301), (84.72208475, -69.08183353),
      (84.72192993, -69.08183405), ..., (84.62207336, -69.08214171),
      (84.62191854, -69.08214214), (84.62176372, -69.08214258)],
     [(84.72223921, -69.0817777 ), (84.72208439, -69.08177822),
      (84.72192958, -69.08177874), ..., (84.62207326, -69.0820864 ),
      (84.62191844, -69.08208683), (84.62176362, -69.08208726)],
     [(84.72223885, -69.08172238), (84.72208404, -69.0817229 ),
      (84.72192922, -69.08172342), ..., (84.62207315, -69.08203108),
      (84.62191834, -69.08203151), (84.62176352, -69.08203194)]]>
>>> %hist
>>> %hist?
>>> %history?
>>> %history -o -f tarantula-ipython-session-2019-10-16.py
>>> %history?
>>> %history -o -p -f tarantula-ipython-session-2019-10-16.py
