# This script is an example to produce Jackknife samples of
# Healpix maps for covariance matrix estimation.
# The Jackknife samples are defined as masking out pixels with
# a low Nside.

from  __future__ import division
import healpy as hp
import numpy as np

Ns = 1024 # Nside of origional map

mapsample = np.arange(hp.nside2npix(Ns))
masksample = np.ones_like(mapsample)

Ns_out = 32 # mask out pixels with Nside=32 in turn to creat Jackknife samples

mask_dg = hp.ud_grade(masksample, Ns_out)
jid = 0

for j in range(hp.nside2npix(Ns_out)):
    if mask_dg[j] == 0:
        continue
    else:
        print('Jack: '+str(jid))
        jackmask = np.ones(hp.nside2npix(Ns_out))
        jackmask[j] = 0
        jackmask_ud = hp.ud_grade(jackmask, Ns)
        mask_jack = jackmask_ud * masksample
        print jid
        hp.write_map('mask_jack'+str(jid)+'_'+str(Ns)+'.fits', mask_jack, overwrite=True)
        jid += 1
