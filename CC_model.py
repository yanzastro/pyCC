# This file contains classes to calculate angular cross-correlations
# with Limber approximation and Halo Model.
# One can config the line-of-sight kernel and profile according to the tracer.

import pyccl as ccl
import numpy as np

class tracer_profile:

    '''
    This is a class defining information of some profile (currently only supports NFW and GNFW) 
    'cosmo' is a cosmology class of pyccl;
    'tracer' is a tracer class of pyccl;
    'kwargs' includes other parameters of the profile.
    '''
    
    def __init__(self, cosmo, tracer, prof_name='NFW', **kwargs):
        self.tracer = tracer
        self.prof_name = prof_name
        self.cosmo = cosmo
        self.massdef = ccl.halos.MassDef(200, 'critical')
        self.conc = ccl.halos.ConcentrationDuffy08(self.massdef)
        if prof_name is 'NFW':
            self.prof = ccl.halos.profiles.HaloProfileNFW(self.conc)
        elif prof_name is 'GNFW':
            mb = 1 - kwargs['b_hydro']
            self.prof = ccl.halos.profiles.HaloProfilePressureGNFW(mass_bias=mb)
        return

class limber_hm_mps:

    '''
    This class defines ingredient needed to calculate the cross-correlation.
    '''

    def __init__(self, cosmo, **kwargs):

        self.cosmo=cosmo
        self.massdef = ccl.halos.MassDef(200, 'critical')
        self.conc = ccl.halos.ConcentrationDuffy08(self.massdef)
        
        #self.massdef = ccl.halos.MassDef(200, 'critical')
        #self.conc = ccl.halos.ConcentrationDuffy08(self.massdef)
        
        self.hmf = ccl.halos.hmfunc.MassFuncTinker08(self.cosmo, self.massdef)
        self.hmb = ccl.halos.hbias.HaloBiasTinker10(self.cosmo, self.massdef)
        self.hmc = ccl.halos.halo_model.HMCalculator(self.cosmo, self.hmf, self.hmb,
                                                     self.massdef,
                                                     log10M_min=6.0,
                                                     log10M_max=17.0,)

    def calc_ang_ps(self, ell, prof1, prof2):
        k_arr = np.geomspace(1e-4, 1e2, 256)
        a_arr = np.linspace(0.2, 1, 64)

        self.Pk2D = ccl.halos.halomod_Pk2D(self.cosmo, self.hmc, prof1.prof, prof2=prof2.prof, 
                                           normprof1=(prof1.prof_name!='GNFW'),
                                           normprof2=(prof2.prof_name!='GNFW'),
                                           lk_arr=np.log(k_arr), a_arr=a_arr,)

        tracer1 = prof1.tracer
        tracer2 = prof2.tracer
        return ccl.angular_cl(self.cosmo, tracer1, tracer2, ell, self.Pk2D)
