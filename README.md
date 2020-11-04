# pyCC
My pipeline to measure cross-correlation between Healpix and model them. One can call emcee_demo to run model fitting.
The codes are written in python. Maps are in Healpix format so healpy package is needed: https://healpy.readthedocs.io/en/latest/
Cross-correlations are measured with PolSpice package, with a python interface. See http://www2.iap.fr/users/hivon/software/PolSpice/
Cross-correlation models are calculated using PYCCL: https://pypi.org/project/pyccl/
