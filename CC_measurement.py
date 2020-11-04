# This code defines some functions that call polspice to measure C_ell
# and functions that are needed for binning C_ell's
# as well as calculating Jackknife covariance matrices.

import numpy as np
import subprocess
import os

def run_polspice(polspice_path, n_threads=20, verbose_output=True, 
                 debug_output_file=None, **kwargs):
    '''
    This function calls Polspice to calculate C_l's between two maps. 
    The information of maps are encoded in **kwargs which has a format:
    params={"mapfile" : map1, # path to 1st map  
               "maskfile" : mask1,
               #"weightfile": weight1,
               "mapfile2" : map2,
               "maskfile2" : mask2,
               'beam': 0, # beam size to be corrected in arcmin
               'beam2': 0,
               "clfile" : out_cl_gc,
               #"corfile" : out_corr_gc,
               #"apodizesigma": 60,
               #"thetamax": 60,
               "nlmax" : 3000,
               #"decouple" : "YES",
               "verbosity" : 0}
    For more information about these parameters see instruction of Polspice
    '''

    args = [polspice_path,]
    for option, param in kwargs.items():
        args.append("-{}".format(option))
        args.append("{}".format(param))

    env = {"OMP_THREAD_LIMIT" : "{}".format(n_threads), 
           "HEALPIX" : os.environ["HEALPIX"], 
           "LD_LIBRARY_PATH" : os.environ["LD_LIBRARY_PATH"]       
    }
    if verbose_output: print(" ".join(args))
    process = subprocess.Popen(args, stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, 
                                     env=env)
    #shell=True)
    status = process.poll()
    log = []
    debug_file = None
    if debug_output_file != None:
        debug_file = open(debug_output_file, "w")

    while status is None:
        l = str(process.stdout.readline())
        if verbose_output:
            print(l.strip("\n"))
        if debug_file:
            debug_file.write(l)
            debug_file.flush()
        status = process.poll()

    if debug_file:
        debug_file.close()

    if status != 0:
        print("Something went seriously wrong:", status)
        raise RuntimeError


def bin_C_ell(ell, C_ell, bin_edges):
    '''
    This function bins C_ell into l_bins defined by bin_edges.
    It returns l_bin centers and binned C_ell as well as std of C_ell in each bin.
    '''
    n_bin = bin_edges.size - 1
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
    Cl_binned = np.zeros(n_bin)
    err = np.zeros(n_bin)
    ell_mean = np.zeros_like(Cl_binned)

    #cell_weightd, bedge = np.histogram(ell, bins=bin_edges, weights=C_ell)
    #ell_bins, bedge = np.histogram(ell, bins=bin_edges)
    #bins = cell_weightd / ell_bins
    for i in range(n_bin):
        mask = np.logical_and(bin_edges[i] < ell, ell <= bin_edges[i+1])
        Cl_binned[i] = np.mean(C_ell[mask])
        err[i] = np.std(C_ell[mask])/np.sqrt(1.0*np.count_nonzero(mask))
        ell_mean[i] = np.mean(ell[mask])
    return bin_centers, Cl_binned, err 


def read_jack_cc(n_jack, fname_front, fname_back, bin_edges):
    '''
    Filename format: 'fname_front'+str(jack_id)+'fname_back'
    e.g: gy_cross_1.cl has fname_front='gy_cross_'; fname_back='.cl'
    The file format is the same as Polspice output.
    '''
    n_bins = bin_edges.size - 1
    clbins_jack = np.zeros((n_jack, n_bins))
    for i in range(n_jack):
        cc_file = np.loadtxt(fname_front+str(i)+fname_back)
        cc_jack = cc_file.T[1]
        ell= cc_file.T[0]
        clbins_jack[i], lbin_centers, cl_err = bin_C_ell(ell,
                                                            cc_jack,
                                                            bin_edges) 

    cc_binned_mean = np.mean(clbins_jack, axis=0)
    cc_binned_cov = np.cov(clbins_jack.T) * n_jack
    return lbin_centers, cc_binned_mean, cc_binned_cov
