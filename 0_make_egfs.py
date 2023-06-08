import numpy as np
import obspy
import time
import matplotlib.pyplot as plt
import os
import glob
import json
import multiprocessing

from EgfLib import FTAN, station_pair_stack, egf, regional_dispersion

def daily_to_egf(daily_paths,stack_type,pws_power):
    """
    Computes EGF from a list of daily cross correlations.
    * Reads cross correlations from disk as an obspy stream
    * Stacks the cross corelations using linear or time domain phase weighted stacking
    * Converts to an EGF using the symetrical component
    * Returns egf as an obspy trace object
    """
    cc_stream = obspy.read(daily_paths)
    #
    cc_stacked_trace = station_pair_stack(cc_stream=cc_stream,stack_type=stack_type,pws_power=pws_power)
    #
    egf_trace = egf(cc_stacked_trace)
    return egf_trace

def egf_worker(outdir,station_pair,daily_paths,stack_type,pws_power):
    """
    Worker function that calls daily_to_egf to compute egf for station pair
    then saves the output.
    """
    egf_trace = daily_to_egf(daily_paths)
    egf_trace.write(f"{outdir}/{station_pair}.mseed")
    return station_pair

#######################################################################
#                             Settings                                #
#######################################################################


# Construct cross correlation list so it is a list of 
# lists of paths for each station pair
# cross_correlation_list = [
#   [List of paths],
#   ...
#   [List of paths]
#   ]
cross_correlations_list = []
# Construct station_pair_list as a list of strings with
# station pair names. This must be in the same order as
# cross_corelation_list
# station_pair = "station1_station2"
station_pair_list = []

outdir = ""
    
#
stack_type="pws"
pws_power = 2

threads = 20

#######################################################################
#                               Main                                  #
#######################################################################

if __name__=="__main__":
    with multiprocessing.Pool(threads) as pool:
        procs = []
        for station_pair, daily_paths in zip(station_pair_list,cross_correlations_list):
            p = pool.apply_async(egf_worker,args=(outdir,station_pair,daily_paths,stack_type,pws_power))
            procs.append(p)
        for p in procs:
            station_pair = p.get()
            print(f"Done station pair {station_pair}")
        pool.join()
        pool.close()
    