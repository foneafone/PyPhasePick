import numpy as np
import obspy
import time
import matplotlib.pyplot as plt
import os
import glob
import json
import multiprocessing
import pandas as pd

from EgfLib import FTAN, station_pair_stack, egf_worker, make_stack_jobs

# def daily_to_egf_zz(daily_paths,stack_type,pws_power):
#     """
#     Computes EGF from a list of daily cross correlations.
#     * Reads cross correlations from disk as an obspy stream
#     * Stacks the cross corelations using linear or time domain phase weighted stacking
#     * Converts to an EGF using the symetrical component
#     * Returns egf as an obspy trace object
#     """
#     cc_stream = obspy.read(daily_paths)
#     #
#     cc_stacked_trace = station_pair_stack(cc_stream=cc_stream,stack_type=stack_type,pws_power=pws_power)
#     #
#     egf_trace = egf(cc_stacked_trace)
#     return egf_trace

# def daily_to_egf_ttrr(daily_paths,stack_type,pws_power):
#     """
#     Computes EGF from a list of daily cross correlations.
#     * Reads cross correlations from disk as an obspy stream
#     * Stacks the cross corelations using linear or time domain phase weighted stacking
#     * Converts to an EGF using the symetrical component
#     * Returns egf as an obspy trace object
#     """
#     cc_stream = obspy.read(daily_paths)
#     #
#     cc_stacked_trace = station_pair_stack(cc_stream=cc_stream,stack_type=stack_type,pws_power=pws_power)
#     #
#     egf_trace = egf(cc_stacked_trace)
#     return egf_trace

# def egf_worker(outdir,station_pair,daily_paths,stack_type,pws_power):
#     """
#     Worker function that calls daily_to_egf to compute egf for station pair
#     then saves the output.
#     """
#     egf_trace = daily_to_egf_zz(daily_paths,stack_type,pws_power)
#     egf_trace.write(f"{outdir}/{station_pair}.mseed")
#     return station_pair

#######################################################################
#                             Settings                                #
#######################################################################

if __name__=="__main__":
    cc_path_structure = "/raid1/jwf39/askja/STACKS/01/001_DAYS/COMP/NET1_STA1_NET2_STA2/YEAR-MM-DD.MSEED"

    stations_csv = "/raid1/jwf39/askja/STATIONS/askja_stations.csv"

    comps = ("ZZ","RR","TT")
    startdate = "1970-01-01"
    enddate = "2021-07-01"

    stack_type="pws" # linear or pws
    # stack_type = "linear"
    pws_power = 2

    save_cc = True

    remake = False

    #Will save in this directory:
    # outdir = f"/raid1/jwf39/askja/pre_jul21/linear"
    outdir = f"/raid1/jwf39/askja/pre_jul21/pws"

    threads = 20

#######################################################################
#                               Main                                  #
#######################################################################

if __name__=="__main__":
    print("Generating jobs list.")
    job_list = make_stack_jobs(stations_csv,cc_path_structure,startdate,enddate,comps,outdir,remake)
    # print(len(job_list))
    #
    print("Making output directories")
    if not os.path.isdir(f"{outdir}/EGF"):
        os.mkdir(f"{outdir}/EGF")
    if save_cc and not os.path.isdir(f"{outdir}/CC"):
        os.mkdir(f"{outdir}/CC")
        os.mkdir(f"{outdir}/CC/TT")
        os.mkdir(f"{outdir}/CC/RR")
        os.mkdir(f"{outdir}/CC/TR")
        os.mkdir(f"{outdir}/CC/RT")
        os.mkdir(f"{outdir}/CC/ZZ")
    for comp in comps:
        if not os.path.isdir(f"{outdir}/EGF/{comp}"):
            os.mkdir(f"{outdir}/EGF/{comp}")
    #
    print(f"Starting {len(job_list)} stacking jobs.")
    multiprocessing.freeze_support()
    with multiprocessing.Pool(threads) as pool:
        procs = []
        for job in job_list:
            p = pool.apply_async(egf_worker,args=(job,save_cc,outdir,stack_type,pws_power))
            procs.append(p)
        for p in procs:
            station_pair = p.get()
            print(f"Done station pair {station_pair}")
        pool.close()
        pool.terminate()
    