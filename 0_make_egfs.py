import numpy as np
import obspy
import time
import matplotlib.pyplot as plt
import os
import glob
import json
import multiprocessing
import pandas as pd

from pyphasepick.stackingandegf import egf_worker, make_stack_jobs


#######################################################################
#                             Settings                                #
#######################################################################

if __name__=="__main__":
    # Path structure of cross correlation output from msnoise
    # COMP = component (NN, EE, ZZ, etc.)
    # NET1/2 = Network
    # STA1/2 = Station
    # YEAR = Year
    # MM = Month (03, 11, etc.)
    # DD = Day (04, 24, etc.)
    cc_path_structure = "./example/STACKS/COMP/NET1_STA1_NET2_STA2/YEAR-MM-DD.MSEED"

    # CSV file containing all of the station infomation (see example for details)
    stations_csv = "./example/all_stations.csv"

    # Gives option to ignore network code and replace with a new code
    #   Useful if some stations have their network changed over time
    #   If used the replacement code must be listed in stations_csv rather than the original
    ignore_network = True
    replacement_net_code = "AJ"

    # Output components to compute
    comps = ("ZZ","RR","TT")

    # Start and end date to stack between
    startdate = "1970-01-01" 
    enddate = "2040-01-01" 

    # Type of stacking used - can be phase weighted stacking (pws) or linear
    stack_type="pws" # linear or pws
    # Power of the phase weighted stack
    pws_power = 2

    # Save the output cross correlations as well as the empirical Green's functions
    save_cc = True

    # Re-stack cross correlations - can be set to false to stop the re-calculation
    #   of EGFs that already exist in the output directory
    remake = True

    #Will save in this directory:
    outdir = f"./example/pws"

    # Number of threads that the Pool will used to parallelise the jobs
    threads = 4

#######################################################################
#                               Main                                  #
#######################################################################

if __name__=="__main__":
    print("Generating jobs list.")
    job_list = make_stack_jobs(stations_csv,cc_path_structure,startdate,enddate,comps,outdir,remake,ignore_net=ignore_network,fake_net=replacement_net_code)
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
    for job in job_list:
        print(job)
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
    