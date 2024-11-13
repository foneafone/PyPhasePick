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


#######################################################################
#                             Settings                                #
#######################################################################

if __name__=="__main__":
    # cc_path_structure = "/raid2/jwf39/askja/STACKS/01/001_DAYS/COMP/NET1_STA1_NET2_STA2/YEAR-MM-DD.MSEED"
    cc_path_structure = "/raid2/jwf39/askja/ALL_STACKS/COMP/NET1_STA1_NET2_STA2/YEAR-MM-DD.MSEED"

    # stations_csv = "/raid2/jwf39/askja/notebooks/pre_21_all_stations.csv"
    stations_csv = "/raid2/jwf39/askja/notebooks/all_stations_sep23.csv"

    ignore_network = True
    replacement_net_code = "AJ"

    comps = ("ZZ","RR","TT")
    # comps = ("ZZ")
    startdate = "1970-01-01" # "2021-08-01" or "1970-01-01"
    enddate = "2035-01-01" # "2021-07-01" or "2035-01-01"

    stack_type="pws" # linear or pws
    # stack_type = "linear"
    pws_power = 2

    save_cc = True

    remake = True

    #Will save in this directory:
    # outdir = f"/raid1/jwf39/askja/sep11_jul21/pws"
    outdir = f"/raid2/jwf39/askja/sep11_sep23/pws"
    # outdir = f"/raid2/jwf39/askja/aug21_sep23/pws"

    threads = 20

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
    