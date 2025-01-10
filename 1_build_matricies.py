import numpy as np
import xarray as xr
import obspy
import obspy.geodetics
import time
import matplotlib.pyplot as plt
import os
import glob
import json
import multiprocessing
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

from pyphasepick.frequencytimeanalisys import calc_and_save_ftan

#######################################################################
#                             Settings                                #
#######################################################################

# CSV file containing all of the station infomation (see example for details)
stations_csv = "./example/all_stations.csv"

# CSV file containing all of the station pair infomation (see example for details)
stationpair_csv = "./example/all_station_pairs.csv"

# Directory that the empirical Green's functions are saved in
egf_dir = "./example/pws"

# Component to compute (ZZ, TT, RR)
comp = "ZZ"

# Gives option to ignore network code and replace with a new code
#   Useful if some stations have their network changed over time
#   If used the replacement code must be listed in stations_csv rather than the original
ignore_network = True
replacement_net_code = "AJ"

# Minimum interstation distance to compute for in meters
min_dist = 12_000

#Filter Settings
#Period Axis
minT = 1
maxT = 10.0
dT = 0.25
#Velocity Axis
dv = 0.001
minv = 1.0
maxv = 3.5
#Filter width
width_type = "dependent" # "dependent" or "fixed"
bandwidth = 0.4 # If dependent this will be 0.4*central_period and if fixed will be 0.4 s

# The directory to save the output FTAN matricies
outdir = "./example/ftan"

# Number of threads Pool object will use to parallelise the process
threads = 4

#######################################################################
#                               Main                                  #
#######################################################################

stationsdf = pd.read_csv(stations_csv)
stationpairdf = pd.read_csv(stationpair_csv)
vel_type = "phase"
fSettings = (minT,maxT,dT,bandwidth,width_type,dv,minv,maxv)

if __name__=="__main__":
    os.makedirs(f"{outdir}",exist_ok=True)
    os.makedirs(f"{outdir}/{vel_type}",exist_ok=True)
    os.makedirs(f"{outdir}/{vel_type}/{comp}",exist_ok=True)
    full_outdir = f"{outdir}/{vel_type}/{comp}"
    for file in glob.glob(f"{full_outdir}/*"):
        os.remove(file)
    #
    egf_pathlist = []
    distance_list = []
    if comp == "RR" or comp == "TT":
        HorV = "TT"
    else:
        HorV = comp
    for row in stationpairdf.iterrows():
        if row[1][HorV]:
            sta1 = row[1]["station1"]
            net1 = row[1]["net1"]
            sta2 = row[1]["station2"]
            net2 = row[1]["net2"]
            dist = row[1]["gcm"]
            if ignore_network:
                net1 = replacement_net_code
                net2 = replacement_net_code
            #
            egf_path = f"{egf_dir}/EGF/{comp}/{net1}_{sta1}_{net2}_{sta2}.mseed"
            #
            if os.path.isfile(egf_path) and dist >= min_dist:
                #
                # print(egf_path)
                egf_pathlist.append(egf_path)
                distance_list.append(dist)
    #
    # raise KeyboardInterrupt
    print(f"Starting {len(egf_pathlist)} ftan jobs, saving output to {full_outdir}")
    snr_df = {"station_pair" : [], "snr" : []}
    multiprocessing.freeze_support()
    with multiprocessing.Pool(threads) as pool:
        procs = []
        for path, distance in zip(egf_pathlist,distance_list):
            p = pool.apply_async(calc_and_save_ftan,args=(path,full_outdir,distance,fSettings,vel_type))
            procs.append(p)
        for p in tqdm(procs):
            station_pair, snr = p.get()
            # print(snr)
            # print(f"Done station pair {station_pair}")
            snr_df["station_pair"].append(station_pair)
            snr_df["snr"].append(snr)
    snr_df = pd.DataFrame(data=snr_df)
    snr_df.to_csv(f"{outdir}/{vel_type}_{comp}_snr_df.csv")
