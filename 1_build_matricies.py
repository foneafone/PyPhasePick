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

stations_csv = "/raid2/jwf39/askja/notebooks/all_stations_sep23.csv"
stationsdf = pd.read_csv(stations_csv)

stationpair_csv = "/raid2/jwf39/askja/notebooks/all_station_pairs_sep23.csv"
stationpairdf = pd.read_csv(stationpair_csv)

egf_dir = "/raid2/jwf39/askja/sep11_sep23/pws"

vel_type = "phase" # "phase" or "group"
comp = "ZZ"

net = "AJ"

min_dist = 2*3000*2
snr_threshold = 2.0

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
divalpha = 5.0

fSettings = (minT,maxT,dT,bandwidth,width_type,dv,minv,maxv,divalpha)

outdir = f"/raid2/jwf39/askja/sep11_sep23/ftan"

threads = 15

#######################################################################
#                               Main                                  #
#######################################################################

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
            sta2 = row[1]["station2"]
            dist = row[1]["gcm"]
            #
            egf_path = f"{egf_dir}/EGF/{comp}/{net}_{sta1}_{net}_{sta2}.mseed"
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
