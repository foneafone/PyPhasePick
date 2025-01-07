import numpy as np
import obspy
import time
import matplotlib.pyplot as plt
import os
import glob
import json
import multiprocessing
import pandas as pd
from tqdm import tqdm

from pyphasepick.picking import auto_phase_picker, auto_group_picker

#######################################################################
#                             Settings                                #
#######################################################################


stations_csv = "/raid2/jwf39/askja/notebooks/all_stations_sep23.csv"
stationsdf = pd.read_csv(stations_csv)

stationpair_csv = "/raid2/jwf39/askja/notebooks/all_station_pairs_sep23.csv"
stationpairdf = pd.read_csv(stationpair_csv)

egf_dir = "/raid2/jwf39/askja/sep11_sep23/pws"
matrix_dir = "/raid2/jwf39/askja/sep11_sep23/ftan"

import sys
args = sys.argv

if len(sys.argv) > 1:
    vel_type = sys.argv[1]
    comp = sys.argv[2]
    stopping_threshold = (float(sys.argv[3]),float(sys.argv[4]))
else:
    vel_type = "phase" # "phase" or "group"
    comp = "ZZ"
    stopping_threshold = (0.25,0.05,0.02, 5) # for phase: starting, decreasing T, increasing T
    # stopping_threshold = (0.5,0.5,10.0) # for group: decreasing T, increasing T, snr

# min_dist = 3*6000 # 10 seconds * 6000 m/s
min_dist = 3.75*6000

net = "AJ"

regional_curve_file = f"/raid2/jwf39/askja/REGIONAL/regional_dispersion_v4_{vel_type}_{comp}.txt"
out_pick_file = f"/raid2/jwf39/askja/DISP_PICKS/auto_picks_snr_5_v1_{vel_type}_{comp}.json"

# regional_curve_file = f"/raid2/jwf39/askja/REGIONAL/regional_dispersion_v4_{vel_type}_{comp}.txt"
# out_pick_file = f"./dev_{vel_type}_{comp}.json"

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

use_matricies = True

threads = 15

show_picks = True

#######################################################################
#                               Main                                  #
#######################################################################

if __name__=="__main__":
    jobs = [] # list of tuples (station_pair, egf_path, distance_m)
    for row in stationpairdf.iterrows():
        if row[1][comp]:
            sta1 = row[1]["station1"]
            sta2 = row[1]["station2"]
            dist = row[1]["gcm"]
            #
            if use_matricies:
                egf_path = f"{matrix_dir}/{vel_type}/{comp}/{net}_{sta1}_{net}_{sta2}.nc"
            else:
                egf_path = f"{egf_dir}/EGF/{comp}/{net}_{sta1}_{net}_{sta2}.mseed"
            #
            if os.path.isfile(egf_path) and dist >= min_dist:
                jobs.append((f"{net}_{sta1}_{net}_{sta2}",egf_path,dist))
    #
    regional_period, regional_phasevel = np.loadtxt(regional_curve_file,unpack=True)
    # print(auto_phase_picker(jobs[0],regional_period,regional_phasevel,fSettings,stopping_threshold))
    #
    out_picks = {}
    with multiprocessing.Pool(threads) as pool:
        procs = []
        for job in jobs:
            if vel_type == "phase":
                do_group = False
                p = pool.apply_async(auto_phase_picker,args=(job,regional_period,regional_phasevel,fSettings,stopping_threshold,use_matricies))
            elif vel_type == "group":
                do_group = True
                p = pool.apply_async(auto_group_picker,args=(job,regional_period,regional_phasevel,fSettings,stopping_threshold,use_matricies))
            else:
                ValueError(f"{vel_type} is not a valid velocity type, must be 'group' or 'phase'")
            procs.append(p)
        for p in tqdm(procs):
            station_pair, pick_T, pick_c = p.get()
            if len(pick_c) > 0:
                out_picks[station_pair] = (list(pick_T),list(pick_c))
        pool.close()
        pool.terminate()
    #
    json_out = json.dumps(out_picks)
    f = open(out_pick_file,"w")
    f.write(json_out)
    f.close()
    #
    if show_picks:
        plt.figure(figsize=(12,8))
        plt.grid(True)
        for key in out_picks:
            pick_T, pick_c = out_picks[key]
            plt.plot(pick_T,pick_c,color="black",linewidth=0.7)
        plt.xlabel("Period (s)")
        plt.ylabel("Velocity (km/s)")
        plt.show()