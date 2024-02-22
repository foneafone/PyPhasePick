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

from EgfLib import FTAN, station_pair_stack, egf, regional_dispersion, auto_phase_picker, auto_group_picker

#######################################################################
#                             Settings                                #
#######################################################################


stations_csv = "/raid1/jwf39/askja/STATIONS/askja_stations.csv"
stationsdf = pd.read_csv(stations_csv)

stationpair_csv = "/raid1/jwf39/askja/STATIONS/station_pairs.csv"
stationpairdf = pd.read_csv(stationpair_csv)

egf_dir = "/raid1/jwf39/askja/pre_jul21/pws"

import sys
args = sys.argv

if len(sys.argv) > 1:
    vel_type = sys.argv[1]
    comp = sys.argv[2]
    stopping_threshold = (float(sys.argv[3]),float(sys.argv[4]))
else:
    vel_type = "group" # "phase" or "group"
    comp = "TT"
    stopping_threshold = (0.05,0.02)

min_dist = 4*6000 # 10 seconds * 6000 m/s

net = "8K"

regional_curve_file = f"/raid1/jwf39/askja/REGIONAL/regional_dispersion_v2_{vel_type}_{comp}.txt"
out_pick_file = f"/raid1/jwf39/askja/DISP_PICKS/auto_picks_v2_{vel_type}_{comp}_log.json"

#Filter Settings
#Period Axis
minT = 0.5
maxT = 13
dT = 0.1
#Velocity Axis
dv = 0.001
minv = 1.0
maxv = 5
#Filter width
width_type = "dependent" # "dependent" or "fixed"
bandwidth = 0.4 # If dependent this will be 0.4*central_period and if fixed will be 0.4 s

fSettings = (minT,maxT,dT,bandwidth,width_type,dv,minv,maxv)

threads = 15

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
                p = pool.apply_async(auto_phase_picker,args=(job,regional_period,regional_phasevel,fSettings,stopping_threshold,do_group))
            elif vel_type == "group":
                do_group = True
                p = pool.apply_async(auto_group_picker,args=(job,regional_period,regional_phasevel,fSettings,stopping_threshold,do_group))
            else:
                ValueError(f"{vel_type} is not a valid velocity type, must be 'group' or 'phase'")
            procs.append(p)
        for p in tqdm(procs):
            station_pair, pick_T, pick_c = p.get()
            out_picks[station_pair] = (list(pick_T),list(pick_c))
        pool.close()
        pool.terminate()
    
    json_out = json.dumps(out_picks)
    f = open(out_pick_file,"w")
    f.write(json_out)
    f.close()
