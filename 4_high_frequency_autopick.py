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

from EgfLib import high_freq_auto_phase_picker

#######################################################################
#                             Settings                                #
#######################################################################


stations_csv = "/raid2/jwf39/askja/notebooks/all_stations_sep23.csv"
stationsdf = pd.read_csv(stations_csv)

stationpair_csv = "/raid2/jwf39/askja/notebooks/all_station_pairs_sep23.csv"
stationpairdf = pd.read_csv(stationpair_csv)

egf_dir = "/raid2/jwf39/askja/sep11_sep23/pws"
matrix_dir = "/raid2/jwf39/askja/sep11_sep23/ftan"



vel_type = "phase" # "phase" or "group"
comp = "ZZ"
stopping_threshold = (0.3,0.05,0.05, 5) # for phase: starting, decreasing , increasing 

net = "AJ"

synth_vels_file = f"/raid2/jwf39/askja/notebooks/synthetic_travel_times_{comp}_1-4s.json"
synth_periods = ["1.0","2.0","3.0","4.0"]

low_freq_pick_file = f"dev_{vel_type}_{comp}.json"
out_pick_file = f"/raid2/jwf39/askja/DISP_PICKS/high_freq_auto_picks_snr_5_v1_{vel_type}_{comp}.json"
# out_pick_file = f"dev_high_freq_{vel_type}_{comp}.json"

#Filter Settings
#Frequency Axis
minf = 0.1
maxf = 4.0
df = 0.01
#Velocity Axis
dv = 0.001
minv = 1.0
maxv = 3.5
#Filter width
width_type = "dependent" # "dependent" or "fixed"
bandwidth = 0.4 # If dependent this will be 0.4*central_period and if fixed will be 0.4 s

low_freq_cutoff = 1.0 # Lowest period used in 3_automatic_pick.py (in seconds)

fSettings = (minf,maxf,df,bandwidth,width_type,dv,minv,maxv)

threads = 28
mindist = 500 # minimum distance to stop divzero errors

show_picks = True

#######################################################################
#                               Main                                  #
#######################################################################

if __name__=="__main__":
    with open(low_freq_pick_file,"r") as pickfile:
        low_freq_picks = json.load(pickfile)
    with open(synth_vels_file,"r") as synthfile:
        synth_vels = json.load(synthfile)
    synth_periods_float = np.array(synth_periods,dtype=float)
    synth_freqs = 1/synth_periods_float
    jobs = [] # list of tuples (station_pair, egf_path, distance_m, ref_vel, ref_freq)
    for row in stationpairdf.iterrows():
        if row[1][comp]:
            sta1 = row[1]["station1"]
            sta2 = row[1]["station2"]
            dist = row[1]["gcm"]
            #
            egf_path = f"{egf_dir}/EGF/{comp}/{net}_{sta1}_{net}_{sta2}.mseed"
            #
            key = f"{net}_{sta1}_{net}_{sta2}"
            #
            if os.path.isfile(egf_path) and dist > mindist:
                if key in low_freq_picks:
                    T_pick, c_pick = low_freq_picks[key]
                    min_t = np.min(T_pick)
                    if min_t <= low_freq_cutoff:
                        ref_freq = 1/min_t
                        ref_vel = c_pick[0]
                        jobs.append((f"{net}_{sta1}_{net}_{sta2}",egf_path,dist,ref_vel,ref_freq))
                        print((f"{net}_{sta1}_{net}_{sta2}",egf_path,dist,ref_vel,ref_freq))
                else:
                    vels = []
                    for period in synth_periods:
                        try:
                            tt = synth_vels[f"{sta1}_{sta2}_{period}"]
                        except KeyError:
                            tt = synth_vels[f"{sta2}_{sta1}_{period}"]
                        vel = (dist/1000)/tt
                        vels.append(vel)
                    vels = np.array(vels)
                    fmin = (2*np.min(vels)*1000)/dist
                    if fmin < 1/low_freq_cutoff:
                        frequs = np.arange(minf,maxf+df,df)
                        central_frequencies = np.array([f for f in frequs if f > fmin])
                        ref_freq = np.min(central_frequencies)
                        ref_vel = np.interp(ref_freq,synth_freqs,vels)
                        jobs.append((f"{net}_{sta1}_{net}_{sta2}",egf_path,dist,ref_vel,ref_freq))
    print(f"Starting {len(jobs)} picking jobs")
    #
    out_picks = {}
    with multiprocessing.Pool(threads) as pool:
        procs = []
        for job in jobs:
            p = pool.apply_async(high_freq_auto_phase_picker,args=(job,fSettings,stopping_threshold))
            procs.append(p)
        for p in tqdm(procs):
            station_pair, pick_f, pick_c = p.get()
            if len(pick_c) > 0:
                out_picks[station_pair] = (list(pick_f),list(pick_c))
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
            pick_f, pick_c = out_picks[key]
            plt.plot(pick_f,pick_c,color="black",linewidth=0.7)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Velocity (km/s)")
        plt.show()