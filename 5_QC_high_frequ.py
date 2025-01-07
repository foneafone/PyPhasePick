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

from pyphasepick.frequencytimeanalisys import hff_worker

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

net = "AJ"

synth_vels_file = f"/raid2/jwf39/askja/notebooks/synthetic_travel_times_{comp}_1-4s.json"
synth_periods = ["1.0","2.0","3.0","4.0"]

low_freq_pick_file = f"/raid2/jwf39/askja/DISP_PICKS/auto_picks_snr_5_v1_{vel_type}_{comp}.json"
out_pick_file = f"/raid2/jwf39/askja/DISP_PICKS/high_freq_auto_picks_snr_5_v1_{vel_type}_{comp}.json"

out_qc_file = f"/raid2/jwf39/askja/DISP_PICKS/qc_dict_snr_5_v1_{vel_type}_{comp}.json"

#Filter Settings
#Frequency Axis
minf = 0.1
maxf = 4.0
df = 0.01
#Velocity Axis
dv = 0.001
minv = 1.8
maxv = 3.5
#Filter width
width_type = "dependent" # "dependent" or "fixed"
bandwidth = 0.4 # If dependent this will be 0.4*central_period and if fixed will be 0.4 s

low_freq_cutoff = 1.0 # Lowest period used in 3_automatic_pick.py (in seconds)

fSettings = (minf,maxf,df,bandwidth,width_type,dv,minv,maxv)

threads = 28

#######################################################################
#                               Main                                  #
#######################################################################

if __name__=="__main__":
    with open(out_pick_file,"r") as pickfile:
        high_freq_picks = json.load(pickfile)
    with open(low_freq_pick_file,"r") as pickfile:
        low_freq_picks = json.load(pickfile)
    with open(synth_vels_file,"r") as synthfile:
        synth_vels = json.load(synthfile)
    try: 
        with open(out_qc_file,"r") as outfile:
            qc_dict = json.load(outfile)
    except FileNotFoundError:
        qc_dict = {}
    synth_periods_float = np.array(synth_periods,dtype=float)
    synth_freqs = 1/synth_periods_float
    #
    with multiprocessing.Pool(threads) as pool:
        procs = []
        for row in stationpairdf.iterrows():
            if row[1][comp]:
                sta1 = row[1]["station1"]
                sta2 = row[1]["station2"]
                dist = row[1]["gcm"]
                if dist > 500:
                    fmin = (2*2.6*1000)/dist
                    #
                    egf_path = f"{egf_dir}/EGF/{comp}/{net}_{sta1}_{net}_{sta2}.mseed"
                    key = f"{net}_{sta1}_{net}_{sta2}"
                    #
                    if key in high_freq_picks and not key in qc_dict:
                        p = pool.apply_async(hff_worker,(key,egf_path,dist,fmin,fSettings))
                        procs.append(p)
        print(f"Left to check: {len(procs)}")
        try:
            for p in procs:
                key, dist, c_array_interp, central_frequencies, c_f_array = p.get()
                net, sta1, net2, sta2 = key.split("_")
                synth_vel = []
                dist = dist/1000
                for period in synth_periods:
                    try:
                        synth_vel.append(dist/synth_vels[f"{sta1}_{sta2}_{period}"])
                    except KeyError:
                        synth_vel.append(dist/synth_vels[f"{sta2}_{sta1}_{period}"])
                plt.figure(figsize=(10,8))
                plt.pcolormesh(central_frequencies,c_array_interp,c_f_array,cmap=plt.get_cmap("seismic"))
                # plt.scatter([1.0],[vel],color="green")
                pick_f,pick_c = high_freq_picks[key]
                plt.plot(pick_f,pick_c,color="black")
                if key in low_freq_picks:
                    low_T, low_c = low_freq_picks[key]
                    low_f = 1/np.array(low_T)
                    plt.plot(low_f,low_c,color="black")
                plt.plot(synth_freqs,synth_vel,color="green")
                plt.xlim(xmin=np.min(synth_freqs),xmax=2.5)
                ax = plt.gca()
                ax.set_facecolor('tab:gray')
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Velocity (km/s)")
                plt.show()
                cmd = input("Is good? [y]/n")
                if cmd == "n":
                    qc_dict[key] = False
                else:
                    qc_dict[key] = True
                #
                json_out = json.dumps(qc_dict)
                f = open(out_qc_file,"w")
                f.write(json_out)
                f.close()
        except KeyboardInterrupt:
            f = open(out_qc_file,"r")
            dn = len(json.load(f))
            f.close()
            raise KeyboardInterrupt(f"Exiting Script with {dn} done and {len(procs)-dn} to do")
