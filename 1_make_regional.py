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

from EgfLib import regional_dispersion, pick_regional

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
else:
    vel_type = "group" # "phase" or "group"
    comp = "TT"

net = "8K"

min_dist = 4*6000

#Filter Settings
#Period Axis
minT = 0.5
maxT = 13
dT = 0.05
#Velocity Axis
dv = 0.001
minv = 1.5
maxv = 4.0
#Filter width
width_type = "dependent" # "dependent" or "fixed"
bandwidth = 0.4 # If dependent this will be 0.4*central_period and if fixed will be 0.4 s

fSettings = (minT,maxT,dT,bandwidth,width_type,dv,minv,maxv)

outfile = f"/raid1/jwf39/askja/REGIONAL/regional_dispersion_v2_{vel_type}_{comp}.nc"
regional_curve_file = f"/raid1/jwf39/askja/REGIONAL/regional_dispersion_v2_{vel_type}_{comp}.txt"
out_plot = f"/raid1/jwf39/askja/STATIONS/regionalImage/regional_dispersion_v2_{vel_type}_{comp}.png"

use_outfile = True # Use previously computed grid to pick
pick = True # Plot grid and pick after processing/loading grid

threads = 15

#######################################################################
#                               Main                                  #
#######################################################################

if __name__=="__main__":
    egf_pathlist = []
    distance_list = []
    for row in stationpairdf.iterrows():
        if row[1][comp]:
            sta1 = row[1]["station1"]
            sta2 = row[1]["station2"]
            dist = row[1]["gcm"]
            #
            egf_path = f"{egf_dir}/EGF/{comp}/{net}_{sta1}_{net}_{sta2}.mseed"
            #
            # print(egf_path)
            if os.path.isfile(egf_path) and dist >= min_dist:
                egf_pathlist.append(egf_path)
                distance_list.append(dist)
    
    # raise KeyboardInterrupt
    if use_outfile:
        print(f"Using previously computed netcdf4 grid")
        regional = xr.load_dataarray(outfile)
        c = regional.coords["velocity"].data
        T = regional.coords["period"].data
        c_T_regional = regional.data
    else:
        print(f"There are {len(egf_pathlist)} files to stack")
        c, T, c_T_regional = regional_dispersion(egf_pathlist,distance_list,fSettings,vel_type,threads,wave_num=2)

        regional = xr.DataArray(
            data=c_T_regional,
            dims=("velocity","period"),
            coords=dict(
                velocity=c,
                period=T
            )
        )
        regional.to_netcdf(outfile)
    
    if pick:
        T_disp, c_disp = pick_regional(c,T,c_T_regional,out_plot)

        with open(regional_curve_file,"w") as f:
            for T_ref, c_ref in zip(T_disp,c_disp):
                f.write(f"{T_ref} {c_ref}\n")