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

stations_csv = "/raid2/jwf39/askja/notebooks/all_stations_sep23.csv"
stationsdf = pd.read_csv(stations_csv)

stationpair_csv = "/raid2/jwf39/askja/notebooks/all_station_pairs_sep23.csv"
stationpairdf = pd.read_csv(stationpair_csv)

egf_dir = "/raid2/jwf39/askja/sep11_sep23/pws"
matrix_dir = "/raid2/jwf39/askja/sep11_sep23/ftan"

import sys
args = sys.argv

# if len(sys.argv) > 1:
#     vel_type = sys.argv[1]
#     comp = sys.argv[2]
# else:
vel_type = "group" # "phase" or "group"
comp = "ZZ"

net = "AJ"

min_dist = 2*6000
snr_threshold = 2.0

#Filter Settings
#Period Axis
minT = 2
maxT = 7.0
dT = 0.5
#Velocity Axis
dv = 0.001
minv = 1.0
maxv = 3.5
#Filter width
width_type = "dependent" # "dependent" or "fixed"
bandwidth = 0.4 # If dependent this will be 0.4*central_period and if fixed will be 0.4 s
divalpha = 5.0

fSettings = (minT,maxT,dT,bandwidth,width_type,dv,minv,maxv,divalpha)

c_std,T_std = 0.05,0.5

outfile = f"/raid2/jwf39/askja/REGIONAL/regional_dispersion_v4_{vel_type}_{comp}.nc"
regional_curve_file = f"/raid2/jwf39/askja/REGIONAL/regional_dispersion_v4_{vel_type}_{comp}.txt"
out_plot = f"/raid2/jwf39/askja/notebooks/regionalImage/regional_dispersion_v4_{vel_type}_{comp}.png"

use_matricies = False
use_outfile = False # Use previously computed grid to pick
pick = True # Plot grid and pick after processing/loading grid

threads = 15

#######################################################################
#                               Main                                  #
#######################################################################

if __name__=="__main__":
    if use_outfile:
        print(f"Using previously computed netcdf4 grid")
        regional = xr.load_dataarray(outfile)
        c = regional.coords["velocity"].data
        T = regional.coords["period"].data
        c_T_regional = regional.data
    else:
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
                if use_matricies:
                    egf_path = f"{matrix_dir}/{vel_type}/{comp}/{net}_{sta1}_{net}_{sta2}.nc"
                else:
                    egf_path = f"{egf_dir}/EGF/{comp}/{net}_{sta1}_{net}_{sta2}.mseed"
                    #
                if os.path.isfile(egf_path) and dist >= min_dist:
                    egf_pathlist.append(egf_path)
                    distance_list.append(dist)
        print(f"There are {len(egf_pathlist)} files to stack")
        c, T, c_T_regional = regional_dispersion(egf_pathlist,distance_list,fSettings,vel_type,threads,load_matrix=use_matricies,wave_num=2)
        #
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
        T_disp, c_disp = pick_regional(c,T,c_T_regional,out_plot,c_std,T_std)

        with open(regional_curve_file,"w") as f:
            for T_ref, c_ref in zip(T_disp,c_disp):
                f.write(f"{T_ref} {c_ref}\n")