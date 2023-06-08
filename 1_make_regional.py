import numpy as np
import obspy
import time
import matplotlib.pyplot as plt
import os
import glob
import json
import multiprocessing

from EgfLib import regional_dispersion, pick_regional

#######################################################################
#                             Settings                                #
#######################################################################

station_pair_list = []

egf_dir = ""

station_file = ""

vel_type = "phase" # "phase" or "goup"

#Filter Settings
#Period Axis
minT = 1
maxT = 60
dT = 0.25
#Velocity Axis
dv = 0.01
minv = 1.5
maxv = 5
#Filter width
width_type = "dependent" # "dependent" or "fixed"
bandwidth = 0.4 # If dependent this will be 0.4*central_period and if fixed will be 0.4 s

fSettings = (minT,maxT,dT,bandwidth,width_type,dv,minv,maxv)

outfile = "./regional_dispersion.npy"
regional_curve_file = "./regional_dispersion.txt"
out_plot = "./regional_dispersion.png"

threads = 20

#######################################################################
#                               Main                                  #
#######################################################################

station_loc = {}
stations = np.loadtxt(station_file,usecols=(0),dtype=str,unpack=True)
lats, lons = np.loadtxt(station_file,usecols=(1,2),unpack=True)
for station,lat,lon in zip(stations,lats,lons):
    station_loc[station] = (lat,lon)

egf_pathlist = []
distance_list = []
for station_pair in station_pair_list:
    station1, station2 = station_pair.split("_")
    lat1, lon1 = station_loc[station1]
    lat2, lon2 = station_loc[station2]
    dist, azab, azbc = obspy.geodetics.base.gps2dist_azimuth(lat1,lon1,lat2,lon2)
    #
    egf_path = f"{egf_dir}/{station_pair}.mseed"
    #
    egf_pathlist.append(egf_path)
    distance_list.append(dist)

c, T, c_T_regional = regional_dispersion(egf_pathlist,distance_list,fSettings,vel_type,threads,wave_num=2)

T_disp, c_disp = pick_regional(c,T,c_T_regional,out_plot)

with open(regional_curve_file,"w") as f:
    for T_ref, c_ref in zip(T_disp,c_disp):
        f.write(f"{T_ref} {c_ref}\n")