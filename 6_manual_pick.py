import numpy as np
import xarray as xr
import obspy
import obspy.geodetics.base
import time
import matplotlib.pyplot as plt
import os
import glob
import json
import multiprocessing
import pandas as pd

from pyphasepick.frequencytimeanalisys import FTAN
from pyphasepick.picking import make_c_T_lists, peak_array, find_closest_v2, conect_points_v2

#######################################################################
#                             Settings                                #
#######################################################################

stations_csv = "/raid2/jwf39/askja/notebooks/all_stations_sep23.csv"
stationsdf = pd.read_csv(stations_csv)

stationpair_csv = "/raid2/jwf39/askja/notebooks/all_station_pairs_sep23.csv"
stationpairdf = pd.read_csv(stationpair_csv)

egf_dir = "/raid2/jwf39/askja/sep11_sep23/pws"
matrix_dir = "/raid2/jwf39/askja/sep11_sep23/ftan"

disp_done = "/raid2/jwf39/askja/notebooks/disp_done.json"
disp_to_do = "/raid2/jwf39/askja/notebooks/disp_to_do.json"

vel_type = "group" # "phase" or "group"
comp = "ZZ"

net = "AJ"

regional_curve_file = f"/raid2/jwf39/askja/REGIONAL/regional_dispersion_v4_{vel_type}_{comp}.txt"

min_dist = 8*6000

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

threads = 5

use_matricies = False

showEndFigure = True

#######################################################################
#                               Main                                  #
#######################################################################


# station_pairs = np.load(disp_to_do)
# station_pairs = ["MY_SRM_YC_SBF4"]

if __name__=="__main__":
    egf_pathlist = []
    distance_list = []
    station_pairs = []
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
                egf_pathlist.append(egf_path)
                distance_list.append(dist)
                station_pairs.append(f"{net}_{sta1}_{net}_{sta2}")

regional_period, regional_phasevel = np.loadtxt(regional_curve_file,unpack=True)

try:
    f = open(disp_done,"r")
    done_pairs = json.load(f)
    f.close()
except Exception:
    done_pairs = {}

if vel_type == "group":
    do_group = True
else:
    do_group = False

try:
    #station_pairs = {"YC_SBA2_YC_SBD2":station_pairs["YC_SBA2_YC_SBD2"]}
    for station_pair,egf_path,distance in zip(station_pairs,egf_pathlist,distance_list):
        if not station_pair in done_pairs:
            rePick = True
            while rePick:
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FTAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
                if use_matricies:
                    ftan_grid = xr.load_dataarray(egf_path,engine="netcdf4")
                    T = ftan_grid.coords["period"].data
                    c = ftan_grid.coords["velocity"].data
                    c_T_array = ftan_grid.data
                else:
                    egf_trace = obspy.read(egf_path)
                    egf_trace = egf_trace[0]
                    T, tt, c, c_T_array, snr_with_period = FTAN(egf_trace,distance,fSettings,threads=threads,do_group=do_group)
                #
                # ftn = xr.DataArray(
                #     data=c_T_array,
                #     dims=("velocity","period"),
                #     coords=dict(
                #         velocity=c,
                #         period=T
                #     )
                # )
                # ftn.to_netcdf("./example_ftan.nc")
                #
                minT = 1
                maxT_disp = int(distance/6000)
                maxT_ind = np.argmin(np.abs(T-maxT_disp))
                # maxval = np.max(c_T_array[:,:maxT_ind])
                c_T_array = c_T_array[:,:maxT_ind]
                T = T[:maxT_ind]
                #
                # peak_c_T = define_peaks(c_T_array)
                c_peak, T_peak = peak_array(c,T,c_T_array)
                T_peak_list, c_peak_list = make_c_T_lists(T_peak,c_peak)
                #
                fig, ax = plt.subplots(1,1,figsize=(1080/180,1080/180),dpi=180,constrained_layout=True)
                ax.set_title(station_pair)
                if vel_type == "group":
                    cmap = "rainbow"
                else:
                    cmap = "seismic"
                ax.pcolormesh(T,c,c_T_array,cmap=plt.get_cmap(cmap),zorder=1) # type: ignore
                ax.plot(regional_period,regional_phasevel,color="black")
                ax.scatter(T_peak,c_peak,marker=".",color="black",s=0.7,picker=True,zorder=3) # type: ignore
                three_lambda = distance/9000
                ax.plot([three_lambda,three_lambda],[minv,maxv],color="green")
                ax.set_ylabel("Velocity (km/s)")
                ax.set_xlabel("Period (s)")
                if maxT_disp > 150:
                    maxT_disp = 150
                ax.set_xlim(xmin=minT-0.5,xmax=maxT_disp+0.5)
                ax.set_ylim(ymin=minv,ymax=maxv)
                # pick_simple(fig,ax)
                c_picks = []
                T_picks = []
                #
                def onclick(event):
                    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % ('double' if event.dblclick else 'single', event.button, event.x, event.y, event.xdata, event.ydata))
                    # return event.xdata, event.ydata
                    c_picks.append(event.ydata)
                    T_picks.append(event.xdata)
                #
                cid = fig.canvas.mpl_connect('button_press_event', onclick)
                plt.show()
                plt.close()
                #
                if T_picks[0] < T_picks[1]:
                    pick1 = (T_picks[0],c_picks[0])
                    pick2 = (T_picks[1],c_picks[1])
                else:
                    pick1 = (T_picks[1],c_picks[1])
                    pick2 = (T_picks[0],c_picks[0])
                pick1_Tind, pick1_cind = find_closest_v2(T_peak_list,c_peak_list,pick1[0],pick1[1])
                pick2_Tind, pick2_cind = find_closest_v2(T_peak_list,c_peak_list,pick2[0],pick2[1])
                print("Pick 1: Index (%s,%s) at %s, %s" % (pick1_Tind,pick1_cind,T_peak_list[pick1_Tind],c_peak_list[pick1_Tind][pick1_cind]))
                print("Pick 2: Index (%s,%s) at %s, %s" % (pick2_Tind,pick2_cind,T_peak_list[pick2_Tind],c_peak_list[pick2_Tind][pick2_cind]))
                pick1 = (pick1_Tind, pick1_cind)
                pick2 = (pick2_Tind, pick2_cind)
                phase_disp_T, phase_disp_c = conect_points_v2(T_peak_list,c_peak_list,pick1,pick2)
                #
                if showEndFigure:
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
                    fig, ax = plt.subplots(1,1,figsize=(1080/180,1080/180),dpi=180,constrained_layout=True)
                    ax.set_title(station_pair)
                    if vel_type == "group":
                        cmap = "rainbow"
                    else:
                        cmap = "seismic"
                    #
                    ax.pcolormesh(T,c,c_T_array,cmap=plt.get_cmap(cmap),zorder=1) # type: ignore
                    # ax.scatter(disp_T,disp_c,marker=".",color="green")
                    ax.plot(phase_disp_T,phase_disp_c,color="green",zorder=3)
                    for i in range(len(T_peak_list)):
                        Temp = T_peak_list[i]*np.ones((len(c_peak_list[i])))
                        ax.scatter(Temp,c_peak_list[i],marker=".",color="black",s=0.7,zorder=4) # type: ignore
                    # ax.scatter(T_peak,c_peak,marker=".",color="black",s=0.7)
                    three_lambda = distance/9000
                    # ax.plot([three_lambda,three_lambda],[2,5],color="yellow",zorder=2)
                    ax.set_ylabel("Velocity (km/s)")
                    ax.set_xlabel("Period (s)")
                    if maxT_disp > 150:
                        maxT_disp = 150
                    ax.set_xlim(xmin=minT-0.5,xmax=maxT_disp+0.5)
                    # ax.set_ylim(ymin=2,ymax=5)
                    plt.show()
                    plt.close()
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
                cmd = input("If done press enter. To re-pick type \'r\'. To remove path type \'d\'. To save pick but mark as difficult type \'s\'.")
                if cmd == "r":
                    rePick = True
                elif cmd == "d":
                    done_pairs[station_pair] = "Station pair removed during picking."
                    rePick = False
                elif cmd == "s":
                    done_pairs[station_pair] = (list(phase_disp_T),list(phase_disp_c),True)
                    rePick = False
                else:
                    done_pairs[station_pair] = (list(phase_disp_T),list(phase_disp_c),False)
                    rePick = False
            json_out = json.dumps(done_pairs)
            f = open(disp_done,"w")
            f.write(json_out)
            f.close()
            #
except KeyboardInterrupt:
    f = open(disp_done,"r")
    dn = len(json.load(f))
    f.close()
    td = len(station_pairs)
    raise KeyboardInterrupt("Exiting Script with %s of %s picked" % (dn,td))
