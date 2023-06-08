import numpy as np
import obspy
import time
import matplotlib.pyplot as plt
import os
import glob
import json
import multiprocessing

from EgfLib import FTAN, make_c_T_lists, peak_array, find_closest_v2, conect_points_v2

#######################################################################
#                             Settings                                #
#######################################################################

egf_dir = ""

station_file = ""

regional_curve_file = "./regional_dispersion.txt"

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

disp_to_do = "./disp_to_do.npy"
disp_done = "./picked_dispersion.json"

threads = 4

showEndFigure = True

#######################################################################
#                               Main                                  #
#######################################################################

station_pairs = np.load(disp_to_do)

station_loc = {}
stations = np.loadtxt(station_file,usecols=(0),dtype=str,unpack=True)
lats, lons = np.loadtxt(station_file,usecols=(1,2),unpack=True)
for station,lat,lon in zip(stations,lats,lons):
    station_loc[station] = (lat,lon)

egf_pathlist = []
distance_list = []
for station_pair in station_pairs:
    station1, station2 = station_pair.split("_")
    lat1, lon1 = station_loc[station1]
    lat2, lon2 = station_loc[station2]
    dist, azab, azbc = obspy.geodetics.base.gps2dist_azimuth(lat1,lon1,lat2,lon2)
    #
    egf_path = f"{egf_dir}/{station_pair}.mseed"
    #
    egf_pathlist.append(egf_path)
    distance_list.append(dist)

try:
    f = open(disp_done,"r")
    done_pairs = json.load(f)
    f.close()
except Exception:
    done_pairs = {}

try:
    #station_pairs = {"YC_SBA2_YC_SBD2":station_pairs["YC_SBA2_YC_SBD2"]}
    for station_pair,egf_path,distance in zip(station_pairs,egf_pathlist,distance_list):
        if not station_pair in done_pairs:
            rePick = True
            while rePick:
                vel_type = "phase"
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FTAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
                egf_trace = obspy.read(egf_path)
                T, tt, c, c_T_array = FTAN(egf_trace,distance,fSettings,threads=threads,do_group=False)
                #
                minT = 1
                maxT_disp = int(distance/6000)
                #
                # peak_c_T = define_peaks(c_T_array)
                c_peak, T_peak = peak_array(c,T,c_T_array)
                T_peak_list, c_peak_list = make_c_T_lists(T_peak,c_peak)
                mintime = distance/5500
                maxtime = distance/1500
                #
                fig, ax = plt.subplots(1,1,figsize=(1080/180,1080/180),dpi=180,constrained_layout=True)
                ax.set_title(station_pair)
                if vel_type == "group":
                    cmap = "rainbow"
                else:
                    cmap = "seismic"
                ax.pcolormesh(T,c,c_T_array,cmap=plt.get_cmap(cmap),zorder=1)
                ax.scatter(T_peak,c_peak,marker=".",color="black",s=0.7,picker=True,zorder=3)
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
                    ax.pcolormesh(T,c,c_T_array,cmap=plt.get_cmap(cmap),zorder=1)
                    # ax.scatter(disp_T,disp_c,marker=".",color="green")
                    ax.plot(phase_disp_T,phase_disp_c,color="green",zorder=3)
                    for i in range(len(T_peak_list)):
                        Temp = T_peak_list[i]*np.ones((len(c_peak_list[i])))
                        ax.scatter(Temp,c_peak_list[i],marker=".",color="black",s=0.7,zorder=4)
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
    f = open(disp_to_do,"r")
    td = len(json.load(f))
    f.close()
    f = open(disp_done,"r")
    dn = len(json.load(f))
    f.close()
    raise KeyboardInterrupt("Exiting Script with %s of %s picked" % (dn,td))
