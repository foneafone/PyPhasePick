import numpy as np
import xarray as xr
import obspy
import obspy.core.trace
from obspy import UTCDateTime
import time
import matplotlib.pyplot as plt
import matplotlib.pyplot
import os
import glob
import json
from scipy.signal import hilbert, find_peaks
from scipy.special import expit
import multiprocessing
import pandas as pd
from numpy import cos,sin
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

from frequencytimeanalisys import calc_snr, FTAN, high_freq_ftan
    
#######################################################################################################
#                                            Reional FTAN                                             #
#######################################################################################################

def define_peaks(c_T):
    c_len, T_len = c_T.shape
    c_T_out = np.zeros(c_T.shape)
    for i in range(T_len):
        inds, prop = find_peaks(c_T[:,i])
        c_T_out[inds,i] = 1
    return c_T_out

def regional_dispersion_worker(egf_path,distance,fSettings,vel_type,wave_num,load_matrix):
    if not load_matrix:
        egf_trace = obspy.read(egf_path)
        egf_trace = egf_trace[0]
    if vel_type == "group":
        do_group = True
    else:
        do_group = False
    #
    minT = fSettings[0]
    maxT = fSettings[1]
    dT = fSettings[2]
    dv = fSettings[5]
    minv = fSettings[6]
    maxv = fSettings[7]
    c_T_array_shape = (len(np.arange(minv,maxv+dv,dv)),len(np.arange(minT,maxT,dT)))
    c_T_out = np.zeros((c_T_array_shape[0],c_T_array_shape[1],2))
    #
    if load_matrix:
        ftan_grid = xr.load_dataarray(egf_path,engine="netcdf4")
        T = ftan_grid.coords["period"].data
        c = ftan_grid.coords["velocity"].data
        c_T_array = ftan_grid.data
    else:
        snr = calc_snr(egf_trace,distance)
        maxtime = distance/2000
        lenTrace = len(egf_trace.data)/egf_trace.stats["sampling_rate"]
        T, tt, c, c_T_array, snr_with_period = FTAN(egf_trace,distance,fSettings,do_group=do_group)
    c_T_array = define_peaks(c_T_array)
    #
    for i in range(len(c)):
        for j in range(len(T)):
            if c[i]*1000*T[j] <= distance/wave_num:
                c_T_out[i,j,0] = float(c_T_array[i,j])
                c_T_out[i,j,1] = 1
            else:
                c_T_out[i,j,0] = 0
                c_T_out[i,j,1] = 0
    return c_T_out

def regional_dispersion(egf_pathlist,distance_list,fSettings,vel_type,threads,load_matrix=False,wave_num=2):
    """
    
    """
    minT = fSettings[0]
    maxT = fSettings[1]
    dT = fSettings[2]
    dv = fSettings[5]
    minv = fSettings[6]
    maxv = fSettings[7]
    T = np.arange(minT,maxT,dT)
    c = np.arange(minv,maxv+dv,dv)
    c_T_shape = (len(c),len(T))
    c_T_regional = np.zeros(c_T_shape)
    ones_sum = np.zeros(c_T_shape)
    #
    #if __name__=="__main__":
    with multiprocessing.Pool(threads) as pool:
        procs = []
        for i in range(len(egf_pathlist)):
            egf_path = egf_pathlist[i]
            distance = distance_list[i]
            p = pool.apply_async(regional_dispersion_worker,args=(egf_path,distance,fSettings,vel_type,wave_num,load_matrix))
            procs.append(p)
        print("All FTAN processes running...")
        count=1
        total = len(procs)
        for p in tqdm(procs):
            c_T_array = p.get()
            c_T_regional = c_T_regional+c_T_array[:,:,0]
            ones_sum = ones_sum+c_T_array[:,:,1]
            # print(f"Done {count} of {total} FTAN   ",end="\r")
            count+=1
        print("Done all FTAN, joining threads           ")
        pool.close()
        pool.terminate()
    for i in range(c_T_shape[0]):
        for j in range(c_T_shape[1]):
            if ones_sum[i,j] != 0:
                c_T_regional[i,j] = c_T_regional[i,j]/ones_sum[i,j]
    return c, T, c_T_regional

def c_peaks(c,c_T):
    c_len, T_len = c_T.shape
    out_c = [[] for i in range(T_len)]
    for i in range(T_len):
        inds, prop = find_peaks(c_T[:,i])
        for j in inds:
            out_c[i].append(c[j])
    return out_c
def c_peaks_scatter(T,c_lists):
    out_T = []; out_c = []
    for i in range(len(T)):
        for c in c_lists[i]:
            out_T.append(T[i])
            out_c.append(c)
    return np.array(out_T), np.array(out_c)

def find_closest(c_list,c_ref):
    return np.argmin([np.abs(c-c_ref) for c in c_list])

def add_to_peaks(T_list,c_list,T,c_peak_list):
    inds = np.argsort(T_list)
    T_list = np.array(T_list)[inds]
    c_list = np.array(c_list)[inds]
    T_interp = [i for i in T if i < max(T_list)]
    c_list = np.interp(T_interp,T_list,c_list)
    c_list = np.round(c_list,decimals=2)
    for i in range(len(T_interp)):
        c_peak_list[i].append(c_list[i])
    return c_peak_list

def gen_curve_from_regional_period(T,c_lists,c_regional,T_regional,maxT,grad_threshold):
    T_ind_start = find_closest(T,T_regional)
    max_T_ind = find_closest(T,maxT)
    c_ind_start = find_closest(c_lists[T_ind_start],c_regional)
    # Look increasing periods
    T_ind = int(T_ind_start)
    c = c_lists[T_ind][c_ind_start]
    T_disp = [T[T_ind]]; c_disp = [c]
    notEnd = True
    while notEnd:
        T_ind += 1
        if T_ind > max_T_ind:
            notEnd = False
        else:
            try:
                new_c = c_lists[T_ind][find_closest(c_lists[T_ind],c)]
                diff = abs(new_c-c)
                if diff > grad_threshold:
                    notEnd = False
                else:
                    c = float(new_c)
                    T_disp.append(T[T_ind])
                    c_disp.append(c)
            except:
                notEnd = False
    # Look decreasing periods
    T_ind = int(T_ind_start)
    c = c_lists[T_ind][c_ind_start]
    notEnd = True
    while notEnd:
        T_ind -= 1
        if T_ind < 0:
            notEnd = False
        else:
            try:
                new_c = c_lists[T_ind][find_closest(c_lists[T_ind],c)]
                diff = abs(new_c-c)
                if diff > grad_threshold:
                    notEnd = False
                else:
                    c = float(new_c)
                    T_disp = [T[T_ind]] + T_disp
                    c_disp = [c] + c_disp
            except:
                notEnd = False
    return np.array(T_disp), np.array(c_disp)

def pick_regional(c,T,c_T,plotfile,c_std,T_std):
    """Input: Unsmoothed output from regional dispersion file"""
    from scipy.ndimage import gaussian_filter
    # for i in range(len(T)):
    #     c_T[:,i] = (c_T[:,i]-min(c_T[:,i]))/(max(c_T[:,i]-min(c_T[:,i])))
    c_std = c_std/(c[1]-c[0])
    c_T = gaussian_filter(c_T,c_std,order=0,axes=0)
    T_std = T_std/(T[1]-T[0])
    c_T = gaussian_filter(c_T,T_std,order=0,axes=1)
    c_peak_list = c_peaks(c,c_T)
    T_scatter, c_scatter = c_peaks_scatter(T,c_peak_list)
    fig = plt.figure(figsize=(1080/180,1080/180),dpi=180,constrained_layout=True)
    plt.pcolormesh(T,c,c_T,cmap=plt.get_cmap("rainbow"),zorder=1) # type: ignore
    plt.scatter(T_scatter,c_scatter,marker=".",color="black",s=0.7,picker=True) # type: ignore
    plt.xlabel("Period (s)")
    plt.ylabel("Velocity (km/s)")
    c_click = []
    T_click = []
    #
    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % ('double' if event.dblclick else 'single', event.button, event.x, event.y, event.xdata, event.ydata))
        # return event.xdata, event.ydata
        c_click.append(event.ydata)
        T_click.append(event.xdata)
    #
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    plt.close()
    print(c_click)
    print(T_click)
    # c_peak_list = add_to_peaks(T_click[:-1],c_click[:-1],T,c_peak_list)
    # T_scatter, c_scatter = c_peaks_scatter(T,c_peak_list)
    grad_thresh = 0.4*abs(T[1]-T[0])
    T_disp,c_disp = gen_curve_from_regional_period(T,c_peak_list,c_click[-1],T_click[-1],60,grad_thresh)
    minT = min((T_click[-1],T_click[-2]))
    maxT = max((T_click[-1],T_click[-2]))
    condition = [minT <= Ti <= maxT for Ti in T_disp]
    inds = np.where(condition)
    T_disp, c_disp = T_disp[inds], c_disp[inds]
    fig = plt.figure(figsize=(1080/180,1080/180),dpi=180,constrained_layout=True)
    plt.pcolormesh(T,c,c_T,cmap=plt.get_cmap("rainbow"),zorder=1) # type: ignore
    plt.scatter(T_scatter,c_scatter,marker=".",color="black",s=0.7,picker=True) # type: ignore
    plt.plot(T_disp,c_disp,color="black")
    plt.xlabel("Period (s)")
    plt.ylabel("Velocity (km/s)")
    plt.show()
    fig.savefig(plotfile)
    return T_disp, c_disp

#######################################################################################################
#                                           Picking Funcs                                             #
#######################################################################################################

def peak_array(c,T,c_T):
    c_len, T_len = c_T.shape
    out_c = []
    out_T = []
    for i in range(T_len):
        inds, prop = find_peaks(c_T[:,i])
        for j in inds:
            out_c.append(c[j])
            out_T.append(T[i])
    return np.array(out_c), np.array(out_T)

def make_c_T_lists(T_peaks,c_peaks):
    out_T = []
    out_c = []
    for i in range(len(T_peaks)):
        if out_T == [] or out_T[-1] != T_peaks[i]:
            out_T.append(T_peaks[i])
            out_c.append([])
            out_c[-1].append(c_peaks[i])
        else:
            out_c[-1].append(c_peaks[i])
    return out_T, out_c

def find_closest_v2(T_list,c_list,T_point,c_point):
    min_dist = 5000000
    out_inds = (0,0)
    for i in range(len(T_list)):
        cl = c_list[i]
        for j in range(len(cl)):
            dist = np.sqrt((T_list[i]-T_point)**2 + (cl[j]-c_point)**2)
            if dist < min_dist:
                min_dist = float(dist)
                out_inds = (i,j)
    return out_inds

def conect_points_v2(T_list,c_list,pick1,pick2):
    Tl = T_list[pick1[0]:pick2[0]+1]
    cl = c_list[pick1[0]:pick2[0]+1]
    c_out = []
    for i in range(len(Tl)):
        cl_i = cl[i]
        if c_out == []:
            c_out.append(cl_i[pick1[1]])
        else:
            min_dist = 5000000
            next_c = 0
            for j in range(len(cl_i)):
                dist = abs(cl_i[j]-c_out[-1])
                if dist < min_dist:
                    min_dist = float(dist)
                    next_c = float(cl_i[j])
            c_out.append(next_c)
    return np.array(Tl), np.array(c_out)

def auto_phase_picker(job,regional_period,regional_phasevel,fSettings,stopping_threshold,use_matricies):
    station_pair, egf_path, distance = job
    maxT = distance/6000
    #
    starting_threshold, decreasing_threshold, increasing_threshold, snr_threshold = stopping_threshold
    #
    if use_matricies:
        ftan_grid = xr.load_dataarray(egf_path,engine="netcdf4")
        T = ftan_grid.coords["period"].data
        c = ftan_grid.coords["velocity"].data
        c_T_array = ftan_grid.data
        snr_path = egf_path.split(".")[0] + "_snr.npy"
        snr_with_period = np.load(snr_path)
    else:
        egf_trace = obspy.read(egf_path)
        egf_trace = egf_trace[0]
        T, tt, c, c_T_array, snr_with_period = FTAN(egf_trace,distance,fSettings,threads=1,do_group=False)
    #
    c_peak_list = c_peaks(c,c_T_array)
    #
    pick_c = []
    pick_T = []
    pick_snr = []
    for i in range(len(T)): # Itterate backwards through the periods
        i = -i -1
        period = T[i]
        snr = snr_with_period[i]
        # print(snr)
        if period < maxT: # When period is less that the maximum period determined by lamda/2 start picking
            if pick_c == []: # pick first based on reference curve
                regional_c = np.interp(period,regional_period,regional_phasevel)
                previous_c = c_peak_list[i][find_closest(c_peak_list[i],regional_c)]
                diff = np.abs(regional_c-previous_c)
                if diff > starting_threshold:
                    return station_pair, pick_T,pick_c
                if snr > snr_threshold:
                    pick_c.append(float(previous_c))
                    pick_T.append(float(period))
            else: # Pick next based on previous velocity
                new_c = c_peak_list[i][find_closest(c_peak_list[i],previous_c)]
                if -decreasing_threshold < new_c-previous_c < increasing_threshold: # if jump is within bounds save and pick next
                    if snr > snr_threshold:
                        pick_c.append(float(new_c))
                        pick_T.append(float(period))
                    previous_c = float(new_c)
                else: # if jump is too high return without short periods
                    pick_c = np.array(pick_c)
                    pick_T = np.array(pick_T)
                    inds = np.argsort(pick_T)
                    pick_c = pick_c[inds]
                    pick_T = pick_T[inds]
                    return station_pair, pick_T, pick_c
    #
    pick_c = np.array(pick_c)
    pick_T = np.array(pick_T)
    inds = np.argsort(pick_T)
    pick_c = pick_c[inds]
    pick_T = pick_T[inds]
    return station_pair, pick_T, pick_c

def auto_group_picker_v1(job,regional_period,regional_groupvel,fSettings,stopping_threshold,use_matricies):
    station_pair, egf_path, distance = job
    maxT = distance/6000
    #
    decreasing_threshold, increasing_threshold, snr_threshold = stopping_threshold
    #
    if use_matricies:
        ftan_grid = xr.load_dataarray(egf_path,engine="netcdf4")
        T = ftan_grid.coords["period"].data
        c = ftan_grid.coords["velocity"].data
        c_T_array = ftan_grid.data
        snr_file = egf_path[:-3] + "_snr.npy"
        snr_with_period = np.load(snr_file)
    else:
        egf_trace = obspy.read(egf_path)
        egf_trace = egf_trace[0]
        T, tt, c, c_T_array, snr_with_period = FTAN(egf_trace,distance,fSettings,threads=1,do_group=True)
    #
    # c_peak_list = c_peaks(c,c_T_array)
    #
    pick_c = []
    pick_T = []
    pick_snr = []
    try:
        for i in range(len(T)): # Itterate backwards through the periods
            i = -i -1
            period = T[i]
            snr = snr_with_period[i]
            if period < maxT: # When period is less that the maximum period determined by lamda/2 start picking
                if pick_c == []: # pick first based on maximum amplitude at maxT
                    previous_c = c[int(np.argmax(c_T_array[:,i]))]
                    pick_c.append(float(previous_c))
                    pick_T.append(float(period))
                    pick_snr.append(float(snr))
                else: # Pick next maximum
                    # new_c = c_peak_list[i][find_closest(c_peak_list[i],previous_c)]
                    new_c = c[int(np.argmax(c_T_array[:,i]))]
                    if -decreasing_threshold < new_c-previous_c < increasing_threshold: # if jump is within bounds save and pick next
                        pick_c.append(float(new_c))
                        pick_T.append(float(period))
                        pick_snr.append(float(snr))
                        previous_c = float(new_c)
                    else: # if jump is too high return without short periods
                        pick_c = np.array(pick_c)
                        pick_T = np.array(pick_T)
                        pick_snr = np.array(pick_snr)
                        inds = np.argsort(pick_T)
                        pick_c = pick_c[inds]
                        pick_T = pick_T[inds]
                        pick_snr = pick_snr[inds]
                        pick_snr = np.where(pick_snr > snr_threshold)
                        pick_c = pick_c[pick_snr]
                        pick_T = pick_T[pick_snr]
                        return station_pair, pick_T, pick_c
    except Exception as e:
        print(e)
        return station_pair, np.array([]), np.array([])
    #
    pick_c = np.array(pick_c)
    pick_T = np.array(pick_T)
    pick_snr = np.array(pick_snr)
    inds = np.argsort(pick_T)
    pick_c = pick_c[inds]
    pick_T = pick_T[inds]
    pick_snr = pick_snr[inds]
    pick_snr = np.where(pick_snr > snr_threshold)
    pick_c = pick_c[pick_snr]
    pick_T = pick_T[pick_snr]
    return station_pair, pick_T, pick_c

    
def high_freq_auto_phase_picker(job,fSettings,stopping_threshold):
    """
    
    """
    from scipy.ndimage import gaussian_filter
    # Load settings
    station_pair,egf_path,distance,ref_vel,ref_freq = job
    starting_threshold, decreasing_threshold, increasing_threshold, snr_threshold = stopping_threshold
    #
    # Load trace for EGF
    egf_trace = obspy.read(egf_path)
    egf_trace = egf_trace[0]
    c, central_frequencies, c_f_array, snr_with_frequency = high_freq_ftan(egf_trace,distance,ref_freq,fSettings)
    #
    c_peak_list = c_peaks(c,c_f_array)
    #
    pick_c = []
    pick_f = []
    #
    try:
        for i in range(len(central_frequencies)): # Itterate up through frequencies
            f = central_frequencies[i]
            snr = snr_with_frequency[i]
            # print(snr)
            if pick_c == []:
                previous_c = c_peak_list[i][find_closest(c_peak_list[i],ref_vel)]
                diff = np.abs(ref_vel-previous_c)
                if diff > starting_threshold:
                    return station_pair, pick_f,pick_c
                if snr > snr_threshold:
                    pick_c.append(float(previous_c))
                    pick_f.append(float(f))
            else: # Pick next based on previous velocity
                new_c = c_peak_list[i][find_closest(c_peak_list[i],previous_c)]
                if -decreasing_threshold < new_c-previous_c < increasing_threshold: # if jump is within bounds save and pick next
                    if snr > snr_threshold:
                        pick_c.append(float(new_c))
                        pick_f.append(float(f))
                    previous_c = float(new_c)
                else: # if jump is too high return without short periods
                    pick_c = np.array(pick_c)
                    pick_f = np.array(pick_f)
                    return station_pair, pick_f, pick_c
    except Exception as e:
        print(e)
        return station_pair, np.array([]), np.array([])
    #
    pick_c = np.array(pick_c)
    pick_c = gaussian_filter(pick_c,4)
    pick_f = np.array(pick_f)
    return station_pair, pick_f, pick_c