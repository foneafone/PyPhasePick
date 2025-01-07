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

#######################################################################################################
#                                          Stacking Functions                                         #
#######################################################################################################

def make_stack_jobs(stations_csv,cc_path_structure,start_date,end_date,comps,outdir,remake,ignore_net=False,fake_net="AJ"):
    stations_df = pd.read_csv(stations_csv)
    job_list = []
    jobtypes = []
    if "TT" in comps or "RR" in comps:
        jobtypes.append("horezontal")
    if "ZZ" in comps:
        jobtypes.append("vertical")
    for i,row1 in stations_df.iterrows():
        for j,row2 in stations_df.iterrows():
            net1, sta1, lat1, lon1 = row1["network"], row1["station"], row1["lat"], row1["lon"]
            net2, sta2, lat2, lon2 = row2["network"], row2["station"], row2["lat"], row2["lon"]
            if ignore_net:
                path = cc_path_structure.replace("NET1","*").replace("NET2","*")
            else:
                path = cc_path_structure.replace("NET1",net1).replace("NET2",net2)
            path = path.replace("STA1",sta1).replace("STA2",sta2)
            path = path.replace("YEAR","*").replace("MM","*").replace("DD","*")
            #
            if ignore_net:
                outpath = f"{outdir}/EGF/COMP/{fake_net}_{sta1}_{fake_net}_{sta2}.mseed"
            else:
                outpath = f"{outdir}/EGF/COMP/{net1}_{sta1}_{net2}_{sta2}.mseed"
            outZZ = outpath.replace("COMP","ZZ")
            outTT = outpath.replace("COMP","TT")
            outRR = outpath.replace("COMP","RR")
            #
            for jobtype in jobtypes:
                if jobtype == "vertical" and (remake or not os.path.exists(outZZ)):
                    paths = glob.glob(path.replace("COMP","ZZ"))
                    if paths != []:
                        paths = [pt for pt in paths if UTCDateTime(start_date) < UTCDateTime(pt.split("/")[-1].split(".")[0]) < UTCDateTime(end_date)]
                        job = (jobtype,net1,sta1,net2,sta2,paths)
                        job_list.append(job)
                if jobtype == "horezontal" and (remake or not os.path.exists(outTT)):
                    gcm, az, baz = obspy.geodetics.base.gps2dist_azimuth(lat1,lon1,lat2,lon2)
                    pathsEE = glob.glob(path.replace("COMP","EE"))
                    if pathsEE != []:
                        pathsEE = [pt for pt in pathsEE if UTCDateTime(start_date) < UTCDateTime(pt.split("/")[-1].split(".")[0]) < UTCDateTime(end_date)]
                        pathsEN = glob.glob(path.replace("COMP","EN"))
                        pathsEN = [pt for pt in pathsEN if UTCDateTime(start_date) < UTCDateTime(pt.split("/")[-1].split(".")[0]) < UTCDateTime(end_date)]
                        pathsNN = glob.glob(path.replace("COMP","NN"))
                        pathsNN = [pt for pt in pathsNN if UTCDateTime(start_date) < UTCDateTime(pt.split("/")[-1].split(".")[0]) < UTCDateTime(end_date)]
                        pathsNE = glob.glob(path.replace("COMP","NE"))
                        pathsNE = [pt for pt in pathsNE if UTCDateTime(start_date) < UTCDateTime(pt.split("/")[-1].split(".")[0]) < UTCDateTime(end_date)]
                        job = (jobtype,net1,sta1,net2,sta2,az,baz,comps,pathsEE,pathsEN,pathsNN,pathsNE)
                        job_list.append(job)
    return job_list

def gen_delays(cc_stats):
    dt = 1/cc_stats["sampling_rate"]
    delay = int((cc_stats["npts"]-1)/2)*dt
    delays = np.arange(-delay,delay+dt,dt)
    return delays

def stream2array(stream):
    """
    Converts a stream of traces to a numpy 2darray.
    Has shape (len(stream),trace_npts)
    """
    npts = stream[0].stats["npts"]
    out_array = np.zeros((len(stream),npts))
    for i in range(len(stream)):
        out_array[i,:] = stream[i].data
    return out_array

def linear_stack(stream_array):
    out_stack = stream_array.sum(axis=0)/stream_array.shape[0]
    return out_stack

def gen_phase_array(stream_array):
    return np.exp(1j*np.angle(hilbert(stream_array,axis=1)))

def pws(stream_array,pws_power,ktime,fs):
    """
    Produces a phase weigted stack of cross correlations by producing a phase weighted stack using the
    Hilbert transoform and then multiplying it by a linear stack.
    """
    len_stream = stream_array.shape[0]
    phase_array = gen_phase_array(stream_array)
    p_stack = np.power(np.abs(phase_array.sum(axis=0)/len_stream),pws_power)
    # Tested smoothing but didn't do anything
    # kernel = np.ones((int(ktime*fs)))
    # kernel = kernel/sum(kernel)
    # p_stack = np.convolve(p_stack,kernel,mode="same")
    l_stack = linear_stack(stream_array)
    return l_stack*p_stack

def station_pair_stack(**kwargs):
    """
    Function to call a stacking function to stack cross corelations from a single
    station pair. 

    Inputs:
     * path2component : Path to component file where the station pair data is. / must be at end
                        example: "/raid2/jwf39/borneo_cc/msnoise_dir/STACKS/01/001_DAYS/ZZ/
     * station1       : First station to calculate stack from. Eg "YC_SBF2"
     * station2       : Second station to calculate stack from. Eg "YC_SBF4"
     * cc_stream      : obspy stream containing all the cross correlations for a station pair
     * stack_type     : two options "linear" or "pws" for phase weighted stacking

     Note: Must include either path2component, station1 and station2 OR cc_stream
    """
    # path2component=None
    # station1=None
    # station2=None
    # cc_stream=None
    #Defaults for stack type and phase weighted stacking
    stack_type = "linear" if not "stack_type" in kwargs else kwargs["stack_type"]
    pws_power = 2 if not "pws_power" in kwargs else kwargs["pws_power"]
    ktime = 1
    #
    if "station1" in kwargs and "station2" in kwargs and "path2component" in kwargs:
        station1 = kwargs["station1"]
        station2 = kwargs["station2"]
        path2component = kwargs["path2component"]
        station_pair = station1+"_"+station2
        if not os.path.exists(path2component+station_pair):
            station_pair = station2+"_"+station1
        if not os.path.exists(path2component+station_pair):
            raise ValueError("Path to data no found. Invalid path2component or station pair.")
        #
        path = path2component + station_pair +"/*.MSEED"
        cc_stream = obspy.read(path)
        cc_stats = cc_stream[0].stats
    elif "cc_stream" in kwargs:
        cc_stream = kwargs["cc_stream"]
        assert type(cc_stream)==obspy.Stream
        cc_stats = cc_stream[0].stats
    else:
        raise KeyError("The ")
    fs = cc_stats["sampling_rate"]
    stream_array = stream2array(cc_stream)
    if stack_type == "linear":
        cc_stack = linear_stack(stream_array)
    elif stack_type == "pws":
        cc_stack = pws(stream_array,pws_power,ktime,fs)
    else:
        raise ValueError("%s not a valid stacking type." % (stack_type))
    #
    cc_stacked_trace = obspy.Trace(data=cc_stack,header=cc_stats)
    return cc_stacked_trace

#######################################################################################################
#                                         Rotation Functions                                          #
#######################################################################################################

def rotate_cc(ee_stacked_trace,en_stacked_trace,nn_stacked_trace,ne_stacked_trace,az,baz):
    sinaz = sin(np.pi*az/180)
    cosaz = cos(np.pi*az/180)
    sinbaz = sin(np.pi*baz/180)
    cosbaz = cos(np.pi*baz/180)
    # rotation_matrix = np.array([[-cosaz*cosbaz,  cosaz*sinbaz, -sinaz*sinbaz,  sinaz*cosbaz],
    #                             [-sinaz*sinbaz, -sinaz*cosbaz, -cosaz*cosbaz, -cosaz*sinbaz],
    #                             [-cosaz*sinbaz,  cosaz*cosbaz,  sinaz*cosbaz,  sinaz*sinbaz],
    #                             [-sinaz*cosbaz,  sinaz*cosbaz,  cosaz*cosbaz, -cosaz*sinbaz]])
    rotation_matrix = np.array([[-cosaz*cosbaz,  cosaz*sinbaz, -sinaz*sinbaz,  sinaz*cosbaz],
                                [-sinaz*sinbaz, -sinaz*cosbaz, -cosaz*cosbaz, -cosaz*sinbaz],
                                [-cosaz*sinbaz, -cosaz*cosbaz,  sinaz*cosbaz,  sinaz*sinbaz],
                                [-sinaz*cosbaz,  sinaz*cosbaz,  cosaz*cosbaz, -cosaz*sinbaz]])
    component_matrix = np.array([ee_stacked_trace.data,
                                 en_stacked_trace.data,
                                 nn_stacked_trace.data,
                                 ne_stacked_trace.data])
    rotated_cc = np.matmul(rotation_matrix,component_matrix)
    #
    cc_stats = ee_stacked_trace.stats
    #
    cc_stats.channel = "TT"
    tt_stacked_trace = obspy.Trace(data=rotated_cc[0,:],header=cc_stats)
    cc_stats.channel = "RR"
    rr_stacked_trace = obspy.Trace(data=rotated_cc[1,:],header=cc_stats)
    cc_stats.channel = "TR"
    tr_stacked_trace = obspy.Trace(data=rotated_cc[2,:],header=cc_stats)
    cc_stats.channel = "RT"
    rt_stacked_trace = obspy.Trace(data=rotated_cc[3,:],header=cc_stats)
    #
    return tt_stacked_trace, rr_stacked_trace, tr_stacked_trace, rt_stacked_trace


#######################################################################################################
#                                            EGF Functions                                            #
#######################################################################################################

def egf(cc_stacked_trace,comp,fmin=0.0166,fmax=2,dxdt=True):
    """
    Function to compute the egf from a cross corelation trace by using the formula
              d  ( NCFab(t) + NCFba(-t) )
    EGFab = - __ |______________________|
              dt (          2           )
    For t >= 0
    Inputs:
    * cc_stacked_trace   : 
    """
    cc_stacked_trace.filter("bandpass",freqmin=fmin,freqmax=fmax,zerophase=True,corners=6)
    x = cc_stacked_trace.data
    fs = cc_stacked_trace.stats["sampling_rate"]
    npts = cc_stacked_trace.stats["npts"]
    delta = cc_stacked_trace.stats["delta"]
    hlf = int((npts-1)/2)
    x1 = x[0:hlf]
    x2 = x[hlf+1:]
    x1 = np.flip(x1)
    xout = (x1+x2)/2
    if dxdt:
        xout = (-1)*np.diff(xout)/delta
    #
    outStats = obspy.core.trace.Stats()
    outStats.sampling_rate = fs
    outStats.delta = delta
    outStats.npts = len(xout)
    outStats.channel = comp
    #
    egf_trace = obspy.Trace(data=xout,header=outStats)
    return egf_trace

def egf_worker(job,save_cc,outdir,stack_type,pws_power):
    """
    Worker function that runs a vertical or horezontal job. For a vertical job ZZ is read in as a stream
    and fed to station_pair_stack and then to egf. For a horezontal job NN,EE,NE,EN is read in as a set
    of streams and then each passed through station_pair_stack sequentialy, the stacked cc are then rotated
    using rotate_cc and the rotated cc are then put through egf.

    If save_cc is True all cc stacked cc will be saved. 
    """
    if job[0] == "vertical":
        try:
            jobtype,net1,sta1,net2,sta2,paths = job
            zz_stream = obspy.Stream()
            for path in paths:
                zz_stream += obspy.read(path)
            #
            zz_stacked_trace = station_pair_stack(cc_stream=zz_stream,stack_type=stack_type,pws_power=pws_power)
            if save_cc:
                zz_stacked_trace.write(f"{outdir}/CC/ZZ/{net1}_{sta1}_{net2}_{sta2}.mseed")
            #
            egf_trace = egf(zz_stacked_trace,"ZZ")
            egf_trace.write(f"{outdir}/EGF/ZZ/{net1}_{sta1}_{net2}_{sta2}.mseed")
            return f"{net1}_{sta1}_{net2}_{sta2}__ZZ"
        except Exception as e:
            print(f"WARNING: Following exeption raised in job: {job[0]}_{job[1]}_{job[2]}_{job[3]}_{job[4]}")
            print(e)
            with open("egf_failed.log","a") as f:
                f.write(f"WARNING: Following exeption raised in job: {job[0]}_{job[1]}_{job[2]}_{job[3]}_{job[4]}\n{net1}_{sta1}_{net2}_{sta2}__ZZ  <-------  FAILED\n")
            return f"{net1}_{sta1}_{net2}_{sta2}__ZZ  <-------  FAILED"
    elif job[0] == "horezontal":
        try:
            jobtype,net1,sta1,net2,sta2,az,baz,comps,pathsEE,pathsEN,pathsNN,pathsNE = job
            #
            ee_stream = obspy.Stream()
            for path in pathsEE:
                ee_stream += obspy.read(path)
            ee_stacked_trace = station_pair_stack(cc_stream=ee_stream,stack_type=stack_type,pws_power=pws_power)
            #
            en_stream = obspy.Stream()
            for path in pathsEN:
                en_stream += obspy.read(path)
            en_stacked_trace = station_pair_stack(cc_stream=en_stream,stack_type=stack_type,pws_power=pws_power)
            #
            nn_stream = obspy.Stream()
            for path in pathsNN:
                nn_stream += obspy.read(path)
            nn_stacked_trace = station_pair_stack(cc_stream=nn_stream,stack_type=stack_type,pws_power=pws_power)
            #
            ne_stream = obspy.Stream()
            for path in pathsNE:
                ne_stream += obspy.read(path)
            ne_stacked_trace = station_pair_stack(cc_stream=ne_stream,stack_type=stack_type,pws_power=pws_power)
            #
            tt_stacked_trace, rr_stacked_trace, tr_stacked_trace, rt_stacked_trace = rotate_cc(ee_stacked_trace,en_stacked_trace,nn_stacked_trace,ne_stacked_trace,az,baz)
            if save_cc:
                tt_stacked_trace.write(f"{outdir}/CC/TT/{net1}_{sta1}_{net2}_{sta2}.mseed")
                rr_stacked_trace.write(f"{outdir}/CC/RR/{net1}_{sta1}_{net2}_{sta2}.mseed")
                tr_stacked_trace.write(f"{outdir}/CC/TR/{net1}_{sta1}_{net2}_{sta2}.mseed")
                rt_stacked_trace.write(f"{outdir}/CC/RT/{net1}_{sta1}_{net2}_{sta2}.mseed")
            #
            for comp in comps:
                if comp == "TT":
                    egf_trace = egf(tt_stacked_trace,comp)
                    egf_trace.write(f"{outdir}/EGF/TT/{net1}_{sta1}_{net2}_{sta2}.mseed")
                if comp == "RR":
                    egf_trace = egf(rr_stacked_trace,comp)
                    egf_trace.write(f"{outdir}/EGF/RR/{net1}_{sta1}_{net2}_{sta2}.mseed")
                if comp == "TR":
                    egf_trace = egf(tr_stacked_trace,comp)
                    egf_trace.write(f"{outdir}/EGF/TR/{net1}_{sta1}_{net2}_{sta2}.mseed")
                if comp == "RT":
                    egf_trace = egf(rt_stacked_trace,comp)
                    egf_trace.write(f"{outdir}/EGF/RT/{net1}_{sta1}_{net2}_{sta2}.mseed")
            return f"{net1}_{sta1}_{net2}_{sta2}__TT_RR"
        except Exception as e:
            print(f"WARNING: Following exeption raised in job: {job[0]}_{job[1]}_{job[2]}_{job[3]}_{job[4]}")
            print(e)
            with open("egf_failed.log","a") as f:
                f.write(f"WARNING: Following exeption raised in job: {job[0]}_{job[1]}_{job[2]}_{job[3]}_{job[4]}\n{net1}_{sta1}_{net2}_{sta2}__TT_RR  <-------  FAILED\n")
            return f"{net1}_{sta1}_{net2}_{sta2}__TT_RR  <-------  FAILED"
    else:
        return f"Invalid job type {job[0]}"