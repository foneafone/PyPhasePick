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

    

#######################################################################################################
#                                           FTAN Functions                                            #
#######################################################################################################

def calc_snr(tr_in,distance):
    """
    Calculates an estimate snr by summing the energy of the signal between 2 and 5 km/s and dividing it
    by the remaining energy. 
    """
    trace = tr_in.copy()
    trace.filter("bandpass",freqmin=1/80,freqmax=1,zerophase=True,corners=6)
    egf = trace.data
    fs = trace.stats["sampling_rate"]
    delta = 1/fs
    egf_tt = np.arange(delta,len(egf)*delta+delta,delta)
    #
    mintime = distance/4500
    maxtime = distance/1500
    len_egf = len(egf)*delta
    len_signal = maxtime-mintime
    signal = np.sqrt(sum([x*x for x,t in zip(egf,egf_tt) if t > mintime and t < maxtime])/len_signal)
    noise = np.sqrt(sum([x*x for x,t in zip(egf,egf_tt) if t > maxtime])/(len_egf-maxtime))
    if noise == 0:
        noise = 0.000001
    SNR = signal/noise
    return SNR

def narrow_band_butter(tr_in,central,width,width_type):
    """
    Function to perform a narrow band filter on an array with central period (s)
    and the width (s) of that central period given as central and width.
    """
    trace = tr_in.copy()
    trace = trace.detrend("linear")
    trace = trace.taper(0.01)
    if width_type == "dependent":
        minT = central-(central*width)
        maxT = central+(central*width)
    elif width_type == "fixed":
        minT = central-(width)
        maxT = central+(width)
    else:
        raise ValueError("%s is not a valid width_type" % (width_type))
    #
    fmin = 1/maxT
    fmax = 1/minT
    trace.filter("bandpass",freqmin=fmin,freqmax=fmax,zerophase=True,corners=6)
    out = trace.data
    # norm = np.max((np.max(out),-1*np.min(out)))
    # out = np.array(out)/norm
    return out

def taper_function(x):
    """
    Taper function defined between x: 0->1 that can be stretched and interpolated
    to apply to the early part of the egf.
    """
    stretch = 17
    xshift = 0.8
    top = expit((1*stretch) - xshift*stretch)
    bottom = expit((0*stretch) - xshift*stretch)
    y = (expit((x*stretch) - xshift*stretch) - bottom)/top
    return y

def egf_taper(tt,egf,distance,maxvel=5500,minvel=1400):
    """
    Tapers an egf to between two velocity values using the function taper_function.
    """
    mintime = distance/maxvel
    maxtime = distance/minvel
    #
    taper_t = np.arange(0,1,0.001)  
    taper_x = taper_function(taper_t) # Define half of the taper between 0 and 1
    taper_t = taper_t*mintime # Stretch it to go from zero and the minimum time
    #
    flip_taper_x = np.flip(taper_x)
    flip_taper_t = taper_t + maxtime # Flip taper and set it to go from max time
    #
    taper_x = np.concatenate((taper_x,flip_taper_x,np.array([0])))
    taper_t = np.concatenate((taper_t,flip_taper_t,np.array([tt[-1]]))) # concatanate the two halfs and add a zero at the end of the egf length
    #
    taper_x_interp = np.interp(tt,taper_t,taper_x) # Interpolate the taper to the same time steps as the egf
    egf = egf*taper_x_interp # Apply the taper
    return egf, taper_x_interp

def group(x):
    """
    Finds the envalope of the array x by taking the absolute of the analytical signal.
    """
    x = np.abs(hilbert(x))**2
    return x
def phase(x):
    x = np.angle(hilbert(x))
    x = np.cos(x)
    return x

def gen_c_array(tt,T,w,distance,do_group,minv,maxv):
    """
    Function that converts from a narrow band filtered waveform from the time
    domain into the velocity domain by doing:
    c = d/(t-T/8)
    If finding phase velocity and
    c = d/t
    If finding group velocity

    Inputs:
     * tt - Travel time array [1darray]
     * T - Central period of filtered waveform [float]
     * w - Waveform array [1darray]
     * distance - Interstation distance (m) [float]
     * do_group - If this is group velocity or phase velocity (True for group) [bool]
     * minv - minimum velocity value (km/s) [float]
     * maxv - maximum velocity value (km/s) [float]
    
    Outputs:
     * out_c - Velocity array (minv to maxv) [1darray]
     * out_v - Output waveform [1darray]
    """
    out_c = []
    out_w = []
    distance = distance/1000
    if do_group: # If doing group do c = d/t
        for i in range(len(tt)):
            t = tt[i]
            if t > 0:
                c = distance/t
                out_c.append(c)
                out_w.append(w[i])
    else: # If doing phase do c = d/(t-T/8)
        for i in range(len(tt)):
            t = tt[i] -T/8
            if t > 0:
                c = distance/t
                out_c.append(c)
                out_w.append(w[i])
    out_w = np.array(out_w)
    w_oi = np.array([wi for wi,ci in zip(out_w,out_c) if ci >= minv and ci <= maxv])
    if do_group:
        out_w = out_w - min(w_oi)
        w_oi = w_oi - min(w_oi)
        norm = max(w_oi)
        out_w = out_w/norm
    else:
        norm = np.max((np.max(w_oi),-1*np.min(w_oi)))
        out_w = out_w/norm
    out_c = np.array(out_c)
    out_w = np.array(out_w)
    return out_c, out_w

def filter_worker(i,trace,tt,central_periods,distance,minv,maxv,bandwidth,width_type,snr_vars):
    min_tt_ind, max_tt_ind, signal_time, total_time = snr_vars
    central = central_periods[i]
    wave_filtered = narrow_band_butter(trace,central,bandwidth,width_type)
    #
    # wave_filtered = phase(wave_filtered)
    # wave_filtered = group(wave_filtered)
    #
    # signal = np.sum(np.abs(wave_filtered[min_tt_ind:max_tt_ind])**2)/signal_time
    # noise = (np.sum(np.abs(wave_filtered[:min_tt_ind]))+np.sum(np.abs(wave_filtered[max_tt_ind:])))/(total_time-signal_time)
    # snr = np.sqrt(signal/noise)
    signal = wave_filtered[min_tt_ind:max_tt_ind]
    noise = np.concatenate((wave_filtered[:min_tt_ind],wave_filtered[max_tt_ind:]))
    signal_rms = np.sqrt(np.mean(signal**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    snr = signal_rms/noise_rms
    #
    c_array, wave_filtered  = gen_c_array(tt,central,wave_filtered,distance,False,minv,maxv)
    #
    inds = np.argsort(c_array)
    c_array = c_array[inds]
    wave_filtered = wave_filtered[inds]
    return i, c_array,wave_filtered, snr

@np.vectorize
def get_xfrq(data,datad):
    if np.abs(data) > 0:
        xfrq = np.imag(datad/data)
    else:
        xfrq = 1.0e5
    return xfrq

def group_ftn(x, dt, periods, alpha):
    """
    Frequency-time analysis of a time series.
    Calculates the Fourier transform of the signal (xarray),
    calculates the analytic signal in frequency domain,
    applies Gaussian bandpass filters centered around given
    center periods, and calculates the filtered analytic
    signal back in time domain.
    Returns the amplitude/phase matrices A(f0,t) and phi(f0,t),
    that is, the amplitude/phase function of time t of the
    analytic signal filtered around period T0 = 1 / f0.
    See. e.g., Levshin & Ritzwoller, "Automated detection,
    extraction, and measurement of regional surface waves",
    Pure Appl. Geoph. (2001) and Bensen et al., "Processing
    seismic ambient noise data to obtain reliable broad-band
    surface wave dispersion measurements", Geophys. J. Int. (2007).
    @param dt: sample spacing
    @type dt: float
    @param x: data array
    @type x: L{numpy.ndarray}
    @param periods: center periods around of Gaussian bandpass filters
    @type periods: L{numpy.ndarray} or list
    @param alpha: smoothing parameter of Gaussian filter
    @type alpha: float
    @rtype: (L{numpy.ndarray}, L{numpy.ndarray})
    """
    # Initializing amplitude/phase matrix: each column =
    # amplitude function of time for a given Gaussian filter
    # centered around a period
    amplitude = np.zeros(shape=(len(periods), len(x)))
    phase = np.zeros(shape=(len(periods), len(x)))
    #
    # Fourier transform
    Xa = np.fft.fft(x)
    # aray of frequencies
    freq = np.fft.fftfreq(x.size, d=dt)
    df = 1/(len(x)*dt)
    #
    # analytic signal in frequency domain:
    #         | 2X(f)  for f > 0
    # Xa(f) = | X(f)   for f = 0
    #         | 0      for f < 0
    # with X = fft(x)
    Xa[freq < 0] = 0.0
    Xa[freq > 0] *= 2.0
    # plt.plot(freq, Xa.real, freq, Xa.imag)
    # plt.show()
    #
    for iperiod, T0 in enumerate(periods):
        # bandpassed analytic signal
        f0 = 1.0 / T0
        #
        # plt.plot(freq,np.exp(-alpha * ((freq - f0) / f0) ** 2))
        # plt.show()
        # dXa_f0 = np.array(Xa_f0)
        Xa_f0 = np.array(Xa).copy()
        #
        beta = np.pi
        fac = np.sqrt(beta/alpha)
        freqmax = (1.0+fac)*f0
        freqmin = (1.0-fac)*f0
        if freqmin <= 0.0:
            freqmin = df
            freqmax = f0+(f0-freqmin)
        for i,f in enumerate(freq):
            if freqmin <= f <= freqmax:
                Xa_f0[i] = Xa[i] * np.exp(-alpha * ((f - f0) / f0)**2)
                # dXa_f0[i] = Xa_f0[i]
                #As trace is in velocity divide by 0 + 2*pi*freq*i
                # Xa_f0[i] = Xa_f0[i]/complex(0.0,2*np.pi*f)
            else:
                # dXa_f0[i] = 0 
                Xa_f0[i] = complex(0.0,0.0)
        #
        # back to time domain
        xa_f0 = np.fft.ifft(Xa_f0)
        # dxa_f0 = np.fft.ifft(dXa_f0)
        # filling amplitude and phase of column
        amplitude[iperiod, :] = np.abs(xa_f0)
        # phase[iperiod, :] = np.angle(xa_f0)
    return amplitude

def FTAN(egf_trace,distance,fSettings,threads=1,do_group=False):
    """
    Generates a FTAN array with showing the amplitude against velocity and Period.

    Process:
    Takes an egf and applies a series of narrow band filters at various central periods
    then places them in an array of shape (velocity,period). 

    Inputs:
    * egf       : The input array containing the EGF generated from cross correlations.
                  If the phase velocity is desired input the raw EGF if group velocity is 
                  desired input a bandpass filtered and enveloped EGF.
    * fs        : Sampling frequency of EGF
    * distance  : The great circle distance (m) between the two stations used to compute the velocity and 
                  the minimum valid period. 
    * fSettings : The filter settings in the form: (minT,maxT,dT,bandwidth)
                  * minT      : The smallest central period (s)
                  * maxT      : The largest central period (s)
                  * dT        : The interval between each filter's central frequency in time domain (s)
                  * bandwidth : The width of the bandpass filter in time domain (s). Note this 
                                should be as large as possible but is limited by df (or max travel time).
    """
    trace = egf_trace.copy()
    egf = trace.data
    fs = trace.stats["sampling_rate"]
    # Generate time and period arrays
    minT = fSettings[0]
    maxT = fSettings[1]
    dT = fSettings[2]
    bandwidth = fSettings[3]
    width_type = fSettings[4]
    dv = fSettings[5]
    minv = fSettings[6]
    maxv = fSettings[7]
    divalpha = fSettings[8]
    delta = 1/fs
    #
    distkm = distance/1000
    tt = np.arange(delta,len(egf)*delta+delta,delta)
    max_tt_ind = np.argmin(np.abs(tt-distkm/minv))
    min_tt_ind = np.argmin(np.abs(tt-distkm/maxv))
    total_time = len(egf)*delta
    signal_time = distkm/minv - distkm/maxv
    snr_vars = (min_tt_ind, max_tt_ind, signal_time, total_time)
    # tt, egf_cut = cut_egf(tt,egf,distance,minvel=2000,maxvel=5000)
    # #print(tt)
    # trace.data, taper_x = egf_taper(tt,trace.data,distance)
    central_periods = np.arange(minT,maxT,dT)
    snr_with_period = np.zeros_like(central_periods)
    #
    # Taper and detrend/demean
    trace = trace.detrend("linear")
    trace = trace.taper(0.05)
    #
    c_array_interp = np.arange(minv,maxv+dv,dv) # Velocity array to interpolate to
    c_T_array = np.zeros((len(c_array_interp),len(central_periods)))
    #
    # Generate t-T array and c-T array
    if do_group:
        alpha = distkm/divalpha
        amplitude = group_ftn(trace.data,delta,central_periods,alpha)
        c_array = distkm / tt
        inds = np.argsort(c_array)
        c_array = c_array[inds]
        for i in range(len(central_periods)):
            wave_filtered = amplitude[i,:]
            # mv = np.max(wave_filtered)
            imax = np.argmax(wave_filtered)
            lb_i = imax
            while wave_filtered[lb_i] > wave_filtered[imax]*0.8:
                lb_i = lb_i - 1
            ub_i = imax
            while wave_filtered[ub_i] > wave_filtered[imax]*0.8:
                ub_i = ub_i + 1
            signal = np.sum(np.abs(wave_filtered[lb_i:ub_i])**2)
            signal = signal/signal_time
            noise = np.sum(np.abs(wave_filtered[:lb_i]))**2 + np.sum(np.abs(wave_filtered[ub_i:]))**2
            noise = noise/(total_time-signal_time)
            # snr = np.sqrt(signal/noise)
            snr = signal/noise
            snr_with_period[i] = snr
            # wave = wave[inds]/np.max(wave[inds])
            wave_filtered = wave_filtered[inds]
            c_T_array[:,i] = np.interp(c_array_interp,c_array,wave_filtered)
        # c_T_array = c_T_array/np.max(c_T_array)
    else:
        if threads != 1 and __name__=="__main__":
            procs = []
            with multiprocessing.Pool(threads) as pool:
                for i in range(len(central_periods)):
                    p = pool.apply_async(filter_worker, args=(i,trace,tt,central_periods,distance,minv,maxv,bandwidth,width_type,snr_vars))
                    procs.append(p)
                for p in procs:
                    i, c_array,wave_filtered, snr = p.get()
                    c_T_array[:,i] = np.interp(c_array_interp,c_array,wave_filtered)
                    snr_with_period[i] = snr
                pool.close()
                pool.join()
        else:
            for i in range(len(central_periods)):
                i, c_array,wave_filtered, snr = filter_worker(i,trace,tt,central_periods,distance,minv,maxv,bandwidth,width_type,snr_vars)
                c_T_array[:,i] = np.interp(c_array_interp,c_array,wave_filtered)
                snr_with_period[i] = snr
    return central_periods, tt, c_array_interp, c_T_array, snr_with_period

def calc_and_save_ftan(egf_path,outdir,distance,fSettings,vel_type,threads=1):
    egf_trace = obspy.read(egf_path)[0]
    #
    if vel_type == "group":
        do_group = True
    else:
        do_group = False
    #
    station_pair = egf_path.split("/")[-1].split(".")[0]
    outfile = f"{outdir}/{station_pair}.nc"
    outsnr = f"{outdir}/{station_pair}_snr.npy"
    #
    snr = calc_snr(egf_trace,distance)
    #
    T, tt, c, c_T_array, snr_with_period = FTAN(egf_trace,distance,fSettings,threads=threads,do_group=do_group)
    #
    ftan = xr.DataArray(
        data=c_T_array,
        dims=("velocity","period"),
        coords=dict(
            velocity=c,
            period=T
        )
    )
    ftan.to_netcdf(outfile)
    #
    np.save(outsnr,snr_with_period)
    #
    return station_pair, snr

def high_freq_ftan(egf_trace,distance,fmin,fSettings):
    minf,maxf,df,bandwidth,width_type,dv,minv,maxv = fSettings
    distkm = distance/1000
    if type(egf_trace)==str:
        egf_trace = obspy.read(egf_trace)
        egf_trace = egf_trace[0]
    egf = egf_trace.copy()
    fs = egf.stats["sampling_rate"]
    delta=1/fs
    tt = np.arange(delta,len(egf.data)*delta+delta,delta)
    max_tt_ind = np.argmin(np.abs(tt-distkm/minv))
    min_tt_ind = np.argmin(np.abs(tt-distkm/maxv))
    c_array_interp = np.arange(minv,maxv+dv,dv)
    #
    frequs = np.arange(minf,maxf+df,df)
    central_frequencies = np.array([f for f in frequs if f >= fmin])
    snr_with_frequency = np.zeros_like(frequs)
    #
    c_f_array = np.zeros((len(c_array_interp),len(central_frequencies)))
    for i,f in enumerate(central_frequencies):
        T = 1/f
        wave_filtered = narrow_band_butter(egf,T,bandwidth,width_type)
        #
        signal = wave_filtered[min_tt_ind:max_tt_ind]
        noise = np.concatenate((wave_filtered[:min_tt_ind],wave_filtered[max_tt_ind:]))
        signal_rms = np.sqrt(np.mean(signal**2))
        noise_rms = np.sqrt(np.mean(noise**2))
        snr = signal_rms/noise_rms
        snr_with_frequency[i] = snr
        #
        wave_filtered = phase(wave_filtered)
        c_array, wave_filtered  = gen_c_array(tt,T,wave_filtered,distance,False,minv,maxv)
        inds = np.argsort(c_array)
        c_array = c_array[inds]
        wave_filtered = wave_filtered[inds]
        c_f_array[:,i] = np.interp(c_array_interp,c_array,wave_filtered)
    return c_array_interp, central_frequencies, c_f_array, snr_with_frequency

def hff_worker(key,egf_trace,distance,fmin,fSettings):
    c_array_interp, central_frequencies, c_f_array, snr_with_frequency = high_freq_ftan(egf_trace,distance,fmin,fSettings)
    return key, distance, c_array_interp, central_frequencies, c_f_array

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

def auto_group_picker(job,regional_period,regional_groupvel,fSettings,stopping_threshold,use_matricies=False):
    station_pair, egf_path, distance = job
    maxT = distance/6000
    distkm = distance/1000
    #
    egf_trace = obspy.read(egf_path)
    egf_trace = egf_trace[0]
    #
    trace = egf_trace.copy()
    egf = trace.data
    fs = trace.stats["sampling_rate"]
    # Generate time and period arrays
    minT = fSettings[0]
    maxT = fSettings[1]
    dT = fSettings[2]
    bandwidth = fSettings[3]
    width_type = fSettings[4]
    dv = fSettings[5]
    minv = fSettings[6]
    maxv = fSettings[7]
    divalpha = fSettings[8]
    delta = 1/fs
    #
    decreasing_threshold, increasing_threshold, snr_threshold = stopping_threshold
    #
    tt = np.arange(delta,len(egf)*delta+delta,delta)
    central_periods = np.arange(minT,maxT,dT)
    snr_with_period = np.zeros_like(central_periods)
    #
    alpha = distkm/divalpha
    amplitude = group_ftn(trace.data,delta,central_periods,alpha)
    c_array = distkm / tt
    inds = np.argsort(c_array)
    c_array = c_array[inds]
    #
    pick_u = []
    pick_T = []
    #
    for i,T in enumerate(central_periods):
        wave_filtered = amplitude[i,:]
        # mv = np.max(wave_filtered)
        imax = np.argmax(wave_filtered)
        U = distkm/tt[imax]
        lb_i = imax
        while lb_i > 0 and wave_filtered[lb_i] > wave_filtered[imax]*0.8:
            lb_i = lb_i - 1
        ub_i = imax
        while ub_i < len(wave_filtered) and wave_filtered[ub_i] > wave_filtered[imax]*0.8:
            ub_i = ub_i + 1
        signal = np.sqrt(np.mean(np.abs(wave_filtered[lb_i:ub_i])**2))
        noise = []
        for a in wave_filtered[:lb_i]:
            noise.append(a)
        for a in wave_filtered[ub_i:]:
            noise.append(a)
        noise = np.sqrt(np.mean(np.abs(noise)**2))
        # snr = np.sqrt(signal/noise)
        snr = signal/noise
        snr_with_period[i] = snr
        if snr > snr_threshold and minv < U < maxv:
            pick_u.append(U)
            pick_T.append(T)
    return station_pair, np.array(pick_T), np.array(pick_u)

    
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
    