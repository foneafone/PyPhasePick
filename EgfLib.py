import numpy as np
import obspy
import time
import matplotlib.pyplot as plt
import os
import glob
import json
from scipy.signal import butter, sosfilt, minimum_phase, firls, hilbert
from scipy.special import expit
import multiprocessing

#######################################################################################################
#                                          Stacking Functions                                         #
#######################################################################################################

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
    kernel = np.ones((int(ktime*fs)))
    kernel = kernel/sum(kernel)
    p_stack = np.convolve(p_stack,kernel,mode="same")
    l_stack = linear_stack(stream_array)
    return l_stack*p_stack

def station_pair_stack(path2component=None,station1=None,station2=None,cc_stream=None,stack_type="linear",pws_power=2):
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
    ktime = 1
    #
    if type(cc_stream)==None:
        station_pair = station1+"_"+station2
        if not os.path.exists(path2component+station_pair):
            station_pair = station2+"_"+station1
        if not os.path.exists(path2component+station_pair):
            raise ValueError("Path to data no found. Invalid path2component or station pair.")
        #
        path = path2component + station_pair +"/*.MSEED"
        cc_stream = obspy.read(path)
        cc_stats = cc_stream[0].stats
    else:
        assert type(cc_stream)==obspy.Stream
        cc_stats = cc_stream[0].stats
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
#                                            EGF Functions                                            #
#######################################################################################################

def egf(cc_stacked_trace,fmin=0.0166,fmax=2):
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
    xout = (-1)*np.diff(xout)/delta
    #
    outStats = obspy.core.trace.Stats()
    outStats.sampling_rate = fs
    outStats.delta = delta
    outStats.npts = len(xout)
    outStats.channel = "ZZ"
    #
    egf_trace = obspy.core.trace.Trace(data=xout,header=outStats)
    return egf_trace

#######################################################################################################
#                                           FTAN Functions                                            #
#######################################################################################################

def narrow_band_butter(tr_in,central,width,width_type):
    """
    Function to perform a narrow band filter on an array with central period (s)
    and the width (s) of that central period given as central and width.
    """
    trace = tr_in.copy()
    trace = trace.detrend("linear")
    trace = trace.taper(0.05)
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
    norm = np.max((np.max(out),-1*np.min(out)))
    out = np.array(out)/norm
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
    out = np.abs(hilbert(x))
    return out

def gen_c_array(tt,T,w,distance,do_group,minv,maxv):
    """
    Function that converts from a narrow band filtered waveform from the time
    domain into the velocity domain by doing:
    c = d/(t-T/8)

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

def filter_worker(i,trace,tt,central_periods,distance,minv,maxv,bandwidth,width_type,do_group):
    central = central_periods[i]
    wave_filtered = narrow_band_butter(trace,central,bandwidth,width_type)
    if do_group:
        wave_filtered = group(wave_filtered)
    # wave_filtered = group(wave_filtered)
    c_array, wave_filtered  = gen_c_array(tt,central,wave_filtered,distance,do_group,minv,maxv)
    #
    inds = np.argsort(c_array)
    c_array = c_array[inds]
    wave_filtered = wave_filtered[inds]
    return i, c_array,wave_filtered

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
    delta = 1/fs
    #
    tt = np.arange(delta,len(egf)*delta+delta,delta)
    # tt, egf_cut = cut_egf(tt,egf,distance,minvel=2000,maxvel=5000)
    # #print(tt)
    trace.data, taper_x = egf_taper(tt,trace.data,distance)
    central_periods = np.arange(minT,maxT,dT)
    #
    # Taper and detrend/demean
    trace = trace.detrend("linear")
    trace = trace.taper(0.05)
    #
    c_array_interp = np.arange(minv,maxv+dv,dv) # Velocity array to interpolate to
    c_T_array = np.zeros((len(c_array_interp),len(central_periods)))
    #
    # Generate t-T array and c-T array
    if __name__=="__main__":
        procs = []
        pool = multiprocessing.Pool(threads)
        for i in range(len(central_periods)):
            p = pool.apply_async(filter_worker, args=(i,trace,tt,central_periods,distance,minv,maxv,bandwidth,width_type,do_group))
            procs.append(p)
        for p in procs:
            i, c_array,wave_filtered = p.get()
            c_T_array[:,i] = np.interp(c_array_interp,c_array,wave_filtered)
    return central_periods, tt, c_array_interp, c_T_array