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
    # divalpha = fSettings[8]
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
        raise ValueError("Group velocity calculation is not supported - appologies for the inconvenience.")
        # alpha = distkm/divalpha
        # amplitude = group_ftn(trace.data,delta,central_periods,alpha)
        # c_array = distkm / tt
        # inds = np.argsort(c_array)
        # c_array = c_array[inds]
        # for i in range(len(central_periods)):
        #     wave_filtered = amplitude[i,:]
        #     # mv = np.max(wave_filtered)
        #     imax = np.argmax(wave_filtered)
        #     lb_i = imax
        #     while wave_filtered[lb_i] > wave_filtered[imax]*0.8:
        #         lb_i = lb_i - 1
        #     ub_i = imax
        #     while wave_filtered[ub_i] > wave_filtered[imax]*0.8:
        #         ub_i = ub_i + 1
        #     signal = np.sum(np.abs(wave_filtered[lb_i:ub_i])**2)
        #     signal = signal/signal_time
        #     noise = np.sum(np.abs(wave_filtered[:lb_i]))**2 + np.sum(np.abs(wave_filtered[ub_i:]))**2
        #     noise = noise/(total_time-signal_time)
        #     # snr = np.sqrt(signal/noise)
        #     snr = signal/noise
        #     snr_with_period[i] = snr
        #     # wave = wave[inds]/np.max(wave[inds])
        #     wave_filtered = wave_filtered[inds]
        #     c_T_array[:,i] = np.interp(c_array_interp,c_array,wave_filtered)
        # # c_T_array = c_T_array/np.max(c_T_array)
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