"""
Module containing a couple of utility functions
"""

"""
Native imports
"""

"""
Library Imports
"""
import numpy as np
from obspy import UTCDateTime

def mass_wildcard_replace(string,*args):
        """
        Returns string with all values of *args replaced by '*'
        
        :param string: string to be replaced
        :param args: list of strings to return
        """
        for arg in args:
            string = string.replace(arg,"*")
        return string

def split_list(l,n):
    """
    Splits list l into n equal parts
    
    :param l: List to be split
    :param n: number of parts
    """
    for i in range(0, n):
        yield l[i::n]

def make_station_ids(network_station_array):
    """
    Takes in a 2D array of network and stations and converts it into a list of NET_STA ids
    
    :param network_station_array: 2-d array of [networks,station]
    """
    if network_station_array.size != 0:
        ids = [f"{net}_{sta}" for net,sta in zip(network_station_array[:,0],network_station_array[:,1])]
    else:
        ids = []
    return ids

def date2yrjday(date_list):
    return np.array([f"{UTCDateTime(date).year}_{UTCDateTime(date).julday}" for date in date_list])

def make_job_ids(job_array):
    """
    Make list of job ids from an extracted array
    
    :param job_array: Description
    """
    if job_array.size != 0:
        year_jday_list = date2yrjday(job_array[:,4])
        ids = [f"{year_jday}_{net1}_{sta1}_{net2}_{sta2}_{comp}" for year_jday,net1,sta1,net2,sta2,comp in zip(year_jday_list,job_array[:,0],job_array[:,1],job_array[:,2],job_array[:,3],job_array[:,5])]
    else:
        ids = []
    return ids

def clear_low_sps(stream):
    """
    From a stream of traces return a stream with only the traces that have the highest sample rate
    
    :param stream: Input obspy stream
    """
    max_sps = 0
    for trace in stream:
        sps = trace.stats.sampling_rate
        if sps > max_sps:
            max_sps = sps
    return stream.select(sampling_rate=max_sps)

def secondsintoday(utcdatetime):
    """
    Returns the number of seconds since midnight of the same day from a UTCDateTime
    
    :param utcdatetime: UTCDateTime object to process
    """
    year = utcdatetime.year
    month = utcdatetime.month
    day = utcdatetime.day
    midnight = UTCDateTime(year, month, day, 0).timestamp
    return utcdatetime.timestamp - midnight


    
def align_streams(stream1,stream2,test=False):
    """
    Checks stream alignment. This involves checking first that the start times and end 
    times are the same, if they are not the data will be trimmed and phase shifted if 
    nessasary.
    
    :param stream1: Obspy stream for first station
    :param stream2: Obspy stream for second station
    """
    starttime1 = stream1[0].stats.starttime
    starttime2 = stream2[0].stats.starttime
    endtime1 = stream1[0].stats.endtime
    endtime2 = stream2[0].stats.endtime
    #
    starttime1_seconds = secondsintoday(stream1[0].stats.starttime)
    starttime2_seconds = secondsintoday(stream2[0].stats.starttime)
    endtime1_seconds = secondsintoday(stream1[0].stats.endtime)
    endtime2_seconds = secondsintoday(stream2[0].stats.endtime)
    #
    delta1 = stream1[0].stats.delta
    delta2 = stream2[0].stats.delta
    npts1 = stream1[0].stats.npts
    npts2 = stream2[0].stats.npts
    #
    if starttime1_seconds == starttime2_seconds and endtime1_seconds == endtime2_seconds:
        if test:
            print("Both start and end time are same, return inputs")
        # Both start and end time are same, return inputs
        return stream1, stream2
    #
    if starttime1_seconds == starttime2_seconds and endtime1_seconds != endtime2_seconds:
        if test:
            print("The start times are the same (samples aligned) but end times are not")
        # The start times are the same (samples aligned) but end times are not
        endtime = min((endtime1,endtime2))
        stream1.trim(endtime=endtime)
        stream2.trim(endtime=endtime)
        return stream1, stream2
    #
    # If the start times are not the same, trim and get the sample alignment
    if np.fmod(np.abs(starttime1_seconds-starttime2_seconds)/max((delta1,delta2)),1) == 0:
        if test:
            print("Samples are aligned, trim")
        # Samples are aligned, trim
        endtime = min((endtime1,endtime2))
        starttime = max((starttime1,starttime2))
        stream1.trim(endtime=endtime,starttime=starttime)
        stream2.trim(endtime=endtime,starttime=starttime)
        return stream1, stream2
    else:
        if test:
            print("Samples are not aligned, phase shift and trim")
        # Samples are not aligned, phase shift and trim
        endtime = min((endtime1,endtime2))
        stream_num_to_trim = np.argmin((starttime1,starttime2))
        full_time_diff = np.abs(starttime1_seconds-starttime2_seconds)
        #
        if stream_num_to_trim == 0:
            stream_to_trim = stream1
            delta = delta1
            npts = npts1
            starttime = stream2[0].stats.starttime
        else:
            stream_to_trim = stream2
            delta = delta2
            npts = npts2
            starttime = stream1[0].stats.starttime
        #
        # Find time Shift
        full_time_diff_samples = full_time_diff/delta
        whole_samples = int(np.floor(full_time_diff_samples))
        remainder = np.fmod(full_time_diff_samples, 1)
        backward_shift = -(full_time_diff_samples - whole_samples)
        # print(remainder,backward_shift)
        #
        # Do FFT
        stream_to_trim_data = stream_to_trim[0].data
        stream_to_trim_fft = np.fft.fft(stream_to_trim_data)
        #
        # Calculate phase shift
        k = np.linspace(0,npts-1,npts)
        phase_shift = np.exp((-2*np.pi*1j*k*backward_shift)/npts + np.pi*1j*backward_shift)
        phase_shift = np.fft.fftshift(phase_shift)
        #
        # Multiply with fft coeficients
        fft_stream_shifted = stream_to_trim_fft * phase_shift
        #
        # IFFT
        shifted_stream = np.fft.ifft(fft_stream_shifted)
        shifted_stream = np.real(shifted_stream)
        #
        # Remove starting samples
        shifted_stream = shifted_stream[whole_samples:]
        stream_to_trim[0].data = shifted_stream
        stream_to_trim[0].stats.starttime = starttime
        #
        # Trim endtime
        stream_to_trim[0] = stream_to_trim[0].trim(endtime=endtime)
        #
        # Finish and return
        if stream_num_to_trim == 0:
            stream2 = stream2.trim(endtime=endtime)
            stream_to_trim.trim(endtime=stream2[0].stats.endtime)
            return stream_to_trim, stream2
        else:
            stream1 = stream1.trim(endtime=endtime)
            stream_to_trim.trim(endtime=stream1[0].stats.endtime)
            return stream1, stream_to_trim

def npts2nfft(npts):

    if npts & 0x1:
        nfft = 2 * (npts + 1)