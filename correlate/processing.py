"""
Module containing all the functions for pre-processing the cross correlations
and performing the cross correlations.

Should be able to use torch for batch cross-correlations.
"""

"""
Native python imports
"""
import os
import pathlib
import glob
import logging
import time

"""
Library imports
"""
import numpy as np
import obspy
from obspy.core import UTCDateTime

"""
Local module imports
"""
try:
    from .utils import *
except ModuleNotFoundError:
    from utils import *

def single_cc_job_torch(row,cc_params):
    """
    Docstring for single_cc_job
    
    :param inputs: Description
    """
    filepath1,network1,station1,filepath2,network2,station2,date,components,status = row
    #
    comp1 = components[0]
    comp2 = components[1]
    #
    # Read data
    stream1 = obspy.read(filepath1,format=cc_params["fileformat"])
    stream2 = obspy.read(filepath2,format=cc_params["fileformat"])
    #

def preprocessing(stream1,stream2, network1,network2, station1,station2, comp1,comp2, cc_params):
    """
    Performs the pre-processing on single days worth of data for two stations as 
    two obspy streams.

    :param stream1: Data from the first file
    :param stream2: Data from the second file
    :param network1: The desired network code for the first station
    :param network2: The desired network code for the second station
    :param station1: The desired station code for the first station
    :param station2: The desired station code for the second station
    :param comp1: The desired component from the first station
    :param comp2: The desired component from the second station
    :param cc_params: Dictionary containing the cross-correlation settings
    """
    # Seclect station and components
    stream1 = clear_low_sps(stream1.select(network=network1,station=station1,component=comp1))
    stream2 = clear_low_sps(stream2.select(network=network2,station=station2,component=comp2))
    #
    # Merge any streams
    if len(stream1) > 1:
        stream1 = stream1.merge(fill_value=0)
    if len(stream2) > 1:
        stream2 = stream2.merge(fill_value=0)
    #
    # Check alignment and trim if needed
    stream1, stream2 = align_streams(stream1,stream2)
    #
    # Resample streams
    stream1 = stream1.resample(cc_params["sample_rate"])
    stream2 = stream2.resample(cc_params["sample_rate"])
    #
    inv = cc_params["inventory"]
    pre_filt = [0.001, 0.005, 45, 50]
    #
    # Remove response
    print("Starting first response removal")
    response_comp_time = time.perf_counter()
    # stream1 = stream1.remove_response(inventory=inv)
    stream1 = stream1.remove_response(inventory=inv,pre_filt=pre_filt)
    response_comp_time = time.perf_counter() - response_comp_time
    print(f"Remove response took {response_comp_time} s")
    print("Starting second response removal")
    response_comp_time = time.perf_counter()
    # stream2 = stream2.remove_response(inventory=inv)
    stream2 = stream2.remove_response(inventory=inv,pre_filt=pre_filt)
    response_comp_time = time.perf_counter() - response_comp_time
    print(f"Remove response took {response_comp_time} s")
    #
    return True