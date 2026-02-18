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
    stream1 = obspy.read(filepath1,format=cc_params["fileformat"])
    stream2 = obspy.read(filepath2,format=cc_params["fileformat"])
    print(stream1)
    print(stream2)
    #
    # Merge any streams
    if len(stream1) > 1:
        stream1 = stream1.merge(fill_value=0)
    if len(stream2) > 1:
        stream2 = stream2.merge(fill_value=0)
    #
    # Check alignment
    
    #
    inv = cc_params["inventory"]
    pre_filt = [0.001, 0.005, 45, 50]
    #
    # Remove response
    print("Starting first response removal")
    response_comp_time = time.perf_counter()
    stream1 = stream1.remove_response(inventory=inv,pre_filt=pre_filt)
    response_comp_time = time.perf_counter() - response_comp_time
    print(f"Remove response took {response_comp_time} s")
    print("Starting second response removal")
    response_comp_time = time.perf_counter()
    stream2 = stream2.remove_response(inventory=inv,pre_filt=pre_filt)
    response_comp_time = time.perf_counter() - response_comp_time
    print(f"Remove response took {response_comp_time} s")
    #