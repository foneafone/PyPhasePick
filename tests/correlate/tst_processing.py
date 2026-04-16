"""
Native Imports
"""
import sys
import pathlib
import os
import time

"""
Library Imports
"""
import numpy as np
import obspy
import matplotlib.pyplot as plt


# Append PyPhasePick to path to import
dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(f"{str(dir_path)}/../..")
#
from correlate import processing


def preprocessing_unit_test(plot=True):

    filepath1 = "./data/YC.SBA3..BHZ.D.2018.338"
    network1 = "YC"
    station1 = "SBA3"
    filepath2 = "./data/YC.SBB7..BHZ.D.2018.338"
    network2 = "YC"
    station2 = "SBB7"
    date = "2018_338"
    components = "ZZ"
    status = "T"
    #
    dataless = "./data/YC_MY_9G_station_FEB_20.xml"
    inventory = obspy.read_inventory(dataless)
    #
    cc_params = {
        "fileformat" : "MSEED",
        "inventory" : inventory,
        "sample_rate" : 20
    }
    #
    comp1 = components[0]
    comp2 = components[1]
    #
    # Read data
    stream1 = obspy.read(filepath1,format=cc_params["fileformat"])
    stream2 = obspy.read(filepath2,format=cc_params["fileformat"])

    st = time.perf_counter()
    response1 = stream1[0]._get_response(inventory)
    response2 = stream2[0]._get_response(inventory)
    print(f"Getting Responces took {time.perf_counter()-st}")

    st = time.perf_counter()
    stream1 = stream1.resample(cc_params["sample_rate"])
    print(f"Resample stream1 took {time.perf_counter()-st}")

    # nfft = obspy.signal.util._npts2nfft(stream1[0].npts)
    freq_response, freqs = response1.get_evalresp_response(1./cc_params["sample_rate"], )
    print(freq_response,freqs)

    st = time.perf_counter()
    return_val = processing.preprocessing(stream1,stream2, network1,network2, station1,station2, comp1,comp2, cc_params)
    print(f"Preprocessing for one day took {time.perf_counter()-st}")


if __name__=="__main__":
    preprocessing_unit_test(plot=True)