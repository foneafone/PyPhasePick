"""
Native Imports
"""
import sys
import pathlib
import os

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
from correlate import utils


def align_streams_unit_test(plot=True):
    """
    Unit test to check that the align streams function works
    """

    # Two sets of 6 hour 5 Hz sin waves at 100 sps
    tests = [
        {
            "dt" : 0,
            "sps1" : 100,
            "sps2" : 100,
            "desc" : "dt at zero same sample rate"
        },
        {
            "dt" : 0.9483,
            "sps1" : 100,
            "sps2" : 100,
            "desc" : "dt at non-zero same sample rate"
        },
        {
            "dt" : 0,
            "sps1" : 100,
            "sps2" : 50,
            "desc" : "dt at zero stream2 at lower sample rate"
        },
        {
            "dt" : 0.9483,
            "sps1" : 100,
            "sps2" : 50,
            "desc" : "dt at non-zero stream2 at lower sample rate"
        },
        {
            "dt" : 0,
            "sps1" : 50,
            "sps2" : 100,
            "desc" : "dt at zero stream1 at lower sample rate"
        },
        {
            "dt" : 0.9483,
            "sps1" : 50,
            "sps2" : 100,
            "desc" : "dt at non-zero stream1 at lower sample rate"
        }
        ]
    for test in tests:
        dt = test["dt"]
        sps1 = test["sps1"]
        sps2 = test["sps2"]
        print("-------")
        print(test["desc"])

        # Stream 1
        nsamples = 6*60*60*sps1
        t1 = (1/sps1)*np.linspace(0,nsamples-1,nsamples)
        sin5hz_6hours = np.sin(2*np.pi*5*t1)
        sin8hz_6hours = np.sin(2*np.pi*8*t1)
        #
        stats = obspy.core.trace.Stats()
        stats.network = "TS"
        stats.station = "TST1"
        stats.sampling_rate = sps1
        stats.npts = len(sin5hz_6hours)
        stats.starttime = obspy.UTCDateTime("2026-03-05T00:00:00")
        trace1 = obspy.Trace(sin5hz_6hours+sin8hz_6hours,header=stats)
        stream1 = obspy.Stream((trace1))

        # Stream 2
        nsamples = 6*60*60*sps2
        t2 = (1/sps2)*np.linspace(0,nsamples-1,nsamples)
        sin5hz_6hours_withshift = np.sin(2*np.pi*5*t2 + 2*np.pi*5*dt)
        sin8hz_6hours_withshift = np.sin(2*np.pi*8*t2 + 2*np.pi*8*dt)
        stats = obspy.core.trace.Stats()
        stats.network = "TS"
        stats.station = "TST2"
        stats.sampling_rate = sps2
        stats.npts = len(sin5hz_6hours)
        stats.starttime = obspy.UTCDateTime(f"2026-03-05T02:26:{35+dt}")
        trace2 = obspy.Trace(sin5hz_6hours_withshift+sin8hz_6hours_withshift,header=stats)
        stream2 = obspy.Stream((trace2))

        # Plot streams before
        # fullstream = obspy.Stream((stream1[0],stream2[0]))
        # if plot:
        #     print("Input streams that don't overlap or align, check the overlap point")
        #     fullstream.plot(method="full")
        
        # Check trim
        stream1, stream2 = utils.align_streams(stream1,stream2,test=True)
        print(stream1)
        print(stream2)
        if plot:
            print("Output streams that fully overlap and align, check the start and end for alignment")
            # fullstream.plot(method="full", starttime=fullstream[0].stats.starttime-2, endtime=fullstream[0].stats.starttime+2)
            plt.figure()
            plt.plot(np.arange(len(stream1[0].data))/sps1, stream1[0].data)
            plt.plot(np.arange(len(stream2[0].data))/sps2, stream2[0].data)
            plt.xlim(xmin=-0.05,xmax=0.2)
            plt.show()

if __name__ == "__main__":
    align_streams_unit_test(plot=True)