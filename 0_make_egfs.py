import numpy as np
import obspy
import time
import matplotlib.pyplot as plt
import os
import glob
import json
import multiprocessing

from EgfLib import FTAN, station_pair_stack, egf, regional_dispersion

def daily_to_egf(daily_paths,stack_type,pws_power):
    cc_stream = obspy.read(daily_paths)
    #
    cc_stacked_trace = station_pair_stack(cc_stream=cc_stream,stack_type=stack_type,pws_power=pws_power)
    #
    egf_trace = egf(cc_stacked_trace)
    return egf_trace
    

#######################################################################
#                             Settings                                #
#######################################################################

cross_correlations_list = ""






#######################################################################
#                               Main                                  #
#######################################################################

