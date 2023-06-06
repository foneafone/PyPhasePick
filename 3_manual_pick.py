import numpy as np
import obspy
import time
import matplotlib.pyplot as plt
import os
import glob
import json
import multiprocessing

from EgfLib import FTAN, station_pair_stack, egf, regional_dispersion

#######################################################################
#                             Settings                                #
#######################################################################


stationstxt = ""




#######################################################################
#                               Main                                  #
#######################################################################