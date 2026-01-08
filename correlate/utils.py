"""
Module containing a couple of utility functions
"""

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

def make_job_ids(job_array):
    """
    Make list of job ids from an extracted array
    
    :param job_array: Description
    """
    if job_array.size != 0:
        year_jday_list = [f"{UTCDateTime(date).year}_{UTCDateTime(date).julday}" for date in job_array[:,4]]
        ids = [f"{year_jday}_{net1}_{sta1}_{net2}_{sta2}" for year_jday,net1,sta1,net2,sta2 in zip(year_jday_list,job_array[:,0],job_array[:,1],job_array[:,2],job_array[:,3])]
    else:
        ids = []
    return ids