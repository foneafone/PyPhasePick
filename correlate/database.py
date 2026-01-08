"""
Native python imports
"""
import os
import pathlib
import glob
import concurrent.futures
import multiprocessing
import time
import gc

"""
Library imports
"""
import numpy as np
import sqlite3
import obspy
from obspy.core import UTCDateTime
import pandas as pd

"""
Optional tqdm import for terminal progress
"""
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x
    
"""
Local module imports
"""
try:
    from .utils import *
except ModuleNotFoundError:
    from utils import *

class CorrelateLiteDB():

    def __init__(self,workingdir:pathlib.Path, dbfile:str="noiselite.db"):
        """
        Docstring for __init__
        
        :param self: Description
        :param workingdir: Description
        :type workingdir: pathlib.Path
        :param dbfile: Description
        :type dbfile: str
        """
        self.workingdir = pathlib.Path(workingdir)
        self.db_path = self.workingdir / dbfile
        #
        self.connect()
        #
        self.cur.execute("SELECT name FROM sqlite_master")
        res = self.cur.fetchall()
        # print(res)
        if res == []:
            # Database is empty add new tables
            self.build_new_db_tables()
        # self.cur.execute("SELECT name FROM sqlite_master")
        # res = self.cur.fetchall()
        # print(res)
    
    def connect(self):
        self.con = sqlite3.connect(self.db_path)
        self.cur = self.con.cursor()
    
    def disconnect(self):
        self.cur.close()
        self.con.close()
    
    def sql_execute(self,comand):
        self.cur.execute(comand)
        self.con.commit()
    
    def cc_config_value(self,*args):
        """
        Simple function to query the corelateconfig table for specific values
        """
        return [self.cur.execute(f"SELECT value FROM correlateconfig WHERE label LIKE '{arg}'").fetchall()[0][0] for arg in args]
        
    def build_new_db_tables(self):
        """
        Builds the tables if the db is empty
        """
        self.cur.execute("CREATE TABLE data(filepath,station,network,channel,starttime,endtime,maxgap)")
        self.cur.execute("CREATE TABLE stations(station,network,lat,lon)")
        self.cur.execute("CREATE TABLE correlateconfig(label,value)")
        self.cur.execute("CREATE TABLE pickconfig(label,value)")
        self.con.commit()
        #
        self.build_job_table()
    
    def build_job_table(self):
        self.cur.execute("CREATE TABLE jobs(filepath1,network1,station1,filepath2,network2,station2,date,components,status)")
        self.con.commit()
    
    def reset_job_table(self):
        self.cur.execute("DROP TABLE jobs")
        self.build_job_table()
        self.con.commit()
    
    def build_correlate_config(self,input_file:pathlib.Path=None):
        """
        Docstring for build_correlate_config
        
        :param self: Description
        :param input_file: Description
        :type input_file: pathlib.Path
        """
        if input_file is None:
            script_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
            correlate_config_defaults = script_dir / "defaults" / "correlateconfig_defaults.csv"
            # config_values = np.loadtxt(correlate_config_defaults,delimiter=",",dtype=str)
            config_values = np.array(pd.read_csv(correlate_config_defaults,header=None))
            #
            self.cur.executemany("INSERT INTO correlateconfig VALUES(?,?)", config_values)
            self.con.commit()
        
    def print_db(self,table):
        """
        Print a specific table from the database
        
        :param self: db class instance
        :param table: string of the table to print
        """
        #, station, network, starttime, endtime, maxgap
        cmd = f"SELECT * FROM {table}"
        values = self.cur.execute(cmd).fetchall()
        print(f"There are {len(values)} rows in the {table} table")
        match table:
            case "data":
                print("filepath station network channel starttime endtime maxgap")
                print("-------- ------- ------- ------- --------- ------- ------")
                for row in values:
                    filepath,station,network,channel,starttime,endtime,maxgap = row
                    print(f"{filepath:64} {station:5} {network:2}  {channel:3}  {starttime}  {endtime}  {maxgap}")
            case "stations":
                print("station network     lat      lon")
                print("------- -------     ---      ---")
                for row in values:
                    station,network,lat,lon = row
                    print(f"{station:7} {network:7} {lat:7} {lon:7}")
            case "correlateconfig":
                print("label         value    ")
                print("-----         -----")
                for row in values:
                    label,value = row
                    print(f"{label:13} {value}")
            case "jobs":
                print("filepath1 network1 station1 filepath2 network2 station2 date components status")
                print(" ")
                for row in values:
                    filepath1,network1,station1,filepath2,network2,station2,date,components,status = row
                    print(f"{filepath1:64} {network1:4} {station1:6} {filepath2:64} {network2:4} {station2:6} {date:9} {components:4} {status:2}")
    
    ########################################################################################
    #                                    Scan Archive                                      #
    ########################################################################################

    def add_to_data(self, search_dir:pathlib.Path, num_procs:int, num_threads_per_proc:int):
        """
        Docstring for add_to_data
        
        :param self: Description
        :param search_dir: Description
        :type search_dir: pathlib.Path
        :param num_procs: Description
        :type num_procs: int
        :param num_threads_per_proc: Description
        :type num_threads_per_proc: int
        """
        search_dir = pathlib.Path(search_dir)
        # Gather relevant values from config
        filetemplate, starttime, endtime, fileformat = self.cc_config_value("filetemplate", "starttime", "endtime", "fileformat")
        #
        # Setup times
        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime)
        #
        # Get existing file paths from database
        pathlist = self.cur.execute("SELECT filepath FROM data").fetchall()
        print(f"There are {len(pathlist)} existing files in database")
        #
        # Produce a list of UTCDateTime objects that represent each day from starttime to end time
        list_of_days = []
        current_time = UTCDateTime(starttime)
        while current_time < endtime:
            list_of_days.append((search_dir,filetemplate,current_time))
            current_time = current_time + 24*3600
        list_of_days.append((search_dir,filetemplate,endtime))
        # raise KeyboardInterrupt
        #
        # Search the archive each day within the list of days to produce a full list of files
        full_file_list = []
        total_number = 0
        futures = []
        with concurrent.futures.ThreadPoolExecutor(num_procs) as sub_executor:
            for result in sub_executor.map(self.day_scan, list_of_days):
                for file in result:
                    if not (file,) in pathlist:
                        full_file_list.append(file)
                        total_number += 1
        pathlist = []
        #
        # Set environment Variables
        # original_schedule = os.environ["OMP_SCHEDULE"]
        os.environ["OMP_SCHEDULE"] = "STATIC"
        # original_procbind = os.environ["OMP_PROC_BIND"]
        os.environ["OMP_PROC_BIND"] = "CLOSE"
        # original_numthreads = os.environ["OMP_NUM_THREADS"]
        os.environ["OMP_NUM_THREADS"] = "1"
        #
        # Scan the archive to extract metadata
        print(total_number)
        #
        self.disconnect()
        #
        gc.collect()
        #
        queue = multiprocessing.Queue(maxsize=total_number)
        #
        procs = []
        chunks = [chunk for chunk in split_list(full_file_list,num_procs)]
        for i in range(num_procs):
            p = multiprocessing.Process(target=CorrelateLiteDB.launch_archive_search, args=(chunks[i],queue,fileformat,num_threads_per_proc))
            p.start()
            procs.append(p)
        #
        try:
            rows = []
            count = 0
            for i in tqdm(range(total_number)):
                result = queue.get()
                if result is not None:
                    station,network,channel,filepath,starttime,endtime,maxgap = result
                    rows.append((filepath,station,network,channel,starttime.format_iris_web_service(),endtime.format_iris_web_service(),float(maxgap)))
                    #
                    # Commit the SQL every 1000 insertions
                    count += 1
                    if count > 1000:
                        self.connect()
                        self.con.executemany("INSERT INTO data(filepath,station,network,channel,starttime,endtime,maxgap) VALUES (?,?,?,?,?,?,?)",rows)
                        self.con.commit()
                        self.disconnect()
                        count = 0
                        rows = []
            #
            self.connect()
            self.con.executemany("INSERT INTO data(filepath,station,network,channel,starttime,endtime,maxgap) VALUES (?,?,?,?,?,?,?)",rows)
            self.con.commit()
            #
            for p in procs:
                p.join()
            queue.close()
            #
        except KeyboardInterrupt:
            for p in procs:
                p.kill()
            queue.close()
    
    @staticmethod
    def launch_archive_search(file_list,queue,fileformat,num_threads_per_proc):
        """
        Launches a search of an archive
        
        :param file_list: Description
        :param queue: Description
        :param fileformat: Description
        :param num_threads_per_proc: Description
        """
        if num_threads_per_proc > 1:
            with concurrent.futures.ThreadPoolExecutor(num_threads_per_proc) as sub_executor:
                futures = [sub_executor.submit(CorrelateLiteDB.process_one_file, (file,fileformat)) for file in file_list]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    queue.put(result)
        else:
            for file in file_list:
                result = CorrelateLiteDB.process_one_file((file,fileformat))
                queue.put(result)
        
    @staticmethod
    def process_one_file(inputs):
        """
        Extracts the relevant infomation from a single seismic file
        
        :param infile: tuple containing infile, fileformat
            infile - the file to read
            fileformat - the type of file to be read by obspy (e.g. MSEED)
        """
        infile, fileformat = inputs
        start = time.perf_counter()
        st = obspy.read(infile,headonly=True,format=fileformat)
        read_time = time.perf_counter()-start
        if len(st) == 1:
            # print(read_time, str(infile).split("/")[-1], st[0].stats.sampling_rate)
            return (st[0].stats.station, st[0].stats.network, st[0].stats.channel, infile, st[0].stats.starttime, st[0].stats.endtime, 0)
        elif len(st) == 0:
            print(f"Warning: File {infile} contains no traces")
            return None
        else:
            starttime = st[0].stats.starttime
            #
            prev_station = st[0].stats.station
            prev_network = st[0].stats.network
            prev_endtime = st[0].stats.endtime
            prev_channel = st[0].stats.channel
            #
            gaps = []
            for i in range(len(st)-1):
                if prev_station == st[i+1].stats.station and prev_network == st[i+1].stats.network and prev_channel == st[i+1].stats.channel and prev_endtime < st[i+1].stats.starttime:
                    gaps.append(st[i+1].stats.starttime.timestamp - prev_endtime.timestamp)
                    #
                    prev_station = st[i+1].stats.station
                    prev_network = st[i+1].stats.network
                    prev_endtime = st[i+1].stats.endtime
                    prev_channel = st[i+1].stats.channel
                else:
                    print(f"Warning: File {infile} contains jumbled data")
                    return None
            # print(read_time, str(infile).split("/")[-1], st[0].stats.sampling_rate)
            return (prev_station, prev_network, prev_channel, infile, starttime, prev_endtime, np.max(gaps))
    
    @staticmethod
    def day_scan(inputs):
        """
        Use a glob search to search for a specific day in an archive.
        
        :param inputs: A tupe containing search_dir,filetemplate,utcdatetime
            search_dir - the base directory to perform the archive search
            filetemplate - the archive structure from the database
            utcdatetime - a datetime object containing the specific day to search
        """
        search_dir,filetemplate,utcdatetime = inputs
        year = utcdatetime.year
        month = utcdatetime.month
        day = utcdatetime.day
        jday = utcdatetime.julday
        filetemplate = filetemplate.replace("YEAR",str(year))
        filetemplate = filetemplate.replace("MM",str(month))
        filetemplate = filetemplate.replace("DD",str(day))
        filetemplate = filetemplate.replace("DAY",str(jday))
        #
        filetemplate = mass_wildcard_replace(filetemplate,"STA","NET","COMP")
        #
        return glob.glob(str(search_dir / filetemplate))

    ########################################################################################
    #                                    Station list                                      #
    ########################################################################################
    
    def fill_station_db(self,station_csv=None):
        """
        Fills the station table using a supplied csv file that contains the columns titled
        network, station, lat and lon. If no csv file is supplied then it finds all unique
        station occurances within the data file.
        
        :param self: Instance of database class
        :param station_csv: Path to pandas readable csv file that contains the columns 
            'network', 'station', 'lat', 'lon'
        """
        # Get existing stations data
        existing_data = np.array(self.cur.execute("SELECT network,station FROM stations").fetchall(),dtype=str)
        existing_ids = make_station_ids(existing_data)
        if station_csv is None:
            # Fill from the available stations in the data table
            stations_and_networks_from_data = np.array(self.cur.execute("SELECT network,station FROM data").fetchall(),dtype=str)
            ids = make_station_ids(stations_and_networks_from_data)
            #
            # Sort and loop to extract unique station_network id strings
            ids.sort()
            unique_ids = []
            current_id = ids[0]
            for id in ids[1:]:
                if id != current_id:
                    unique_ids.append(current_id)
                    current_id = id
            #
            # Split ids back to network and station
            for id in unique_ids:
                if not id in existing_ids:
                    network,station = id.split("_")
                    self.cur.execute("INSERT INTO stations(station,network,lat,lon) VALUES (?,?,?,?)",(station,network,0,0))
            self.con.commit()
        else:
            # Fill from a station csv file
            station_csv_df = pd.read_csv(station_csv)
            ids = make_station_ids(np.array(station_csv_df[["network","station"]]))
            stations = np.array(station_csv_df["station"])
            networks = np.array(station_csv_df["network"])
            lats = np.array(station_csv_df["lat"])
            lons = np.array(station_csv_df["lon"])
            for id,station,network,lat,lon in zip(ids,stations,networks,lats,lons):
                if not id in existing_ids:
                    self.cur.execute("INSERT INTO stations(station,network,lat,lon) VALUES (?,?,?,?)",(station,network,lat,lon))
            self.con.commit()

    ########################################################################################
    #                                      Job list                                        #
    ########################################################################################

    def fill_job_list(self,reset:bool=False):
        """
        Docstring for fill_job_list

        jobid structure = YEAR_JDAY_NET1_STA1_NET2_STA2_COMP - where NET1_STA1 < NET2_STA2 (where the station ids are sorted)
        
        :param self: Description
        :param reset: 
        """
        if reset:
            # Drop the old jobs table and repopulate
            self.reset_job_table()
            existing_ids = []
        else:
            # Load job ids from jobs table
            existing_ids = make_job_ids(np.array(self.cur.execute("SELECT network1,station1,network2,station2,date,components FROM jobs").fetchall()))
        #
        comps_to_compute,configmaxgap = self.cc_config_value("components","maxgap")
        comps_to_compute = comps_to_compute.split(",")
        unique_comps = []
        for comps in comps_to_compute:
            if len(comps) == 2:
                if not comps[0] in unique_comps:
                    unique_comps.append(comps[0])
                if not comps[1] in unique_comps:
                    unique_comps.append(comps[1])
            else:
                raise ValueError(f"The string {comps} is not a valid pair of components to compute")
        #
        # Get station ids from table
        station_data = np.array(self.cur.execute("SELECT network,station FROM stations").fetchall(),dtype=str)
        station_ids = make_station_ids(station_data)
        station_ids.sort()
        #
        # Make unique station pair purmutations in sorted order
        station_pair_ids = []
        for i in range(len(station_ids)):
            id1 = station_ids[0]
            station_ids = station_ids[1:]
            for id2 in station_ids:
                station_pair_id = f"{id1}_{id2}"
                station_pair_ids.append(station_pair_id)
        #
        for station_pair_id in tqdm(station_pair_ids):
            net1,sta1,net2,sta2 = station_pair_id.split("_")
            #
            station1_data = {}
            station2_data = {}
            for comp in unique_comps:
                data1 = self.find_all_data_and_sort(net1,sta1,comp,configmaxgap)
                data2 = self.find_all_data_and_sort(net2,sta2,comp,configmaxgap)
                #
                station1_data[comp] = data1
                station2_data[comp] = data2
            #
            rows = []
            #
            for component_pair in comps_to_compute:
                comp1 = component_pair[0]
                comp2 = component_pair[1]
                #
                data1 = station1_data[comp1]
                data2 = station2_data[comp2]
                #
                if data1 is not None and data2 is not None:
                    index1 = 0
                    index2 = 0
                    while index1 < data1.shape[0] and index2 < data2.shape[0]:
                        filepath1, year_jday1 = data1[index1,:]
                        filepath2, year_jday2 = data2[index2,:]
                        if year_jday1 == year_jday2:
                            #New Job - YEAR_JDAY_NET1_STA1_NET2_STA2_COMP
                            # jobs(filepath1,network1,station1,filepath2,network2,station2,date,components,status)
                            jobid = f"{year_jday1}_{net1}_{sta1}_{net2}_{sta2}_{component_pair}"
                            if not jobid in existing_ids:
                                row = [filepath1,net1,sta1,filepath2,net2,sta2,year_jday1,component_pair,"T"]
                                rows.append(row)
                                filepath1,network1,station1,filepath2,network2,station2,date,components,status = row
                                print(f"{filepath1:64} {network1:4} {station1:6} {filepath2:64} {network2:4} {station2:6} {date:9} {components:4} {status:2}")
                            index1 += 1
                            index2 += 1
                        elif year_jday1 < year_jday2:
                            index1 += 1
                        elif year_jday1 > year_jday2:
                            index2 += 1
                        else:
                            print("Odd while loop behaviour")
                            break
            #
            self.con.executemany("INSERT INTO jobs(filepath1,network1,station1,filepath2,network2,station2,date,components,status) VALUES (?,?,?,?,?,?,?,?,?)",rows)
            self.con.commit()


    
    def find_all_data_and_sort(self,net,sta,comp,configmaxgap):
        """
        Docstring for find_all_data_and_sort
        
        :param self: Description
        :param net: Description
        :param sta: Description
        :param comp: Description
        """
        cmd = f"SELECT filepath,starttime FROM data WHERE network = '{net}' AND station = '{sta}' AND channel LIKE '%{comp}' AND maxgap <= {configmaxgap}"
        data = np.array(self.cur.execute(cmd).fetchall())
        if data.size != 0:
            data[:,1] = date2yrjday(data[:,1])
            inds = np.argsort(data[:,1])
            data = data[inds,:]
            return data
        else:
            return None
