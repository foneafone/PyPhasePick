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
        print(res)
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
    
    def cc_config_value(self,*args):
        """
        Simple function to query the corelateconfig table for specific values
        """
        return [self.cur.execute(f"SELECT value FROM correlateconfig WHERE label LIKE '{arg}'").fetchall()[0][0] for arg in args]
    
    def print_db(self,table,**kwargs):
        """
        Docstring for print_db
        
        :param self: Description
        :param table: Description
        """
        #, station, network, starttime, endtime, maxgap
        cmd = f"SELECT * FROM {table}"
        print(cmd)
        values = self.cur.execute(cmd).fetchall()
        match table:
            case "data":
                print("filepath station network starttime endtime maxgap")
                for row in values:
                    filepath,station,network,starttime,endtime,maxgap = row
                    print(f"{filepath:64} {station:5} {network:2}  {starttime}  {endtime}  {maxgap}")
            case "stations":
                print("station network")
                for row in values:
                    station,network = row
                    print(f"{station:7} {network:7}")
        
    def build_new_db_tables(self):
        """
        Builds the tables if the db is empty
        """
        self.cur.execute("CREATE TABLE data(filepath,station,network,starttime,endtime,maxgap)")
        self.cur.execute("CREATE TABLE stations(station,network)")
        self.cur.execute("CREATE TABLE jobs(filepath2,network1,station1,filepath1,network2,station2,date)")
        self.cur.execute("CREATE TABLE correlateconfig(label,value)")
        self.cur.execute("CREATE TABLE pickconfig(label,value)")
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
            config_values = np.loadtxt(correlate_config_defaults,delimiter=",",dtype=str)
            #
            self.cur.executemany("INSERT INTO correlateconfig VALUES(?,?)", config_values)
            self.con.commit()
    
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
                    station,network,filepath,starttime,endtime,maxgap = result
                    rows.append((filepath,station,network,starttime.format_iris_web_service(),endtime.format_iris_web_service(),float(maxgap)))
                    #
                    # Commit the SQL every 1000 insertions
                    count += 1
                    if count > 1000:
                        self.connect()
                        self.con.executemany("INSERT INTO data(filepath,station,network,starttime,endtime,maxgap) VALUES (?,?,?,?,?,?)",rows)
                        self.con.commit()
                        self.disconnect()
                        count = 0
                        rows = []
            #
            self.connect()
            self.con.executemany("INSERT INTO data(filepath,station,network,starttime,endtime,maxgap) VALUES (?,?,?,?,?,?)",rows)
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
            return (st[0].stats.station, st[0].stats.network, infile, st[0].stats.starttime, st[0].stats.endtime, 0)
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
            return (prev_station, prev_network, infile, starttime, prev_endtime, np.max(gaps))
    
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
    #                                    New Jobs                                          #
    ########################################################################################
    
    def new_jobs(self,station_csv):
        """
        Docstring for new_jobs
        
        :param self: Description
        :param station_csv: Description
        """
        