"""
Native python imports
"""
import os
import pathlib
import glob
import concurrent.futures

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

class CorrelateLiteDB():

    def __init__(self,workingdir:pathlib.Path, dbfile:str="noiselite.db"):
        self.workingdir = pathlib.Path(workingdir)
        self.db_path = self.workingdir / dbfile
        #
        self.con = sqlite3.connect(self.db_path)
        self.cur = self.con.cursor()
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
    
    def cc_config_value(self,*args):
        """
        Simple function to query the corelateconfig table for specific values
        """
        return [self.cur.execute(f"SELECT value FROM correlateconfig WHERE label LIKE '{arg}'").fetchall()[0][0] for arg in args]
        
    def build_new_db_tables(self):
        """
        Builds the tables if the db is empty
        """
        self.cur.execute("CREATE TABLE data(filepath,station,network,starttime,endtime,maxgap)")
        self.cur.execute("CREATE TABLE stations(station,network)")
        self.cur.execute("CREATE TABLE jobs(filepath2,network1,station1,filepath1,network2,station2,date)")
        self.cur.execute("CREATE TABLE correlateconfig(label,value)")
        self.cur.execute("CREATE TABLE pickconfig(label,value)")
    
    def build_correlate_config(self,input_file:pathlib.Path=None):
        """
        
        """
        if input_file is None:
            script_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
            correlate_config_defaults = script_dir / "defaults" / "correlateconfig_defaults.csv"
            config_values = np.loadtxt(correlate_config_defaults,delimiter=",",dtype=str)
            #
            self.cur.executemany("INSERT INTO correlateconfig VALUES(?,?)", config_values)
            self.con.commit()
    
    def add_to_data(self,search_dir:pathlib.Path):
        """
        
        """
        search_dir = pathlib.Path(search_dir)
        # Gather relevant values from config
        filetemplate, starttime, endtime = self.cc_config_value("filetemplate", "starttime", "endtime")
        #
        # Setup times
        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime)
        #
        # Produce a list of UTCDateTime objects that represent each day from starttime to end time
        list_of_days = []
        current_time = UTCDateTime(starttime)
        while current_time < endtime:
            list_of_days.append((search_dir,filetemplate,current_time))
            current_time = current_time + 24*3600
        list_of_days.append((search_dir,filetemplate,endtime))
        #
        # Search the archive each day within the list of days to produce a full list of files
        full_file_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for result in executor.map(self.day_scan, list_of_days):
                full_file_list.extend(result)
        #
        # Scan the archive to extract metadata
        print(len(full_file_list))
        with concurrent.futures.ThreadPoolExecutor(32) as executor:
            insert_df = pd.DataFrame(columns=["filepath","station","network","starttime","endtime","maxgap"])
            #
            futures = [executor.submit(self.process_one_file, inputs) for inputs in full_file_list]
            #
            for future in tqdm(concurrent.futures.as_completed(futures)):
                result = future.result()
                station,network,filepath,starttime,endtime,maxgap = result
                #
                insert_df.loc[-1] = [filepath,station,network,starttime,endtime,float(maxgap)]
                # values = (filepath,station,network,starttime,endtime,maxgap)
                # self.cur.execute("INSERT IGNORE INTO data VALUES (?,?,?,?,?,?)",values)
                # self.con.commit()
                #
                # Commit the SQL every 1000 insertions
                # count += 1
                # if count > 1000:
                #     count = 0
            print(insert_df)

        
    @staticmethod
    def process_one_file(infile):
        """
        Docstring for process_one_file
        
        :param infile: Description
        """
        st = obspy.read(infile,headonly=True)
        if len(st) == 1:
            tr = st[0]
            return (tr.stats.station, tr.stats.network, infile, tr.stats.starttime, tr.stats.endtime, 0)
        elif len(st) == 0:
            print(f"Warning: File {infile} contains no traces")
            return None
        else:
            tr = st[0]
            starttime = tr.stats.starttime
            #
            prev_station = tr.stats.station
            prev_network = tr.stats.network
            prev_endtime = tr.stats.endtime
            prev_channel = tr.stats.channel
            #
            gaps = []
            for i in range(len(st)-1):
                tr = st[i+1]
                if prev_station == tr.stats.station and prev_network == tr.stats.network and prev_channel == tr.stats.channel and prev_endtime < tr.stats.starttime:
                    gaps.append(tr.stats.starttime.timestamp - prev_endtime.timestamp)
                    #
                    prev_station = tr.stats.station
                    prev_network = tr.stats.network
                    prev_endtime = tr.stats.endtime
                    prev_channel = tr.stats.channel
                else:
                    print(f"Warning: File {infile} contains jumbled data")
                    return None
            return (prev_station, prev_network, infile, starttime, prev_endtime, np.max(gaps))




        # return (station, network, filepath, starttime, endtime)
    
    @staticmethod
    def day_scan(inputs):
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
        filetemplate = CorrelateLiteDB.mass_wildcard_replace(filetemplate,"STA","NET","COMP")
        #
        return glob.glob(str(search_dir / filetemplate))

    @staticmethod
    def mass_wildcard_replace(string,*args):
        for arg in args:
            string = string.replace(arg,"*")
        return string