"""
Dealing with the h5 files that come out of the DMT (well, sort-of- they come from the GNOME software)
"""
import h5py
import re
import os
from datetime import datetime, timezone
import numpy as np
import time
import shutil
import pdb
import copy
# State:
yetToProcessL = []
lastProcessedTime = 0;

incomingDir = r"C:\Users\gnome\datatest\incoming"
archiveDir = r"C:\Users\gnome\datatest\archive"
limboDir = r"C:\Users\gnome\datatest\limbo"
uploadDir = r"C:\Users\gnome\datatest\upload"


syncDatasetName = 'MagneticFields' # Name of the dataset that has minute data

def dmtTimeToEpochSeconds(date_string, time_string):
    """
    eg. date_string: 2023/06/17
        time_string: 03:45:39.000
    """
    Y,M,D = [int(val) for val in date_string.split("/")]
    t_st_pieces = time_string.split(":")
    H,m = int(t_st_pieces[0]), int(t_st_pieces[1])
    S_ =  float(t_st_pieces[2])
    S = int(S_)
    us = int( 1000000*(S_-S)) 
    timeStamp = datetime(Y,M,D,H,m,S, us, timezone.utc ).timestamp()
    return timeStamp

#External interface:
def getNextFilePath():
    """Check if there is an unprocessed data file, and return it"""
    global yetToProcessL
    if not yetToProcessL:
        yetToProcessL = _sortedListOfDmtFilesIn(incomingDir)
    if yetToProcessL:
        return yetToProcessL.pop(0)
    
    
def loadSyncDataFromPath(path):
    """Return sync data from the file in numpy form"""
    with h5py.File(path, 'r') as f:
        ds = f[syncDatasetName]
        minuteSyncData = ds[:]
        sRate = ds.attrs['SamplingRate(Hz)']
        t0_st = ds.attrs["t0"]
        date_st = ds.attrs["Date"]
    t0_seconds = dmtTimeToEpochSeconds(date_st, t0_st)
    tAx = t0_seconds + np.arange(len(minuteSyncData))/sRate
    return tAx, minuteSyncData

def copyToNewBaseDir(old_path, new_base_path):
    matchD = parse_DMT_path(old_path)
    matchD['base_path'] = new_base_path
    new_path = "/".join(list(matchD.values()))
    os.makedirs(os.path.dirname(new_path), exist_ok=True )
    shutil.copy(old_path, new_path)
    return new_path

def sendToUploadWithNew(old_path, replacementData):
    tmpPath = old_path + ".tmp"
    copyAndReplaceData(old_path, tmpPath, replacementData)
    newPath = copyToNewBaseDir(tmpPath, uploadDir)
    os.remove(tmpPath)
    archiveFile(old_path)
    print("SENT TO UPLOAD: ", old_path)
    return newPath
    

def copyAndReplaceData(old_path,new_path, replacementData):
    """Make a copy of a file with new data in it
    """
    shutil.copy(old_path, new_path)
    replaceData(new_path, replacementData)

def replaceData(path, replacementData):
    """Replace data in a DMT file"""
    with h5py.File(path, 'r+') as f:
        try:
            for key in replacementData.keys():
                #f.copy(f[key],f, key+"_old")
                f[key][:] = replacementData[key][:]
        except AttributeError: #Mustn't be a dictionary
            for ind,dat in enumerate(replacementData):
                f[ind][:] = dat


def archiveFile(old_path):
    newPath = copyToNewBaseDir(old_path, archiveDir)
    os.remove(old_path)
    return newPath

def limboFile(old_path):
    newPath = copyToNewBaseDir(old_path, limboDir)
    os.remove(old_path)
    return newPath
# Internal 
def archiveOlderThan(tCutoff):
    filePaths = _sortedListOfDmtFilesIn(incomingDir)
    for path in filePaths:
        file_time = parse_DMT_filename_time(path)
        print(f"file_time {file_time}")
        if(file_time < tCutoff): #Can you compare time structs??
            print("archiving...")
            archiveFile(path)
        else:
            print("not archiving")

def _sortedListOfDmtFilesIn(target_dir):
   filePaths = []
   files_final = []
   for root, subdirs, files in os.walk(target_dir, topdown=True):
      for name in copy.copy(subdirs):
            try:
                n=int(name)
                #print(n, name)
                #files_final.append()
            except ValueError:
                subdirs.remove(name)
                #print(f"removing: {name}")
        
      #print("root", root)
      #print("subdirs", subdirs)
      #pdb.set_trace()
      fullPaths = [os.path.join(root,*subdirs, file) for file in files]
      filePaths.extend(fullPaths)#sort()

   filePaths.sort(key = lambda path: os.path.split(path)[1])
   return filePaths


# Pure functions
def parse_DMT_filename_time(file_name):
    pattern = "%Y%m%d_%H%M%S"
    stationString, dtStr = file_name.strip('.h5').split("_",1)
    Y,M, D = int(dtStr[:4]), int(dtStr[4:6]), int(dtStr[6:8])
    H, m, S = int(dtStr[9:11]), int(dtStr[11:13]), int(dtStr[13:15])
    #struct_time = time.strptime(dateTimeString, pattern)
    timeStamp = datetime(Y,M,D,H,m,S, 0, timezone.utc ).timestamp()
    return timeStamp

dmt_path_re = re.compile( r"(?P<base_path>.*)/(?P<year>\d\d\d\d)/(?P<month>\d\d)/(?P<day>\d\d)/(?P<file_name>ANU01_\d{8}_\d{6}\.h5)")
def parse_DMT_path(path):
    path = path.replace("\\", "/")
    matchObj = re.match(dmt_path_re, path)
    return matchObj.groupdict()


if __name__ == "__main__":
    import pylab as pl
    import os
    import shutil
    ion()
    ## CAREFUL! This involves deleting trees...
    sourceTestFiles = "c:/Users/gnome/gnomedatatest/source_data"
    tempoaryTestFiles = "C:/Users/gnome/gnomedatatest/incoming"
    incomingDir = tempoaryTestFiles
    stagingDir = "C:/Users/gnome/gnomedatatest/staging"
    archiveDir = "c:/users/gnome/gnomedatatest/archive"

    def rmIfExists(d):
        if os.path.exists(d):
            shutil.rmtree(d)
    rmIfExists(incomingDir)
    rmIfExists(stagingDir)
    rmIfExists(archiveDir)
    shutil.copytree(sourceTestFiles, tempoaryTestFiles)

    testFilePath = getNextFilePath()
    t, syncData = loadSyncDataFromPath(testFilePath)
    pl.plot(t, syncData)
    newMagData = pl.sin(2*pi*t)
    newPath = copyToNewBaseDir(testFilePath, stagingDir)
    replaceData(newPath, {"MagneticFields": newMagData})



    

