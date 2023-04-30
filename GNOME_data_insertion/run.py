import h5py
import os 
from time import sleep
import pathlib
from shutil import copyfile
import numpy as np


#function to return files in a directory
def fileInDirectory(dir_name: str):
    fileNames= [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    if fileNames:
        fileNames.sort(os.path.getmtime()) # sort by modifciation time
        return fileNames[0] #return only the oldest



src_dir = "C:/Users/gnome/gnomedatatest/1823/05/11"
dest_dir = "C:/Users/gnome/gnomedatatest/1823/05/11/out"

expectedMinuteIndex = -1


pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)
while 1:
    fileName = fileInDirectory(src_dir)
    src_path = os.path.join(src_dir,fileName)
    if fileName is not None:
        # copy to new folder
        copyfile(src_path, dest_dir)
        # load the copy
        h5file = h5py.File(os.path.join(dest_dir,fileName), 'r+')     # open the file
        syncMinutes = h5file['MagneticFields'] 
        minuteIndex = np.where(syncMinutes>1)[0][0]
        if minuteIndex != expectedMinuteIndex:
            print(f"in GNOME date: minute index is {minuteIndex}, expected {expectedMinuteIndex}. Resetting...")
            expectedMinuteIndex = minuteIndex

        # Get next 60 seconds of NECOM data
        # ... getData().getMinute()
        # check for correct number of samples.
        # In case of failure, read timestamps of NECOM data and compare to expected
        # h5file['MagneticFields'].t0 ... necomData.t
        # h5file['MagneticFields'][...] = necomData[sigX]
        # h5file['MagneticFields2'][...] = necomData[sigY]
        # send 
        # newD
        necomDataToProcess.

        #delete src_path
        #os.remove(src_path)

    sleep(10)

data = f1['meas/frame1/data']       # load the data
data[...] = X1                      # assign new values to data
f1.close()      