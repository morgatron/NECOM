import h5py
import time
import shutil
from datetime import datetime
import util
import os
import dmt_files as dmt
import numpy as np

import magDataServer as mag_serve


# version 2
# Rough logic: 
# * read new minute-long files coming in from GNOME software (sorted)
# * load corresponding mag data, from the minute before and the minute after
# * copy GNOME file to new location
# * look for minute flag rising edge in GNOME data
# * take the number of required samples before and after the edge and take them from the
# ... corresponding files as a numpy array
# * interpolate the numpy array for the required data
# * insert into the new hdf file
# * cleanup:
#   * archive processed GNOME data
#   * delete older mag data

# Version 2 Major components:
# * GNOME file sorter and opener
#   * Main interface: isMoreDMTData, loadNextDMTData
# * Actual magnetometer data server
#   * Main interface: getDataAtTimes(t1, t2), clearOlderThan(t), update()




# * read from a stream of incoming experiment data, with sync information
# * Copy the minute long file to a new location
# * Match up the sync information between the two files. In particular, the location of the "minute" flag
#   * if it doesn't match, look for the minute flag (in the GNOME data), and assume that the right minute flag is the closest
#   * also check the second flags all line up
# * most of the time: take the next minute of data, and just put it into the new file


# UTC?GPS strategy
# Will get the MCU to output the minute flag at the first second of every GPS minute
# PPS will be slightly out. Perhaps I can arrange for it to come 1/512 s before the minute flag, i.e. 1 sample of the DMT
# Data "matching" strategy is then hopefully simpler, just look for next

"""
"""



#test
incoming_dir = ""
staging_dir = ""
upload_dir = ""


SIGNAL_NAME_LIST = [ 
    "MagneticFields2",
    "MagneticFields"
]

SAMPLE_RATE_RATIO =4 # DMT samples/mag samples

# cleanup lost DMT files
# ... walk through and cleanup and files from before when the current data is

# cleanup processed DMT files
# ... walk through and archive any file that exists in the target dir

# cleanup empty dirs


def find_idx_of_rising_edges(arr, plateau_thresh = .5):
    """ Find the index of the first rising edge
    (maybe should change this logic to just find the first '1')
    """
    dArr = np.where(np.diff(np.array(arr>plateau_thresh, 'f8') )>.5)[0]
    return dArr

def idx_offs_from_edges(dmt_y, mag_y):#, t_offs_guess=18):
    """ Given the data recorded from the DMT and that from the magnetometer, workout the actual
    time offset between the two.

    Practically, this will probably just look at the minute marker flags
    """
    # Find where the minute marker rising edge is in DMT
    edgeIdxsDmt = find_idx_of_rising_edges(dmt_y, plateau_thresh = 1)
    edgeIdxsMag = find_idx_of_rising_edges(mag_y, plateau_thresh = 0.5)

    if len(edgeIdxsDmt) >1:
        raise ValueError(f"too many rising edges in dmt signal, got {len(edgeIdxsDmt)}")
    if len(edgeIdxsDmt) == 0:
        if dmt_y[0] > 1: #Then assume it was rising just before the first idx!
            edgeIdxsDmt = [0]
        else:
            raise ValueError(f"Found no rising edges in DMT- there should be 2")
    if len(edgeIdxsMag) == 0:
        raise ValueError(f"Found no rising edges in Mag")

    dmtMinuteIdx = edgeIdxsDmt[0]
    magMinuteIdx = edgeIdxsMag[np.argmin(np.abs(dmtMinuteIdx - edgeIdxsMag))]
    idxOffs = magMinuteIdx - dmtMinuteIdx
    return idxOffs
    #target_t = dmt_t[dmt_index]
    #t_guess = target_t + t_offs_guess
    # Find the area near here in mag_markers

    #minute_times 
    #search_slc = mag_minute_flag_times.searchsorted([t_guess-5, t_guess + 5]) # area around
    #mag_index = find_rising_edges(mag_minute_flags[search_slc])

    t_calculated = mag_minute_flag_times[mag_index]
    # calcualte actual offset time (will depend on mag_flags data structure)
    # 
    return t_calculated


# Just for easy debug...
dmt_t = None
mag_sync = None
dmt_y_sync = None
mag_t = None
def process_file(path):
    global mag_sync, dmt_y_sync, dmt_t, mag_t
    try:
        dmt_t, dmt_y_sync = dmt.loadSyncDataFromPath(path)
        #debug help...
        #while dmt_t[0] < tStart+30:
        #    dmt_t += 60
        #offs= (int((time.time()-dmt_t[0])/60)+1)*60
        #if offs>0:
            #dmt_t +=offs
    except (OSError, KeyError) as e:
        print(f"WARNING: COULDN'T OPEN {path}. MAYBE STILL BEING WRITTEN?")
        print(f"error was: {e}")
        return 
    print(f"dmt times: {dmt_t[0]-tStart}, {dmt_t[-1]-tStart}")
    print(f"mag TR: {mag_serve.dataBuffer.tRange()}")
    try:
        
        mag_t, mag_y, mag_sync = mag_serve.getSnapshot(dmt_t[0]-30, dmt_t[-1]+30)
        print(f"mag times: {mag_t[0]}, {mag_t[1]}")
    except ValueError as e:
        print(e)
        availableRange = mag_serve.dataBuffer.tRange()
        if (dmt_t[0]< availableRange[0]+30):
            print("WARNING: FILE HAS TOO OLD DATA. SENDING TO LIMBO")
            dmt.limboFile(path)
            return True
        else:
            print(f"Can't process {path} yet, waiting for data ({availableRange[1]-tStart} < {dmt_t[1]-tStart})")
            return

    print(f"processing {path}...\n\n\n\n\n")
    magIdxOffs = idx_offs_from_edges(dmt_y_sync, mag_sync)
    print(magIdxOffs)
    if (magIdxOffs<0):
        print("magIdxOffs shouldn't be <0!!")

    mag_serve.releaseOld(mag_t[0])
    trim = lambda my: my[np.arange(dmt_t.size) + magIdxOffs]
    replacementD = {name:trim(val) for name,val in zip(SIGNAL_NAME_LIST, mag_y)}
    dmt.sendToUploadWithNew(path,replacementData=replacementD )
    return True
    #dmt.copyAndReplaceData(old_path=path, new_path = upload_dir, replacementData = sig_interpL )


    #for sig in mag_y: #Haven't nailed 
    #    raise NotImplementedError;
    #    #sig_interp = interpolate.interp1d(mag_t, sig_interp)(dmt_t + inds_offset*dmt_delta_t)
    #    sig_interpL.append(sig_interp)

    #return inds_offset

inProgressPath = None
def update():
    global inProgressPath
    mag_serve.update()
    if inProgressPath is None:
        print("mag range: ")
        inProgressPath = dmt.getNextFilePath()
    if not inProgressPath:
        return;
    if process_file(inProgressPath):
        inProgressPath = None
        print(f"processed path {inProgressPath}")

    

#get_new_data()
#for filename in get_sorted_DMT_files():
#    pass;
#    #load file
#    sync_minutes = h5py.File(filename, 'r+')['MagneticFields']
#    # get new mag data
#    # try to match DMT data to mag data
#    # If successful:
#        # copy
#        # insert data
#        # archive old
#        mag_data_buffer.dumpStart() 
#    # if unsuccessful:
#        # if DMT file is too old -i.e. mag times are older than 
#            # give error message
#            # archive as stale
#

#tCutoff = time.time() 
#tCutoff = 1688359179
tStart = time.time()
if __name__ == "__main__":
    #dmt.incomingDir = b"C:\Users\gnome\gnomedatatest2"
    from time import sleep
    mag_serve.setup()
    mag_serve.update()
    #dmt.archiveOlderThan(tStart)

    while 1:
        update()
        r = mag_serve.dataBuffer.tRange()
        print(r[0]-tStart, r[1]-tStart)
        sleep(.2)