""" Whole point is to allow querying a range of incoming data, and ability to limit memory

The data is stored in a ChunkedBuffer which does most of the actual logic. 
This may need to translate timestamps, however
"""
from chunked_buffer import ChunkedBuffer
from async_components import ZMQSubReadChunked
import numpy as np
# What time units? We need to be consistent
# Main thing is we can do time differences/ordering. Might be easiest in seconds since epoch?
# UTC? GPS time?
# Seconds since epoch?

IN_PORT = 5562
reader = None
dataBuffer = None
INTERPOLATE_BY = 4

def setup():
    global reader, dataBuffer
    reader = ZMQSubReadChunked(IN_PORT, topic="fitted")
    dataBuffer = ChunkedBuffer(maxTRange=1200)

def close():
    reader.close()
    del dataBuffer;




#mag_data_buffer = ArrayBuffer(size = (5*60*128, 5), )
tLastMatch = 0
def update():
    """ Just get the latest data from the magnetometer and add it to the buffer

    """
    new_data = reader.retrieve()
    if new_data:
        dat = new_data['data']
        minute_flags = np.array(dat['flags']) & 1 #?? <- Need to extract minute_flags here
        t = dat['tL']
        #y_Bx = dat['Bx_1']
        #y_Bz = dat['Bz_1']
        y_Bx = dat['anom_x']
        #y_Bx = dat['flags']
        y_Bz = dat['anom_z']
        #datBlock = np.vstack()
        datBlock = np.vstack([t,y_Bx, y_Bz, minute_flags])
        dataBuffer.addData(datBlock[0][0], datBlock)

    #Could trim it if it's too long, here
    #tStart,tLast = dataBuffer.tRange()


def releaseOld(tCutoff):
    dataBuffer.releaseOlderThan(tCutoff)

def getSnapshot(t1, t2):
    #t,*data, sync = dataBuffer.getDataBetween(t1, t2)
    yL = dataBuffer.getDataBetween(t1, t2)
    N0 = yL[0].size
    Nnew = N0*INTERPOLATE_BY
    idx0 = np.arange(N0)
    idxNew = np.arange(Nnew)/4
    yL = [np.interp(idxNew, idx0, y) for y in yL]
    t, *data, sync = yL
    
    return t, data, sync


if __name__ == "__main__":
    import time
    print("testing magDataServer")
    print("Should update for 10 seconds, and keep between 1 and 2 seconds of data in memory")
    # Simply try to pull from the field server and release once in a while
    
    setup()
    tStart = time.time()
    while time.time() < tStart +10: #run for 10 seconds
        update()
        tRange = dataBuffer.tRange()
        print( f"tRange: {tRange}")
        if tRange[1]- tRange[0] >2:
            releaseOld(tRange[0]+1)
