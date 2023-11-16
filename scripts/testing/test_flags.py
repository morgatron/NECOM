from async_components import ZMQSubReadChunked
import numpy as np
from time import sleep, time, ctime


reader = ZMQSubReadChunked(port = 5560, topic = "raw")

def get_more_flags():
    dat = reader.retrieve()
    while dat is None:
        dat = reader.retrieve(Nmin = 128) 
        sleep(0.1)
    flags = dat['data']['flagsL']
    t = dat['data']['tL']
    return t, flags
count = 0;

mFlagsAll = []
sFlagsAll = []


sEdges = []
sEdges_ts = []
mEdges = []
mEdges_ts = []

mFlags_last = 1
sFlags_last = 1

N = 0

last_m_t = None
last_m_ind = None

try: 
    while True:
        t, flags = get_more_flags()
        t = np.array(t)
        minute_flags = np.bitwise_and(flags, 1<<0)
        second_flags = np.bitwise_and(flags, 1<<1)

        mRisingInds = np.where(np.diff(np.hstack([[mFlags_last], minute_flags])) > .5)[0]
        mEdges.extend(mRisingInds+N)
        mEdges_ts.extend(t[mRisingInds])

        #sRisingInds = np.diff(second_flags) > 1
        sRisingInds = np.where(np.diff(np.hstack([[sFlags_last], second_flags])) > .5)[0]
        sEdges.extend(sRisingInds+N)
        sEdges_ts.extend(t[sRisingInds])
        #tAll.extend(t)
        #mFlagsAll.extend(minute_flags)
        #sFlagsAll.extend(second_flags)
        mFlags_last = minute_flags[-1]
        sFlags_last = second_flags[-1]

        if last_m_ind is not None and len(mRisingInds)>0:
            ind = mRisingInds[0]
            dM_ind =ind+N-last_m_ind
            dM_t =t[ind]-last_m_t
            if dM_ind !=  7680 or dM_t !=60:
                st = f"WARNING: dM_ind: {dM_ind}, dM_t: {dM_t}, N: {N}, time: {ctime()}"
                print(st)
                open('test_flags_log.4.txt', 'a').write(st + '\r\n')
        if len(mRisingInds)>0:
            ind = mRisingInds[0]
            last_m_t = t[ind]
            last_m_ind = ind + N

        N+= len(t)
        print(f"num recieved: {N}")
except KeyboardInterrupt:
    pass
mfA = np.array(mFlagsAll)
del mFlagsAll
sfA = np.array(sFlagsAll)
del sFlagsAll