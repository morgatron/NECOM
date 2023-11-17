import numpy as np
import pdb
from BitCrc import BitCrc
crc = BitCrc(16,0x1021, 0xebcc)  

def write_RAF_file(data, fname, sampleRate=None, period=None):
    fout = open(fname, 'wb')
    Npts = np.array(data.size, 'i4').tobytes()

    if period is not None:
        raise NotImplementedError
        periodMode = True
        periodOrSampleRate = period
    else:
        periodMode = False
        periodOrSampleRate = sampleRate*1e6
    periodOrSampleRate = np.array(periodOrSampleRate,'i8').tobytes()


    mode = np.array(2, 'u1').tobytes()
    llVal = np.array(data.min()*1e6, 'i4').tobytes()
    hlVal = np.array(data.max()*1e6, 'i4').tobytes()
    fname = np.array(fname, 'S25').tobytes()

    mn, mx = data.min(), data.max()
    data = (data-(mx-mn)/2) / ((mx - mn)/2) * (2**15-1)
    wfBytes = data.astype('i2').tobytes()
    CRC_WF = np.array(crc.generate(wfBytes), 'u2').tobytes()


    header = Npts + np.array(1,'u1').tobytes() + bytes(1) +\
       mode + fname + periodOrSampleRate + hlVal + llVal + \
        CRC_WF

    CRC_H = np.array(crc.generate(header), 'u2').tobytes()
    
    finHeader = header + CRC_H + bytes(4)

    fout.write(finHeader)
    fout.write(wfBytes)
    fout.close()
        

def read_RAF_file(fname):
    bts = open(fname, 'rb').read()

    Npts = np.frombuffer(bts[:4], 'u4')[0]
    mode = np.frombuffer(bts[6:7],'u1')[0]
    fname = bts[7:32]
    SR = np.frombuffer(bts[32:40], 'i8')[0]
    highL = np.frombuffer(bts[40:44], 'i4')[0]
    lowL = np.frombuffer(bts[44:48], 'i4')[0]
    CRC_WF = np.frombuffer(bts[48:50], 'u2')[0]
    CRC_H = np.frombuffer(bts[50:52], 'u2')[0]
    waveform_data = np.frombuffer(bts[56:],'i2')


    return locals()

