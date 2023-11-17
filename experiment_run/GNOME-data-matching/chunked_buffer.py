
import numpy as np
from bisect import bisect

class ChunkedBuffer:
    def __init__(self, maxTRange = 600):
        self.tL = []
        self.yL = []
        self.maxTRange = maxTRange

    def addData(self, t0, datArr):
        # extend the lists
        self.tL.append(t0)
        self.yL.append(datArr)
        curTRange = self.tRange()
        rangeMagnitude = curTRange[1]- curTRange[0];
        if rangeMagnitude > self.maxTRange:
            self.releaseOlderThan( curTRange[0] +(rangeMagnitude - self.maxTRange) )

    def getDataBetween(self, t1, t2):
        #find index of blob before t1, and after t2
        tR = self.tRange()
        if t1< tR[0] or t2> tR[1]:
            st = f"DATA OUT OF RANGE: {(t1-tR[0],  t2 - tR[1])}"
            print(st)
            raise ValueError(st)
        idx1, idx2 = bisect(self.tL, t1)-1, bisect(self.tL, t2)
        yBroad = np.hstack(self.yL[idx1:idx2])
        idx1, idx2 = np.searchsorted(yBroad[0], [t1, t2] )
        return yBroad[:, idx1:idx2]

    def releaseOlderThan(self, tCutoff):
        if len(self.tL) < 2:
            return
        while self.tL[1] <= tCutoff:
            self.tL.pop(0)
            self.yL.pop(0)
            if len(self.tL)<2:
                break

    def tRange(self):
        if (self.tL):
            return self.yL[0][0][0], self.yL[-1][0][-1]
        else:
            return (0,0)

if __name__ == "__main__":
    cb = ChunkedBuffer()
    
    tNow = 0;
    dt = 0.2
    for k in range(30):
        nSamps = 1 + int(np.random.uniform()*100)
        tArr = tNow + np.arange(nSamps)*dt
        tNow  += nSamps*dt
        dat = np.random.normal(size=(5, nSamps)) + tArr[None,:]
        dat[0] = tArr
        cb.addData(dat[0][0], dat)


