from math import tan, atan
from numpy import pi, arange
from numpy.random import uniform

t0 = int(uniform()*2e6)
tCur = t0
pL = [54, 32, 23, 37]
lagsActual = [(t0%period) for period in pL]

phis = [tan((tCur%period)/period*2*pi) for period in pL]

calcedLags = [atan(phi)/2/pi*period + period/2 for phi,period in zip(phis, pL)]

print(calcedLags, lagsActual)

for lagInd in range(4):
    tCur = t0 - calcedLags[lagInd]
    phis = [tan((tCur%period)/period*2*pi) for period in pL]
    calcedLagsNow = [atan(phi)/2/pi*period + 0*period/2 for phi,period in zip(phis, pL)]
    lagsActual = [(tCur%period) for period in pL]
    print(calcedLagsNow, lagsActual)


#for modInd in range(4):
#    lagNow = lags[modInd]
    #t_cycles += lagNow