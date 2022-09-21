"""
Usage: 
    sigG = total_sig_G()
    sigs = [sigG.__next__() for k in range(500)]

"""
import numpy as np
from numpy import *
from box import Box
import util
import shared_parameters

glbP = shared_parameters.SharedParams("NECOM")

T1 = 1e-3
w = 2*pi*1e3
t = linspace(0,3e-3, 2000)

# Should read these from shared_parameters
By = sin(w*t) * exp(-t/T1)
Bz = (1-cos(w*t)) * exp(-t/T1)

pump_z = cos(w*t)*exp(-t/T1) 
pump_y = sin(w*t)*exp(-t/T1) 

DT = 0.01 #  hopeully can keep up!!]


basis = np.array([By, Bz, pump_y, pump_z])

def make_generator(f_of_t, sclNoise, smthT, dt = DT):
    s = np.random.normal()
    t = 0
    Nsm = smthT/dt
    def gen():
        nonlocal s,t
        while 1: 
            y = s + f_of_t(t)
            yield y
            t += dt
            s = (Nsm*s + sclNoise*np.random.normal())/(Nsm+1)
    return gen()

from scipy.interpolate import interp1d
def get_smth_square(period, rise_frac=0.1, amp=1):
    t= linspace(-.5,1.5,200)
    y = util.square_wave(t, 1)
    ySm = util.smooth(y, window_len = int(rise_frac/(t[1]-t[0])) )

    yIntp = interp1d(t, ySm)
    def func(t):
        return amp*yIntp( (t/period)%1)
    return func

mod_periods =Box(
    By = 18,
    Bz = 21,
    pump_y = 85,
    pump_z = 56,
)

sig_gens = Box( 
        By = make_generator(lambda t: sin(2*pi*t), .1, smthT=0.1),
        Bz = make_generator(lambda t: sin(2*pi*2.1*t +pi/3), .1, smthT=0.1),
        pump_y = make_generator(get_smth_square(30*DT), .1, smthT=0.1),
        pump_z = make_generator(get_smth_square(27*DT), .1, smthT=0.1)
        )

def total_sig_G(noise =0.1):
    while 1:
        vals = array([next(sig) for sig in sig_gens.values()])
        #print(vals)
        yield np.sum(vals[:,None]*basis, axis =0) + np.random.normal(size=t.size)*0.5 + np.random.normal()*0.3

# PUBLISHING
#-----------------

glbSOCKET_PUBSUB = None
glbPORT_PUBSUB = 5560
import zmq, pickle, time

def publishRaw(data, t=None, Nds=1):
    print(f"pub args: {len(data)}, {data[0].shape}, {len(t)}")
    msg=b'raw '+ pickle.dumps(dict(data=data, dt=1e-6, t=t))
    glbSOCKET_PUBSUB.send(msg)
    print(f"pub: {len(data)}")

from numpy import random, arange, array
from numpy.random import normal

def beginPublishing():
    global glbSOCKET_PUBSUB
    context = zmq.Context()
    glbSOCKET_PUBSUB = context.socket(zmq.PUB)
    glbSOCKET_PUBSUB.set_hwm(5)
    glbSOCKET_PUBSUB.bind("tcp://*:%s" % glbPORT_PUBSUB) 

    sigG = total_sig_G()
    tLast = time.time()
    t0 = tLast
    while 1:
        # wait a random amount
        slpTime = 0.1 + normal()*0.15
        if slpTime <0.03:
            slpTime =0.03
        print(f"sleep time: {slpTime}")
        time.sleep(slpTime)
        tNow = time.time()
        # Get the elapsed time
        tElapsed = tNow - tLast
        # load an appropriate number of samples in a buffer
        Nsamps = int(tElapsed/DT)
        print(f"Nsamps: {Nsamps}")
        # publish the buffer
        print("get samps")
        sigs = [sigG.__next__() for k in range(Nsamps)]
        publishRaw(sigs, t = arange(tLast, tNow, DT))
        tLast = tLast + Nsamps*DT

if __name__ == "__main__":
    beginPublishing()
    if 0:
        from pylab import *
        ion()
        plot(t, pump_y);
        plot(t, pump_z);
        plot(t, By);
        plot(t, Bz);
        sigG = total_sig_G()
        sigs = [sigG.__next__() for k in range(500)]
        sigA = np.array(sigs)

