"""
Usage: 
    sigG = total_sig_G()
    sigs = [sigG.__next__() for k in range(500)]

"""
from numpy import random, arange, array, linspace, cos, sin, exp, pi
import numpy as np
from box import Box
import util
import shared_parameters
from async_components import ZMQChunkedPublisher

glbP = shared_parameters.SharedParams("NECOM")

T1 = 1e-3
w = 2*pi*1e3
t = linspace(0,3e-3, 2000)

# Should read these from shared_parameters
Bx = (1-cos(w*t)) * exp(-t/T1)
By = sin(w*t) * exp(-t/T1)

pump_Theta = cos(w*t)*exp(-t/T1) 
pump_Phi = sin(w*t)*exp(-t/T1) 

DT = 0.02 #  hopeully can keep up!!]


basis = np.array([Bx, By, pump_Theta, pump_Phi])

def make_generator(f_of_t, sclNoise, smthT, dt = DT):
    s = random.normal()
    t = 0
    Nsm = smthT/dt
    def gen():
        nonlocal s,t
        while 1: 
            y = s + f_of_t(t)
            yield y
            t += dt
            s = (Nsm*s + sclNoise*random.normal())/(Nsm+1)
    return gen()

from scipy.interpolate import interp1d
def get_smth_square(period, rise_frac=0.1, amp=1):
    t= linspace(-.5,1.5,200)
    y = 2*util.square_wave(t, 1) - 1
    ySm = util.smooth(y, window_len = int(rise_frac/(t[1]-t[0])) )

    yIntp = interp1d(t, ySm)
    def func(t):
        val = amp*yIntp( (t/period)%1)
        #print(f"t: {t/DT}, period: {period}, val: {val}")
        return val
    return func

def update_sig_gens(mod_pars):
    sig_gens = Box()
    #sig_gens.By = make_generator(lambda t: sin(2*pi*t), .1, smthT=0.1),
    #sig_gens.By = make_generator(lambda t: sin(2*pi*t), .1, smthT=0.1),
    sig_gens.Bx = make_generator(get_smth_square(mod_pars.Bx.period_cycles*DT, amp = mod_pars.Bx.amp), .1, smthT=0.05)
    sig_gens.By = make_generator(get_smth_square(mod_pars.By.period_cycles*DT, amp = mod_pars.By.amp), .1, smthT=0.05)
    sig_gens.pump_Theta = make_generator(get_smth_square(mod_pars.pump_Theta.period_cycles*DT, amp = mod_pars.pump_Theta.amp), .1, smthT=0.1)
    sig_gens.pump_Phi = make_generator(get_smth_square(mod_pars.pump_Phi.period_cycles*DT, amp = mod_pars.pump_Phi.amp), .1, smthT=0.1)
    return sig_gens
            
def total_sig_G(mod_pars, noise =0.1):
    sig_gens = update_sig_gens(mod_pars)
    k=0
    while 1:
        vals = array([next(sig) for sig in sig_gens.values()])
        #print(f"k: {k}, val: {vals}")
        #print(vals)
        yield np.sum(vals[:,None]*basis, axis =0) + np.random.normal(size=t.size)*0.5 + np.random.normal()*0.3
        k+=1

# PUBLISHING
#-----------------

glbPORT_PUBSUB = 5560
import zmq, pickle, time

def publishRaw(data, t=None, Nds=1):
    #print(f"pub args: {len(data)}, {data[0].shape}, {len(t)}")
    msg=b'raw '+ pickle.dumps(dict(data_L=data, dt=1e-6, t_L=t))
    glbSOCKET_PUBSUB.send(msg)
    #print(f"pub: {len(data)}")


def main():
    sender = ZMQChunkedPublisher(port = glbPORT_PUBSUB, topic = "raw")

    sigG = total_sig_G(glbP.P.modsSynced.copy())
    tLast = time.time()
    t0 = tLast
    while 1:
        if glbP.changedSinceLoad():
            print("changed since load!!!!!")
            sigG = total_sig_G(glbP.P.modsSynced.copy())
        print("...")
        # wait a random amount
        slpTime = 0.1 + random.normal()*0.15
        if slpTime <0.03:
            slpTime =0.03
        #print(f"sleep time: {slpTime}")
        time.sleep(slpTime)
        tNow = time.time()
        # Get the elapsed time
        tElapsed = tNow - tLast
        # load an appropriate number of samples in a buffer
        Nsamps = int(tElapsed/DT)
        sigs = [sigG.__next__() for k in range(Nsamps)]
        sender.send(tL = tLast + arange(Nsamps)*DT, datL =  sigs)
        #publishRaw(sigs, t = tLast + arange(Nsamps)*DT)
        tLast += Nsamps*DT



#def beginPublishing():
#    global glbSOCKET_PUBSUB
#    context = zmq.Context()
#    glbSOCKET_PUBSUB = context.socket(zmq.PUB)
#    glbSOCKET_PUBSUB.set_hwm(5)
#    glbSOCKET_PUBSUB.bind("tcp://*:%s" % glbPORT_PUBSUB) 
#
if __name__ == "__main__":
    main()
    #beginPublishing()
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

