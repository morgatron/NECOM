import numpy as np
pattern = [
    [0.5, 4],
    [0.75,4],
    [1, 4],
    [1.25, 4],
    [1.5,4],
    [2,4],
    [3,3],
    [4,2],
    [5,2],
    [6,2],
    [10,2],
    [35,1],
    [55, 0.6],
    [70, 0.4],
    [80, 0.2],
    [90, 0.2],
    [110,0.2],
    [130, 0.2],
    [160, 0.1],
    [190, 0.1],

]
sampleRate = 300
def make_sin_pattern():
    yL = []
    print("Sin pattern length is", sum([el[1] for el in pattern]))
    for f, T in pattern:
        t_ = np.arange(T*sampleRate)/sampleRate
        y_ = np.sin(2*np.pi*f*t_)
        yL.append(y_)
    return np.hstack(yL)

def calibrationWaveform():
    y = make_sin_pattern()
    T = 16 # Length of square wave-form
    t_ = np.arange(T*sampleRate)/sampleRate
    y_ = np.where( (t_>5) & (t_<15), 1, 0) #Square wave
    y = np.hstack([y, y_])
    t = np.arange(y.size)/sampleRate
    return t, y



