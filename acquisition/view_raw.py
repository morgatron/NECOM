import numpy as np
import DPM
import pyqtgraph as pg
from time import sleep
from PyQt5.QtWidgets import QApplication
from box import Box
from async_components import ZMQSubReadChunked, ChunkedSourcePlotter


def split_into_minus_plus(t, y):
    N = t.size//2
    y1, y2 = y[:N], y[-N:]
    t1 = t[:N]
    return {'m': {'x':t1, 'y':(y1 - y2)}, 'p': {'x':t1, 'y':(y1 + y2)}}


ref = None
ave = None
Nave = 20
def processStream(data, metaData = {}, Nread = None):
    print(f"{data.keys}")
    tL = data['tL']
    y = data['datL']
    mnY = np.mean(y, axis=0)
    N = mnY.size
    t = np.arange(N)
    if 'dt' in metaData:
         t *= metaData['dt']

    global ref, ave
    if ref is None:
        ref = mnY*0
    if ave is None:
        ave = mnY*0
    ave = ((Nave-1)*ave + mnY)/Nave
    mnY = mnY - ref
    data = {'raw': {'x': t, 'y': mnY}}
    data |= split_into_minus_plus(t, mnY)
    return data# {'data':data, "metaData":metaData, 'Nread': N}

def setRef():
    global ref, ave
    ref[:] = ave


IN_PORT = 5560
def main():
    reader = ZMQSubReadChunked(port = IN_PORT, topic = "raw")
    plotter = ChunkedSourcePlotter(
            inputF = lambda : reader.retrieve(10),
            preProcessF = lambda d: processStream(d['data'], d['metaData']) if d else None,
            label = 'raw',
            defaultPlotKwargs = dict(mode=0), # Replace mode
            poll_interval=0.1
            )
    return plotter



#pp.func= preProc
#def wrap(func):
#    def wrapped(t, data):
#        d = func(t, data)
#        return {'nam'}
#    return wrapped


import time
if __name__ == "__main__":
    app = pg.mkQApp("Plotting Example")
    #pg.exec(u)
    #app = QApplication([])
    plotter = main()
    def stop():
        plotter.stop()
    plotter.start()
    #init()
    #start()
