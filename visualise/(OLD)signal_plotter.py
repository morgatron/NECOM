import async_components
import DPM
import numpy as np

# Object that will read new data into a dict with a curve name and a list/array of values
#ZMQSubDataReader...


def update_td_plots(newDataD):
    #for parName, dat in newData.items():
    dpm.addData(parname) # pretty much just this?

def update_fd_plots(newDataD):
    pass
    # FFT new data
    # average with previous traces
    # OR
    # ... FFT whatever the data in the TD window is? Will be slower...

#From pmGui.py
def update_old():
    global updateNum
    for k in range(10):
        newVals=magClient.getNewVals()
        if not newVals:
            break
        updateNum+=1
        t,fitD=newVals#.retrieveVals(socket)
        t = np.array(t) - t0

        #print("got t: {}".format(t))
        #Update magnetic field time-domain graph
        maxTimePoints=graphPT['MaxTimePoints']
        if updateNum>20:
            bFreqPlot=True
            updateNum=0
        else:
            bFreqPlot=False

        #for cv_t, cv_f, V in zip(Bt_curves,Bf_curves, VL):
        for parName, dat in fitD.items():
            #newXData=np.hstack([cv_t.xData, t])
            if parName not in tdPlotWidget.curveD:
                color = colorL[len(tdPlotWidget.curveD)]
                tdPlotWidget.curveD[parName]=tdPlotWidget.plot([],[], pen=col, name=parName) 
            cv_t = tdPlotWidget.curveD[parName]
            Nnew=V.shape[-1]
            if len(cv_t.xData)==0:
                lastX=0
            else:
                lastX=cv_t.xData[-1]
            #newXData=np.hstack([cv_t.xData, lastX+(1+np.arange(Nnew))*1./60.])
            newXData=np.hstack([cv_t.xData, t])
            newYData=np.hstack([cv_t.yData, V])
            cv_t.setData(newXData[-maxTimePoints:], newYData[-maxTimePoints:])

            if bFreqPlot:
                if parName not in fdPlotWidget.curveD:
                    color = colorL[len(fdPlotWidget.curveD)]
                    fdPlotWidget.curveD[parName]=fdPlotWidget.plot([],[], pen=col, name=parName) 
                cv_f = fdPlotWidget.curveD[parName]
                newXData_f,newYData_f=signal.welch(newYData, fs=graphPT['sampleRate'], nperseg=graphPT['nperseg'])
                cv_f.setData(newXData_f, np.sqrt(newYData_f) )
