"""Start everything from here

What is the procedure supposed to be as at August 17?
* Start data streaming (using windows terminal script currently)
* acquisition/view_raw to plot it "real time"
* control.py: run in script (init_comms, setupExperiment)
* processing: signal_fit_and_serve # To process and serve results
* processing: signature_calculator # to update signatures on the fly
"""


# Start acquisition terminal
# e.g. call("ipython")? Or wt.exe?
# cd acquisition
# ipython -i: picoscope_acquire.py
# Start controller process and terminal
# wt.exe contrller
# Start scope gui (view_raw)
## 
# Start rt analysis
# Start gradient calculation
# Ideally arrange them in a grid or something!