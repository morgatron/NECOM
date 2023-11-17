from time import sleep
baseDir = "recorded/oct19"
N = 256*60*10
#for By in [2.0, 1.5, 1.0,  0.5,  0]:
#for By in [5.0, 3.0, 1.0,-1.0, -3.0]:
if 0:
    for By in [ 5.0,3.0, 1.0, 0, -1.0, -3.0, -5.0]:
        changeParams({"biasFields.By":By})
        d.wiggler.reset_modulation_phase()
        fname = f"v1_square_mod_ds8_no_pump_mod{By}"
        sleep(30)
        recordRaw(fname,N, baseDir = baseDir, Nds=8)

if 0:
    for modAmp in [500,250, 1000]:
        By = 1.2
        N = 256*60*5
        changeParams({"biasFields.By":By})
        changeParams({"modsSynced.Bx_1.amp":modAmp})
        changeParams({"modsSynced.Bz_1.amp":modAmp})
        d.wiggler.reset_modulation_phase()
        fname = f"v1_square_mod_ds8_no_pump_mod{By}_modAmp{modAmp}"
        sleep(30)
        recordRaw(fname,N, baseDir = baseDir, Nds=8)



modAmp =500
changeParams({"modsSynced.Bx_1.amp":modAmp})
changeParams({"modsSynced.Bz_1.amp":modAmp})
for bigByWidth in [20e-6, 50e-6, 70e-6]:
    changeParams({"pulses.patternD.bigBy.widths":[bigByWidth, bigByWidth]})
    fname = f"v1_square_mod_ds8_no_pump_mod{By}_modAmp{modAmp}_bigByWidth{bigByWidth}"
    recordRaw(fname,N, baseDir = baseDir, Nds=8)