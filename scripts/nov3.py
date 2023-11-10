from time import sleep
baseDir = "recorded/nov3"
N = 256*60*6

if 0:
    for modAmp in [500, 2000]:
        By = 1.2
        N = 256*60*20
        changeParams({"biasFields.By":By,
                "modsSynced.Bx_1.amp":modAmp,
                "modsSynced.Bz_1.amp":modAmp
                      }, bPermanent=True)
                      
        d.wiggler.reset_modulation_phase()
        fname = f"v2_linearity_scan_By{By}_amp{modAmp}"
        sleep(30)
        recordRaw(fname,N, baseDir = baseDir, Nds=16)




if 1:
    for bigByAmp in [0.02,0.5,1.0]:
        N = 256*60*20
        changeParams({"pulses.patternD.bigBy.heights": [bigByAmp, -bigByAmp]
                      }, bPermanent=True)
                      
        d.wiggler.reset_modulation_phase()
        fname = f"bigByAmp_no_aom_bigByAmp{bigByAmp}"
        sleep(30)
        recordRaw(fname,N, baseDir = baseDir, Nds=16)
