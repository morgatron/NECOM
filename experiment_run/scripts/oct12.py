from time import sleep
baseDir = "recorded/oct13"
N = 256*60*10
#for By in [2.0, 1.5, 1.0,  0.5,  0]:
for By in [5.0, 3.0, -3.0]:
    changeParams({"biasFields.By":By})
    d.wiggler.reset_modulation_phase()
    fname = f"v5_square_mod_ds4_no_pump{By}"
    sleep(30)
    recordRaw(fname,N, baseDir = baseDir)