


# On-line zeroing routines


# Structure for control of FGs

## Example usage

'''
    fgs.setPulses(chan='pump', times = [k*tau for k in range(N)], widths = 10e-6, heights = 5, tTot = 1/fRate)
    fgs.Bz.setPulses(chan='Bz', times = [0, 100e-6], widths = 10e-6, heights = 5, bNow=False, tTot = 1/fRate - 5e-6)
    fgs.Bx.setPulses(times = [0, 100e-6], widths = 10e-6, heights = 5, bNow=False)
    fgs.Bz_big.setPulses(times = [0, 100e-6], widths = 10e-6, heights = [1,-1], bNow=False)
    fgs.uploadAll()
'''