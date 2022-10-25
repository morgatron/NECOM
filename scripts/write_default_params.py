import shared_parameters
from box import Box as B

glbP = shared_parameters.SharedParams("NECOM")
p0= B(
    pulses=B(
        pump = B( 
           t0= 10e-6,
           tWidth = 100e-6,
           amp = 10.0,
        ),
        bigBy = B(
            t0 = 0e-6,
            tWidth = 50e-6,
            amp = 1.0,
        ),
        By = B(
            t0 = 5e-6,
            tWidth = 5e-6,
            amp = 2.0,
        ),
        tTot=1700e-6,
    ),
    biasFields=B(
        Bx=1.00,
        By=-0.0,
        Bz=-0.00,
    ),
    modsSynced = B(
        Bx = B( amp= 100, period_cycles = 54),
        By = B( amp= 100, period_cycles = 84),
        #...
        pump_Theta = B(amp = 100, period_cycles = 66),
        pump_Phi = B(amp = 100, period_cycles = 104),
    ),
    modsPrec = B(
        Bx = B(amp= 0.02, freq= 20.872e-3),
        By = B(amp = 0.02, freq= 15.1e-3),
        Bz = B(amp = 0.02, freq= 23.1e-3),
    ),
)

glbP.replace(p0)