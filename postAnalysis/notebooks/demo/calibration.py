
DAC_resistor = 220e3 #Ohms

def fieldFromAmpl(amp, modString):
    
    amp2current = 3.3/4096/DAC_resistor #Amps
    
    calFactors = {
                "Bx":66800*0.993 * amp2current, # nT
                "By":66800 * amp2current, #nT
                "Bz": 91500, #nT
                "pump_Theta":1., # Not calibrated yet
                "pump_Phi":1., # Not calibrated yet
                "table_H":1., # Not calibrated yet
                 }#nT/A
    calFactors['Bx_1'] = calFactors['Bx']
    calFactors['Bz_1'] = calFactors['Bz']
    
    return current*calFactors[modString]