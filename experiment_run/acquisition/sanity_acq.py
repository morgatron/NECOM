
import serial
import time
import struct
#MagnetometerXYZ(uT): 8.125 -23.625 53.687 7.62 -24.98 54.12 #

with serial.Serial("COM9") as ser:
    ser.timeout = 1
    ser.baudrate = 115200
    n =0
    tL, BxL, ByL, BzL = [], [], [], []
    tStart = time.time()
    tLastPrint = tStart
    labels = ["t", "Bx", "By", "Bz"]
    files = [open(f"mag_log_{st}_3.bin", 'wb') for st in labels]
    BxSum, BySum, BzSum, n = 0,0,0,0
    while 1:
        line = ser.readline()
        
        try:
            if line.startswith(b"Magnetometer"):
                Bx, By, Bz = [float(el) for el in line.split(b" ")[1:4]]
                n+=1
                BxSum+=Bx
                BySum+=By
                BzSum+=Bz
                #tL.append(tNow)
                #BxL.append(Bx)
                #ByL.append(By)
                #BzL.append(Bz)
                tNow = time.time()
                if tNow-tLastPrint >.5:
                    Bvals = BxSum/n, BySum/n, BzSum/n
                    for file, val in zip(files, [tNow, *Bvals]):
                        file.write(struct.pack("d", val))
                    BxSum, BySum, BzSum, n  = 0,0,0,0
                    print(Bvals)
                    tLastPrint+=0.5
        except ValueError:
            print(f"Problematic line: {line}")