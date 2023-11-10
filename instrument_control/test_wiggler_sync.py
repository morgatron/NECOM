import teensy_wiggler as tw
import time
from datetime import datetime

diffL = []
def monitor():
    with tw.serial.Serial(tw.mcu_addr) as ser:
        while(1):
            recv = ser.read_all()
            if recv:
                for line in recv.split(b'\r'):
                    if line.strip().startswith(b"PPS arrivals since"):
                        dt = datetime.utcnow()
                        try:
                            num = int(line[4:].split(b" ")[-1]) # num PPS arrivals
                            diff= (num - dt.second)%60
                            frac = dt.microsecond/1000000
                            print(f"diff: {diff}, | {frac:.2f} | {(num -time.time())%60}" )
                            diffL.append(frac)
                            if len(diffL)>2:
                                print((diffL[-1]-diffL[0])/len(diffL))
                        except ValueError as e:
                            print("interrupted: ", e)
