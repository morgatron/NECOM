import serial
from time import sleep
import time

mcu_addr = "COM7"

def getComms():
    return serial.Serial(mcu_addr)
#ser = getComms()

idxMap = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}

def clear(ser):
    _ = ser.read_all()

def read_all():
    ser = serial.Serial(mcu_addr)
    recieved = ser.read_all()
    ser.close()
    print(recieved)
    


def write_and_check_err(cmd, bVerbose=True):
    ser = serial.Serial(mcu_addr)
    if hasattr(cmd, "encode"):
        cmd = cmd.encode()
    
    if not cmd.endswith(b"\r\n"):
        cmd = cmd.strip(b"\r\n") + b"\r\n"
    clear(ser);
    ser.write(cmd)
    sleep(0.05);
    lines = ser.read_all().split(b"\r\n")
    ser.close() 
    if any([line.startswith(b"err:") for line in lines]):
        raise ValueError(f"Err message from mcu: {lines}\n\n CMD: {cmd}")
    if bVerbose:
        print(lines);

def move(ax, steps):
    if hasattr(ax, "lower"):
        ax = ax.lower()
    else:
        ax = idxMap[ax]
    if ax not in "abcd":
        raise ValueError("ax should resolve to one of ABCD")
    msg = f"mv {ax}{steps}"
    write_and_check_err(msg)


def moveMany(movePairs):
    msg = f"mv"
    for ax,steps in movePairs: 
        msg = msg + f" {ax}{steps}"
    write_and_check_err(msg)

def modInt():
    write_and_check_err("modInt")
def modOn():
    write_and_check_err("modOn")
def modOff():
    write_and_check_err("modOff")

def armCalibration():
    write_and_check_err("arm_cal");

def setMod(ax, amp, period):
    if hasattr(ax, "lower"):
        ax = ax.lower()
        if len(ax)==1:
            ax = ord(ax.lower()) - ord("a")
    if ax not in [0,1,2,3, "dac0", "dac1","dac0_1", "dac1_1", "n_a", "n_b", "table_h", "table_v"]:
        raise ValueError("ax should resolve to one of 0,1,2,3 etc")
    msg = f"set_mod {ax} {amp} {period}"
    write_and_check_err(msg)


def armCalAtNext2Minute(offs= 18):
    tNow = time.gmtime()
    tLeft = (1-tNow.tm_min%2)*60 + 59-tNow.tm_sec
    print(f"tleft: {tLeft}")
    if tLeft >1:
        sleep(tLeft-2)
    while time.gmtime().tm_sec != (offs-2)%60:
        sleep(0.2)
        print('.', end='')
    while time.gmtime().tm_sec!= (offs-1)%60:
        sleep(0.02)
    sleep(0.05) # should now be ~ 0.5 seconds before the minute
    armCalibration()

def minute_sync(offs= 18):
    while time.gmtime().tm_sec != (offs-2)%60:
        sleep(0.2)
        print('.', end='')
    while time.gmtime().tm_sec!= (offs-1)%60:
        sleep(0.02)
    sleep(0.1) # should now be ~ 0.5 seconds before the minute
    write_and_check_err("minute_sync 0")

def write_cur_secs():
    tThen = time.gmtime().tm_sec
    tNow = time.gmtime().tm_sec
    while tThen == tNow: # wait until second ticks over
        tNow = time.gmtime().tm_sec
        sleep(0.05)
    write_and_check_err(f"minute_sync {tNow}")

def set_period(period):
    adjusted_period = period/1.000025
    write_and_check_err("set_period {adjusted_period}")

def PPS_lock_on():
    write_and_check_err("PPS_lock_on")

def reset_modulation_phase():
    write_and_check_err("reset_mod_phase");

def close():
    global ser
    ser.close()
    del ser
    ser = None
