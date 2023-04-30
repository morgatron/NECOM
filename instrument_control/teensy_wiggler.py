import serial
from time import sleep

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

def setMod(ax, amp, period):
    if hasattr(ax, "lower"):
        ax = ax.lower()
        if len(ax)==1:
            ax = ord(ax.lower()) - ord("a")
    if ax not in [0,1,2,3, "dac0", "dac1","dac0_1", "dac1_1"]:
        raise ValueError("ax should resolve to one of 0,1,2,3")
    msg = f"set_mod {ax} {amp} {period}"
    write_and_check_err(msg)

def close():
    global ser
    ser.close()
    del ser
    ser = None