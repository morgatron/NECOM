def set(bit):
    """Set the specifeid bit (1-indexed) eg. set(8) == 0x80"""
    return 1 << (bit - 1)

def setN(n):
    """Set the first n specified bits eg. setN(7) == 0x7F"""
    return set(n + 1) - 1

def reverse(value, length):
    """Reverse an integer value with length bits eg. reverse(0b10,2) == 0b01"""
    output = 0
    for bit in range(length):
        if value & set(bit + 1) != 0:
            output |= set(length - bit)
    return output

def find_last_set(value):
    """ Returns the position of the last set bit (1-indexed)
        eg. find_last_set(0x8012) == 16 """
    output = 0
    while value > 0:
        value >>= 1
        output += 1
    return output
