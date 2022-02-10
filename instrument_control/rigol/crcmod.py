# This file provides support for BitCrc to act as a drop-in replacement for
# crcmod

from BitCrc import BitCrc
import math
import bits
import struct

def mkGenerator(poly, initCrc=None, rev=True, xorOut=0):
    # Find order and unset leading bit
    order = bits.find_last_set(poly) - 1
    poly ^= 1 << order

    # crcmod defaults initCrc to all bits set
    if initCrc == None:
        initCrc = bits.setN(order)

    gen = BitCrc(
        order,
        poly,
        initialValue = initCrc,
        xorOut = xorOut,
        reverseData = rev
    )

    return gen


def mkCrcFun(poly, initCrc=None, rev=True, xorOut=0):
    gen = mkGenerator(poly, initCrc, rev, xorOut)
    def calculate(data):
        return gen.generate(data)

    return calculate

class Crc(object):
    def init_from_gen(self, gen):
        self.gen = gen
        self.digest_size = math.ceil(gen.order / 4)
        self.crcValue = gen.initialValue ^ gen.xorOut

    def __init__(self, poly, initCrc=None, rev=True, xorOut=0):
        self.init_from_gen(mkGenerator(poly, initCrc, rev, xorOut))

    def new(self, arg = None):
        other = Crc.__new__(Crc)
        other.init_from_gen(self.gen)
        return other

    def copy(self):
        other = self.new()
        other.crcValue = self.crcValue
        return other

    def update(self, data):
        if type(data) == str:
            data = struct.unpack("%dB" % len(data), data)

        self.crcValue ^= self.gen.xorOut

        for byte in data:
            if self.gen.reverseData:
                self.crcValue = self.gen.update_byte_r(self.crcValue, byte)
            else:
                self.crcValue = self.gen.update_byte(self.crcValue, byte)

        self.crcValue ^= self.gen.xorOut
