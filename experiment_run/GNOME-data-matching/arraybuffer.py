import enum
import logging
import unittest

import numpy as np


class Side(enum.IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


class ArrayBuffer (object):

    def __init__(self, dtype=np.float32, init_shape=(2048,)):

        if isinstance(init_shape, int):
            init_shape = (init_shape,)
        self._dtype = dtype
        self._init_shape = init_shape
        self._array = np.zeros(init_shape, dtype)
        self._start = 0
        self._stop = 0
        self._extensions = np.zeros(2048, dtype=[
            ('left', np.int32),
            ('right', np.int32),
            ('lenght', np.int32)
        ])
        self._extension_n = 0

    @property
    def log(self):
        return logging.getLogger(name=__name__)

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return self._array.nbytes

    @property
    def itemsize(self):
        return self._array.itemsize

    def __len__(self):
        return self._stop - self._start

    def __getitem__(self, key):
        return self._array[self._start:self._stop][key]

    def __setitem__(self, key, value):
        self._array[self._start:self._stop][key] = value

    def _check_enlarge(self, side, extlen):

        if side:
            self._extensions[self._extension_n] = (
                extlen if side == Side.LEFT else 0,
                extlen if side == Side.RIGHT else 0,
                len(self) + extlen
            )
            self._extension_n = ((self._extension_n + 1) %
                                 (len(self._extensions)))

        if side == Side.RIGHT and len(self._array) - self._stop >= extlen:
            return
        elif side == Side.LEFT and self._start >= extlen:
            return

        left_ext = np.sum(self._extensions['left'])
        right_ext = np.sum(self._extensions['right'])
        max_lenght = np.max(self._extensions['lenght'])

        mult = np.int32(np.ceil(max_lenght / max(1, left_ext + right_ext)))

        left_ext *= mult
        right_ext *= mult

        self.log.debug('left_ext: %d, right_ext: %d, max_lenght: %d',
                       left_ext, right_ext, max_lenght)

        newshape = (
            left_ext + max(len(self), self._init_shape[0]) +
            extlen + right_ext,
            *self._array.shape[1:]
        )

        if side == Side.LEFT:
            newstop = newshape[0] - right_ext
            newstart = newstop - len(self)
        else:
            newstart = left_ext
            newstop = newstart + len(self)

        assert newstop - newstart == len(self)

        newarr = np.zeros(newshape, self._dtype)
        if newstart < newstop:
            newarr[newstart:newstop] = self[:]

        self.log.debug(
            'Resizing %s. Old start=%s, stop=%s, shape=%s. '
            'New start=%s, stop=%s, shape=%s',
            str(self),
            self._start, self._stop, self._array.shape,
            newstart, newstop, newshape,
        )

        self._start = newstart
        self._stop = newstop
        self._array = newarr

    def extend(self, other):

        ol = len(other)

        self.log.debug('Exteding %s with %s rows to right', str(self), ol)

        self._check_enlarge(Side.RIGHT, ol)
        self._array[self._stop:self._stop + ol] = other[:]
        self._stop += ol

    def extendleft(self, other):

        ol = len(other)

        self.log.debug('Exteding %s with %s rows to left', str(self), ol)

        self._check_enlarge(Side.LEFT, ol)
        self._array[self._start - ol:self._start] = other[:]
        self._start -= ol

    def pop(self, count):

        count = min(count, len(self))

        self.log.debug('Removing %s rows from %s right side', count, str(self))

        ret = self[-count:]

        self._stop -= count

        return ret

    def popleft(self, count):

        count = min(count, len(self))

        self.log.debug('Removing %s rows from %s left side', count, str(self))

        ret = self[:count]

        self._start += count

        return ret

    def clear(self):

        self.log.debug('Clearing %s', str(self))

        self._start = 0
        self._stop = 0
        self._check_enlarge(Side.NONE, 0)


class BufferTest(unittest.TestCase):

    # @unittest.skip('ok')
    def test_extend(self):
        a = ArrayBuffer(dtype=np.int32, init_shape=1024)
        b = np.arange(0, 512)
        a.extend(b)
        self.assertTrue(np.all(a[:] == b))

        c = np.arange(512, 2048)
        a.extend(c)
        self.assertTrue(np.all(a[:] == np.concatenate((b, c))))

        d = np.arange(-1, -1001, -1)
        a.extendleft(d)
        self.assertTrue(np.all(a[:] == np.concatenate((d, b, c))))

    # @unittest.skip('ok')
    def test_pop(self):
        a = ArrayBuffer(dtype=np.int32, init_shape=1024)
        b = np.arange(0, 1024)
        a.extend(b)
        c = a.popleft(512)
        self.assertTrue(np.all(a[:] == b[512:]))
        self.assertTrue(np.all(c[:] == b[:512]))
        a.pop(256)
        self.assertTrue(np.all(a[:] == b[512:-256]))
        a.pop(100000)
        self.assertEqual(len(a), 0)

    # @unittest.skip('ok')
    def test_stream(self):

        in_arr = list()
        out_arr = list()
        buf = ArrayBuffer(dtype=np.int32, init_shape=128)
        for n in range(100):
            a = np.arange(n * 10 + 1, (n + 1) * 10 + 1, dtype=np.int32)
            self.assertEqual(len(a), 10)
            in_arr.append(a)
            buf.extend(a)
            if len(buf) > 20:
                buf.popleft(len(buf) - 20)

            out = buf[:]
            out = out[np.searchsorted(out, a[0], 'left'):]
            self.assertEqual(len(out), 10)

            out_arr.append(out)

        self.assertTrue(np.all(
            np.concatenate(in_arr) == np.concatenate(out_arr)
        ))


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.CRITICAL)

    unittest.main()
