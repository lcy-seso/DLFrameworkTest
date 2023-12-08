def shiftr(a, s):
    return a >> s if s > 0 else shiftl(a, -s)


def shiftl(a, s):
    return a << s if s > 0 else shiftr(a, -s)


## A generic Swizzle functor
# 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
#                               ^--^  Base is the number of least-sig bits to keep constant
#                  ^-^       ^-^      Bits is the number of bits in the mask
#                    ^---------^      Shift is the distance to shift the YYY mask
#                                       (pos shifts YYY to the right, neg shifts YYY to the left)
#
# e.g. Given
# 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
# the result is
# 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
#
class Swizzle:
    def __init__(self, bits, base, shift):
        assert bits >= 0
        assert base >= 0
        assert abs(shift) >= bits
        self.bits = bits
        self.base = base
        self.shift = shift
        bit_msk = (1 << bits) - 1

        self.yyy_msk = bit_msk << (base + max(0, shift))

        self.zzz_msk = bit_msk << (base - min(0, shift))

    # operator () (transform integer)
    def __call__(self, offset):
        return offset ^ shiftr(offset & self.yyy_msk, self.shift)

    # Size of the domain
    def size(self):
        return 1 << (bits + base + abs(shift))

    # Size of the codomain
    def cosize(self):
        return self.size()

    # print and str
    def __str__(self):
        return f"SW_{self.bits}_{self.base}_{self.shift}"

    # error msgs and representation
    def __repr__(self):
        return f"Swizzle({self.bits},{self.base},{self.shift})"


def get_1d_ids(i, j, rows, cols, row_major=True):
    if row_major:
        return i * cols + j
    else:
        return j * rows + i


def print_result(rows, cols, swizzle_func, row_major=True):
    res = '||'
    for i in range(cols):
        res += f'*{i}*|'
    res += '\n|'
    for i in range(cols + 1):
        res += ':--|'
    res += '\n'

    for i in range(rows):
        res += f'|***{i}***|'
        for j in range(cols):
            ids = swizzle_func(get_1d_ids(i, j, rows, cols, row_major))
            res += f'{ids}|'
        res += '\n'
    print(res)


if __name__ == "__main__":
    swizzle = Swizzle(2, 2, 3)
    print_result(4, 8, swizzle)
