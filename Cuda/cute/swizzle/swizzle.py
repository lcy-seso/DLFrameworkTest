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


if __name__ == "__main__":
    swizzle = Swizzle(2, 2, 3)

    res = ''
    for i in range(0, 64):
        res += (str(swizzle(i)) + ', ')
        if not (i + 1) % 8:
            res += '\n'
    print(res)
