def test_xor(bits, base, shift):
    res = '||'
    for i in range(2**bits):
        res += f'{i:03b}|'
    res += '\n|:--:|'
    for i in range(2**bits):
        res += ':--:|'
    res += '\n'

    for i in range(2**bits):
        res += f'|**{i:03b}**|'
        for j in range(2**bits):
            xor = i ^ j
            res += f'{xor:03b} **({xor - i})**|'
        res += '\n'

    print(res)


if __name__ == "__main__":
    print('swizzle<3, 3, 3>')
    test_xor(3, 3, 3)

    print('swizzle<2, 3, 3>')
    test_xor(2, 3, 3)
