def swizzle_func_4x8():
    threadIdx_xs = list(range(128))
    # print(threadIdx_xs)

    # M = 32
    # N = 64
    # lda = N  # row-major

    swizzle = {}
    for threadIdx_x in threadIdx_xs:
        lane = threadIdx_x % 32  # index in a warp

        c = lane % 8
        s = int(lane / 8)

        # gmem_offset = c + s * lda

        smem_row = (c & 1) | ((c >> 1 & 2))
        bank = ((c << 1) & 4) | s ^ smem_row

        org = f'[{s}, {c}]'

        if org not in swizzle:
            swizzle[org] = f'[{smem_row},{bank}]'

    output_str = ''
    for i, (orig, swizzled) in enumerate(swizzle.items()):
        # print(f'{i}, {orig}, {swizzled}')
        output_str += f'\t{orig}->{swizzled}'

        if (i + 1) % 8 == 0 and i:
            print(output_str.lstrip('\t'))
            output_str = ''


swizzle_func_4x8()
