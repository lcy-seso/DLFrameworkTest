warp_arrangements = [[1, 4], [4, 1], [2, 2]]

thread_arrangements = [
    [1, 32],
    [32, 1],
    [2, 16],
    [16, 2],
    [4, 8],
    [8, 4],
]


def compute_configs(warp_arrangements, thread_arrangements):
    count = 0
    header = """|NO.|Plan|Thread Arrangement|Thread Layout|Tile Shape for a CTA<br>(1024 numbers in total)|
|:--:|:--|:--:|:--:|:--:|"""

    print(header)
    for i, wa in enumerate(warp_arrangements):
        for j, ta in enumerate(thread_arrangements):
            out_str = ""
            wa1, wa2 = wa
            ta1, ta2 = ta

            thd1 = wa1 * ta1
            thd2 = wa2 * ta2
            out_str += f"|{count+1}|W{i+1}-T{j+1}|${thd1} \\times {thd2}$|"

            cute_layout = f"`({thd1},{thd2}):({thd2},1)`|"
            tile_shape = f"$[{thd1}, {thd2*8}]$|"

            out_str += cute_layout
            out_str += tile_shape

            count += 1
            print(out_str)


if __name__ == "__main__":
    compute_configs(warp_arrangements, thread_arrangements)
