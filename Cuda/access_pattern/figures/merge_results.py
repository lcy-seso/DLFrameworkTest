#!/usr/bin/env python3


def merge_result(prefix):
    with open(prefix+".tsv", "r") as f1, \
            open(prefix+"_vectorized_access.tsv", "r") as f2,\
                open(prefix+"_merged.tsv", "w") as fout:
        header = f1.readline()
        f2.readline()
        fout.write("Method\t%s" % (header))
        for line in f1:
            fout.write("non-vectorized\t%s" % (line))
            fout.write("vectorized\t%s" % (f2.readline()))


if __name__ == "__main__":
    merge_result("30000_1024")
    # merge_result("60000_4096")
