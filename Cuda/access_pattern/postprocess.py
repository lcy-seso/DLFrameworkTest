#!/usr/bin/env python3
import pdb
import sys


def parse1(log_file):
    info = {}
    with open(log_file, 'r') as flog:
        for line in flog:
            if 'void CopyTest1<float, (int)' in line:
                splits = line.strip().split('void CopyTest1<float, (int)')[1]
                splits = splits.split(',')
                grid_size = splits[0]
                splits = splits[1].split('>')[0]
                block_size = splits.replace(' (int)', '')
                key = grid_size + '#' + block_size

                for i in range(4):
                    flog.readline()
                line = flog.readline().strip()
                util = float(line.split(' ')[-1])  # bandwidth utilication

                flog.readline()
                line = flog.readline()
                scale = 1.
                if 'msecond' in line:  # milisecond
                    scale = 1.
                elif 'usecond' in line:  # microsecond to milisecond
                    scale = 1000.
                else:
                    raise KeyError('Unknown unit.')

                elapsed = float(line.strip().split(' ')[-1]) / scale

                line = flog.readline().strip()
                sm = float(line.split(' ')[-1])

                if key in info:
                    info[key]['bandwidth'].append(util)
                    info[key]['elapsed'].append(elapsed)
                    info[key]['sm'].append(sm)
                else:
                    info[key] = {}
                    info[key]['bandwidth'] = [util]
                    info[key]['elapsed'] = [elapsed]
                    info[key]['sm'] = [sm]

    print(("Launch Config\tRow Per TB"
           "\tElapsed Time(ms)\tBandwidth Utilization(%)\t"
           "SM Throughout(%)"))
    for key in info:
        blocks, threads = key.split('#')
        row_per_tb = int(60000 / int(blocks))
        n = len(info[key]['bandwidth'])
        bandwidth = sum(info[key]['bandwidth']) / n
        n = len(info[key]['sm'])
        sm = sum(info[key]['sm']) / n
        n = len(info[key]['elapsed'])
        elapsed = sum(info[key]['elapsed']) / n

        config = f"[{blocks}, {threads}]"
        print('%s\t%d\t%.4f\t%.4f\t%.4f' % (config, row_per_tb, elapsed,
                                            bandwidth, sm))


def parse2(log_file):
    info = {}
    with open(log_file, 'r') as flog:
        for line in flog:
            if 'CopyTest2<float, DirectLoad<float, float>' in line:
                separator = (
                    "void CopyTest2<float, "
                    "DirectLoad<float, float>, DirectStore<float, float>, (int)"
                )
                splits = line.strip().split(separator)[1]
                splits = splits.split(',')
                grid_size = splits[0]
                splits = splits[1].split('>')[0]
                block_size = splits.replace(' (int)', '')
                key = grid_size + '#' + block_size

                for i in range(4):
                    flog.readline()
                line = flog.readline().strip()
                util = float(line.split(' ')[-1])  # bandwidth utilication

                flog.readline()
                line = flog.readline()
                scale = 1.
                if 'msecond' in line:  # milisecond
                    scale = 1.
                elif 'usecond' in line:  # microsecond to milisecond
                    scale = 1000.
                else:
                    raise KeyError('Unknown unit.')

                elapsed = float(line.strip().split(' ')[-1]) / scale

                line = flog.readline().strip()
                sm = float(line.split(' ')[-1])

                if key in info:
                    info[key]['bandwidth'].append(util)
                    info[key]['elapsed'].append(elapsed)
                    info[key]['sm'].append(sm)
                else:
                    info[key] = {}
                    info[key]['bandwidth'] = [util]
                    info[key]['elapsed'] = [elapsed]
                    info[key]['sm'] = [sm]

    print(("Launch Config\tRow Per TB"
           "\tElapsed Time(ms)\tBandwidth Utilization(%)\t"
           "SM Throughout(%)"))
    for key in info:
        blocks, threads = key.split('#')
        row_per_tb = int(60000 / int(blocks))
        n = len(info[key]['bandwidth'])
        bandwidth = sum(info[key]['bandwidth']) / n
        n = len(info[key]['sm'])
        sm = sum(info[key]['sm']) / n
        n = len(info[key]['elapsed'])
        elapsed = sum(info[key]['elapsed']) / n
        config = f"[{blocks}, {threads}]"
        print('%s\t%d\t%.4f\t%.4f\t%.4f' % (config, row_per_tb, elapsed,
                                            bandwidth, sm))


if __name__ == '__main__':
    # print(sys.argv[1])
    parse1(sys.argv[1])
    # parse2(sys.argv[1])
