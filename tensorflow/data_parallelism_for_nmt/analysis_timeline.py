#!/usr/bin/env python
#coding=utf-8
from __future__ import division

import os
import sys
import json
from collections import defaultdict
from collections import OrderedDict
import pdb

import tensorflow as tf
from seq2seq_model import Seq2SeqModel


def get_filename(path):
    return os.path.split(os.path.splitext(path)[0])[-1]


def dump_op_info(num_gpus, log_file):
    model = Seq2SeqModel(num_gpus, hparams)

    with open(log_file, "w") as flog:
        for op in tf.get_default_graph().get_operations():
            if op.device:
                flog.write("%s\t%s\t%s\n" % (op.name, op.device, op.type))


def load_op_info(log_file):
    op_info_dict = OrderedDict()

    with open(log_file, "r") as fin:
        for line in fin:
            op, device, op_type = line.strip().split("\t")
            op_info_dict[op] = device.lower()
    return op_info_dict


def sort_event_by_ts(log_file):
    events = defaultdict(list)

    with open(log_file, "r") as flog:
        info = json.load(flog)["traceEvents"]

    for event in info:
        if "ts" not in event: continue
        if "args" not in event: continue

        if "name" not in event["args"]: continue

        event_name = event["args"]["name"]
        name_parts = event_name.split("/")

        if len(name_parts) < 2: continue
        events[event["ts"]] = [event_name, event["dur"]]

    sorted_event = sorted(
        events.iteritems(), key=lambda x: int(x[0]), reverse=False)
    for ts, info in sorted_event:
        print("%s\t%s\t%s" % (ts, info[0], info[1]))


def analyse_ts(log_file, tower_num, save_dir):
    key_events = {
        "fw_start": "encoder/embedding_lookup",
        "fw_end": "SparseSoftmaxCrossEntropyWithLogits",
        "bw_start": "SparseSoftmaxCrossEntropyWithLogits_grad/mul",
        "bw_end": "gradients/AddN_24",
        "merge_grad_start": "merge_gradients_18/AddN",
        "merge_grad_end": "merge_gradients/Mul",
    }

    save_path = os.path.join(save_dir, get_filename(log_file) + "_ts.txt")
    op_runtime_info = [{
        "fw_start": None,
        "fw_end": None,
        "bw_start": None,
        "bw_end": None
    } for _ in range(tower_num)]

    merge_gradients = [None, None]
    opts = []

    with open(log_file, "r") as flog:
        info = json.load(flog)["traceEvents"]

    for event in info:
        if "ts" not in event: continue
        if "args" not in event: continue
        if "name" not in event["args"]: continue

        event_name = event["args"]["name"]
        name_parts = event_name.split("/")
        if len(name_parts) < 2: continue

        gpu_number = None
        if "tower_" in name_parts[1]:
            gpu_number = int(name_parts[1].replace("tower_", ""))

        last2 = name_parts[-2] + "/" + name_parts[-1]
        last1 = name_parts[-1]

        if last2 == key_events["bw_start"]:
            assert (gpu_number is not None)

            if op_runtime_info[gpu_number]["bw_start"] is None:
                print("gpu %d backward starts\t%s" % (gpu_number, event_name))
                op_runtime_info[gpu_number]["bw_start"] = float(event["ts"])

        elif last2 == key_events["bw_end"]:
            assert (gpu_number is not None)

            if op_runtime_info[gpu_number]["bw_end"] is None:
                print("gpu %d backward ends\t%s" % (gpu_number, event_name))
                op_runtime_info[gpu_number]["bw_end"] = (
                    float(event["ts"]) + float(event["dur"]))

        elif last2 == key_events["fw_start"] and (
                event_name.startswith("v/") or event_name.startswith("v_")):
            assert (gpu_number is not None)

            if op_runtime_info[gpu_number]["fw_start"] is None:
                print("gpu %d forward starts\t%s" % (gpu_number, event_name))
                op_runtime_info[gpu_number]["fw_start"] = float(event["ts"])

        elif last1 == key_events["fw_end"]:
            assert (gpu_number is not None)

            if op_runtime_info[gpu_number]["fw_end"] is None:
                print("gpu %d forward ends\t%s" % (gpu_number, event_name))
                op_runtime_info[gpu_number]["fw_end"] = (
                    float(event["ts"]) + float(event["dur"]))

        elif last2 == key_events["merge_grad_start"]:
            if merge_gradients[0] is None:
                merge_gradients[0] = float(event["ts"])

        elif last2 == key_events["merge_grad_end"]:
            if merge_gradients[1] is None:
                merge_gradients[1] = float(event["ts"]) + float(event["dur"])

        elif "Adam/update_v" in event_name:
            opts.append(float(event["ts"]))

    assert len(opts)
    opts.sort(reverse=False)

    res = "|forward (ms)|"
    fw_start = None
    bw_end = None
    total = 0.
    for i in range(tower_num):
        t = (op_runtime_info[i]["fw_end"] - op_runtime_info[i]["fw_start"]
             ) / 1000.
        res += ("%.3f|" % (t))
        total += t

        if fw_start is None:
            fw_start = op_runtime_info[i]["fw_start"]
        else:
            fw_start = (op_runtime_info[i]["fw_start"]
                        if op_runtime_info[i]["fw_start"] < fw_start else
                        fw_start)
    res += "%.3f|" % (total / tower_num)
    print(res)

    total = 0.
    res = "|backward (ms)|"
    for i in range(tower_num):
        t = (op_runtime_info[i]["bw_end"] - op_runtime_info[i]["bw_start"]
             ) / 1000.
        res += ("%.3f|" % (t))
        total += t

        if bw_end is None:
            bw_end = op_runtime_info[i]["bw_end"]
        else:
            bw_end = (op_runtime_info[i]["bw_end"]
                      if op_runtime_info[i]["bw_end"] > bw_end else bw_end)
    res += "%.3f|" % (total / tower_num)
    print(res)

    if tower_num > 1:
        merge_grad = (merge_gradients[1] - merge_gradients[0]) / 1000.
    else:
        merge_grad = 0.

    update = (opts[-1] - opts[0]) / 1000.
    computate = (bw_end - fw_start) / 1000.
    total_time = merge_grad + update + computate

    print(("|computation (ms)|merge gradients (ms) |update (ms)|total(ms)|\n"
           "|:--|:--|:--|:--|"))
    print("|%.3f(%.4f)|%.3f(%.4f)|%.3f(%.4f)|%.3f(1.0)|" %
          (computate, computate / total_time, merge_grad,
           merge_grad / total_time, update, update / total_time, total_time))


def analysis_timeline(log_file, op_info_file, tower_num):
    op_runtime_info = {}
    op_info_dict = load_op_info(op_info_file)

    with open(log_file, "r") as flog:
        info = json.load(flog)["traceEvents"]

    for event in info:
        if "cat" not in event: continue
        if event["cat"] != "Op": continue

        event_name = event["args"]["name"]
        if event_name not in op_info_dict: continue

        name_parts = event_name.split("/")

        if len(name_parts) <= 2:
            continue

        # this order (variable scope's name followed by name scope's name)
        # is determined by the model defination
        if "gradients" in event_name:
            if "merge_gradients" in event_name:
                op_name = event_name
            else:
                op_name = "gradients/" + "/".join(name_parts[5:])
        elif "tower_" in event_name:
            op_name = "/".join(name_parts[2:])
        else:
            op_name = event_name

        dur = int(event["dur"]) if "dur" in event else 0
        start_wall = int(event["ts"])

        gpu_number = None
        device = None
        if "tower_" in event_name:
            gpu_number = int(name_parts[1].replace("tower_", ""))
            device = "/gpu:%d" % (gpu_number)
        else:
            device = op_info_dict[event_name]
            gpu_number = int(device.split(":")[-1])

        if op_name not in op_runtime_info:
            op_runtime_info[op_name] = [{
                "ts": 0,
                "dur": 0
            } for _ in range(tower_num)]

        op_runtime_info[op_name][gpu_number]["dur"] += dur
        op_runtime_info[op_name][gpu_number]["ts"] += start_wall

    with open("nmt_%02d_cards.txt" % (tower_num), "w") as fout:
        fout.write("op name")
        for i in range(tower_num):
            fout.write("\t/gpu:%d(ts)\t/gpu:%d(dur)" % (i, i))
        fout.write("\n")

        for op in op_info_dict:
            if "gradients" in op:
                if "merge_gradients" in op:
                    name = op
                else:
                    splits = op.split("/")
                    name = "gradients/" + "/".join(splits[5:])
            elif "tower_" in op:
                splits = op.split("/")
                name = "/".join(splits[2:])
            else:
                continue

            if name not in op_runtime_info: continue

            fout.write("%s" % (name))
            info = op_runtime_info[name]

            min_ts = info[0]["ts"]
            for i in range(1, tower_num):
                min_ts = info[i]["ts"] if info[i]["ts"] < min_ts else min_ts

            for i in range(tower_num):
                fout.write("\t%d\t%d" %
                           (info[i]["ts"] - min_ts, info[i]["dur"]))
            fout.write("\n")


def profile_host(log_filename, save_dir="timeline_info"):
    device_num = int(os.path.split(log_filename)[-1].split("_")[0])

    info = defaultdict(list)
    pid_info = {}

    with open(log_filename, "r") as flog:
        events = json.load(flog)["traceEvents"]

        for event in events:
            if "name" not in event: continue
            if "args" not in event: continue
            if "name" not in event["args"]: continue

            name = event["args"]["name"]
            if event["name"] == "process_name":
                if ("Op scheduling threads" in name or
                        "Op execution threads" in name):
                    splits = name.split("/")
                    device_name = splits[-1]

                    if "scheduling" in splits[0]: device_name += "_scheduling"
                    elif "execution" in splits[0]: device_name += "_execution"

                    pid_info[str(event["pid"])] = device_name
                    print("%d\t%s" % (event["pid"], device_name))
                    continue

            if "ts" not in event: continue
            if str(event["pid"]) not in pid_info: continue

            info[name].append({
                "ts": event["ts"] / 1000.,
                "pid": str(event["pid"]),
                "dur": event["dur"],
                "name": event["name"],
            })

    formated_info = {}
    for k, v in info.iteritems():
        for item in v:
            device_name = pid_info[item["pid"]]
            formated_info[k] = [item["ts"], device_name, item["dur"]]

    sorted_info = sorted(
        formated_info.iteritems(), key=lambda x: x[1][0], reverse=False)

    with open(os.path.join(save_dir, "ops_%d_cards.txt" % (device_num)),
              "w") as fout:
        ts_base = sorted_info[0][1][0]
        for v in sorted_info:
            fout.write("%s\t%.3f\t%s\t%d\n" % (v[0], v[1][0] - ts_base,
                                               v[1][1], v[1][2]))


if __name__ == "__main__":
    # analyse_ts(sys.argv[1], int(sys.argv[2]), save_dir="log_files")
    #     "timeline_log/2_cards.json", tower_num=2, save_dir="profiling_log")

    log_dir = "timeline_info/json"
    for f in os.listdir(log_dir):
        print("processing %s" % (f))
        file_path = os.path.join(log_dir, f)
        profile_host(file_path)
