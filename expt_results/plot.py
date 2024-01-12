import matplotlib.pyplot as plt
import numpy as np
import csv

def parse_results(fname):
    pipeline_results = {}
    pipeline_opt_results = {}
    with open(fname) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            row = list(filter(None, row))
            if len(row) == 6:
                size, send_size, avg_time, min_time, avg_algbw, max_algbw = row
                if size == "size(B)":
                    continue
                size, send_size, avg_time, min_time, avg_algbw, max_algbw = int(size), int(send_size), float(avg_time), float(min_time), float(avg_algbw), float(max_algbw)
                pipeline_results[size, send_size] = {
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "avg_algbw": avg_algbw,
                    "max_algbw": max_algbw
                }
                if size not in pipeline_opt_results or pipeline_opt_results[size]["avg_time"] > avg_time:
                    pipeline_opt_results[size] = {
                        "send_size": send_size,
                        "avg_time": avg_time,
                        "min_time": min_time,
                        "avg_algbw": avg_algbw,
                        "max_algbw": max_algbw
                    }
    return pipeline_opt_results


def parse_size(size_str):
    size_str = size_str.strip()
    if 'KiB' in size_str:
        return int(float(size_str.replace('KiB', '')) * 1024)
    elif 'MiB' in size_str:
        return int(float(size_str.replace('MiB', '')) * 1024 * 1024)
    elif 'GiB' in size_str:
        return int(float(size_str.replace('GiB', '')) * 1024 * 1024 * 1024)
    else:
        assert False, size_str


def parse_baseline_results(fname):
    mscclpp_opt_results = {}
    nccl_opt_results = {}
    with open(fname) as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            row = list(filter(None, row))
            if len(row) == 8:
                size, mscclpp_time, mscclpp_algbw, _, nccl_time, nccl_algbw, _, _ = row
                if "Size" in size:
                    continue
                size, mscclpp_time, mscclpp_algbw, nccl_time, nccl_algbw = parse_size(size), float(mscclpp_time), float(mscclpp_algbw), float(nccl_time), float(nccl_algbw)
                if size not in mscclpp_opt_results or mscclpp_opt_results[size]["avg_time"] > mscclpp_time:
                    mscclpp_opt_results[size] = {
                        "avg_time": mscclpp_time,
                        "avg_algbw": mscclpp_algbw
                    }
                if size not in nccl_opt_results or nccl_opt_results[size]["avg_time"] > nccl_time:
                    nccl_opt_results[size] = {
                        "avg_time": nccl_time,
                        "avg_algbw": nccl_algbw
                    }
    return mscclpp_opt_results, nccl_opt_results


if __name__ == "__main__":
    plt.figure(figsize=(10, 6), dpi=300)
    plt.xlabel("size (bytes)")
    plt.ylabel("algbw (GB/s)")
    plt.grid(True)
    plt.xscale("log")

    mscclpp_opt_results, nccl_opt_results = parse_baseline_results("2_node_baseline_allgather.txt")

    # plot mscclpp
    sizes = []
    algbws = []
    for size, data in sorted(mscclpp_opt_results.items()):
        sizes.append(size)
        algbws.append(data["avg_algbw"])
    plt.plot(sizes, algbws, label="mscclpp")

    # plot nccl
    sizes = []
    algbws = []
    for size, data in sorted(nccl_opt_results.items()):
        sizes.append(size)
        algbws.append(data["avg_algbw"])
    plt.plot(sizes, algbws, label="nccl")

    # plot pipeline
    pipeline_opt_results = parse_results("2_node_allgather_results.txt")
    sizes = []
    algbws = []
    for size, data in sorted(pipeline_opt_results.items()):
        sizes.append(size)
        algbws.append(data["avg_algbw"])
    plt.plot(sizes, algbws, color="red", label="pipeline")

    plt.legend()
    plt.savefig("plot.png")
