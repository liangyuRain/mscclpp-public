import os

if __name__ == "__main__":
    folder = "/home/amdautomation/liangyu/mscclpp-public/amd_pipeline_msccl_xml"
    assert os.path.exists(folder)
    assert os.path.exists(os.path.join(folder, "amd_pipeline_results"))
    file_list = sorted(os.listdir(folder))

    rm_cmd = "rm /home/amdautomation/liangyu/rccl_run_schedule/*"
    os.system(rm_cmd)
    os.system("ssh 10.8 " + rm_cmd)
    for idx, fname in enumerate(file_list):
        for buff_size in [2 ** n * 1024 for n in range(6, 14)]:
            print(f"\nrunning {idx} / {len(file_list)}, fname={fname}, buffsize={buff_size}\n", flush=True)
            if not fname.endswith(".xml") or "half" in fname:
                continue
            if "allgather" in fname:
                exe = "all_gather_perf"
            elif "allreduce" in fname:
                exe = "all_reduce_perf"
            elif "reduceScatter" in fname:
                exe = "reduce_scatter_perf"
            else:
                assert False
            xml_file = os.path.join(folder, fname)
            os.system(f"cp {xml_file} /home/amdautomation/liangyu/rccl_run_schedule/")
            os.system(f"ssh 10.8 cp {xml_file} /home/amdautomation/liangyu/rccl_run_schedule/")
            end_size = "10G" if buff_size >= 2 ** 20 else "1G"
            cmd = (
f"""mpirun --allow-run-as-root \
-hostfile ~/hostfile -map-by ppr:16:node \
--bind-to numa -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 \
-x PATH -x LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH -x NCCL_SOCKET_IFNAME=eth0 \
-x LD_PRELOAD=/home/amdautomation/liangyu/rccl/build/librccl.so:$LD_PRELOAD \
-x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=INIT,GRAPH -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
-x NCCL_MIN_NCHANNELS=32 -x NCCL_IB_PCI_RELAXED_ORDERING=1 \
-x NCCL_NET_GDR_LEVEL=3 -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_IBEXT_DISABLE=1 \
-x NCCL_PROTO=Simple \
-x NCCL_BUFFSIZE={buff_size} \
-x RCCL_MSCCL_FORCE_ENABLE=1 \
-x MSCCL_ALGO_DIR=/home/amdautomation/liangyu/rccl_run_schedule \
/home/amdautomation/liangyu/rccl-tests/build/{exe} -b 256 -e {end_size} -f 2 -g 1 -z 0 -n 50 -w 50 -c 1 -a 2"""
            )
            print(cmd)
            res_file = os.path.join(folder, "amd_pipeline_results", fname + f"buff{buff_size}.txt")
            os.system(f"stdbuf --output=L {cmd} 2>&1 | tee {res_file}")
            os.system(rm_cmd)
            os.system("ssh 10.8 " + rm_cmd)
