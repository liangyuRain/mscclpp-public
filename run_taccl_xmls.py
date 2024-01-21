import os

if __name__ == "__main__":
    folder = "/home/azureuser/liangyu/mscclpp-public/taccl_msccl_xml"
    assert os.path.exists(folder)
    assert os.path.exists(os.path.join(folder, "taccl_results"))
    file_list = sorted(os.listdir(folder))
    for idx, fname in enumerate(file_list):
        print(f"\nrunning {idx} / {len(file_list)}\n", flush=True)
        if not fname.endswith(".xml"):
            continue
        xml_file = os.path.join(folder, fname)
        cmd = (
f"""mpirun \
--mca btl_tcp_if_include eth0 \
--mca pml ob1 -mca btl ^openib \
-np 16 -npernode 8 \
-H 10.0.0.4:8,10.0.0.5:8 \
-x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
-x LD_LIBRARY_PATH=/home/azureuser/liangyu/msccl/build/lib/ \
-x NCCL_IB_PCI_RELAXED_ORDERING=1 \
-x NCCL_SOCKET_IFNAME=eth0 \
-x CUDA_DEVICE_ORDER=PCI_BUS_ID \
-x NCCL_NET_GDR_LEVEL=5 \
-x NCCL_DEBUG=WARN \
-x NCCL_DEBUG_SUBSYS=INIT \
-x NCCL_PXN_DISABLE=1 \
-x NCCL_ALGO=MSCCL,TREE,RING \
-x NCCL_TOPO_FILE=/home/azureuser/liangyu/topo.xml \
-x MSCCL_XML_FILES={xml_file} \
/home/azureuser/liangyu/nccl-tests/build/all_gather_perf -b 256 -e 10G -f 2 -g 1 -z 0 -n 100 -w 100 -c 0 -a 2"""
        )
        res_file = os.path.join(folder, "taccl_results", fname + ".txt")
        os.system(f"stdbuf --output=L {cmd} 2>&1 | tee {res_file}")
