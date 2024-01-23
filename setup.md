# Setup
- A100 machines:
```
a100-srg-0 10.0.0.5
10.0.0.4
```
- Clone from `git@github.com:liangyuRain/mscclpp-public.git` using SSH and set the ssh key by
```
git config --add --local core.sshCommand 'ssh -i /home/azureuser/liangyu/.ssh/id_ed25519'
```
- docker
```shell
docker pull ghcr.io/microsoft/mscclpp/mscclpp:base-dev-rocm6.0
docker run --security-opt seccomp=unconfined --group-add video -it --ulimit memlock=-1:-1 --privileged --net=host --ipc=host -p 81:5001 -d --name <container_name> --entrypoint bash <image_name>
```
- Install miniconda:
```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
```
```shell
conda create --name mscclpp python=3.9
conda activate mscclpp
```

# Build msccl++
- Remove, create, and move to `build` folder under `mscclpp-public`.
- Run
```shell
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
make pylib-copy
```
- Possibly also run `pip install .` under `mscclpp-public` folder.
- Run `make pylib-copy` if encouter the following error:
```python
_______________________ ERROR collecting test_mscclpp.py _______________________
ImportError while importing test module '/root/mscclpp/python/test/test_mscclpp.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/root/miniconda3/envs/mscclpp/lib/python3.8/importlib/__init__.py:127: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
test_mscclpp.py:12: in <module>
    from mscclpp import (
../mscclpp/__init__.py:6: in <module>
    from ._mscclpp import (
E   ModuleNotFoundError: No module named 'mscclpp._mscclpp'
```

# Run Pipeline Test
- `cd` to `mscclpp-public/python` and run
```shell
mpirun -np 8 pytest -s ./test/test_pipeline.py
```
- Use `-k <test_prefix>` flag to run tests with the specified prefix only.

# Run Pipeline Expt
- `cd` to `mscclpp-public/python` and run
```shell
mpirun -np 8 python -m test.pipeline_expt
```

# Run mscclpp tests
- Run
```shell
mpirun -np 2 /root/mscclpp-public/build/test/mscclpp-test/allgather_test_perf -b 192M -e 3G -f 2 -n 100 -w 10 -c 0 -k 4
```
- If no ib device is avaiable, modify `mscclpp-public/test/mscclpp-test/common.cc`. Change
```c++
const mscclpp::TransportFlags allTransports = mscclpp::Transport::CudaIpc | IBs[args_.gpuNum];
```
to
```c++
const mscclpp::TransportFlags allTransports = mscclpp::Transport::CudaIpc;
```
**There are three positions in the file where such flags exist.**

# Run mscclpp benchmark
- Make sure `nccl` is installed: `sudo apt-get install libnccl2`
- Run
```shell
mpirun -np 8 python allreduce_bench.py
```
- To perform allgather/reduce-scatter benchmark, one should:
    - `allreduce.cu`: line 672 `allreduce4`, comment out unwanted collective and `devceSyncer`.
    - `nccl_op.py`: line 19, modify nccl's collective, allgather does not need `nccl.NCCL_SUM`. Allgather and reduce-scatter need to adjust the `size` to `size // 16`. Sometimes there is tcp connection timeout error, then try adding `-u` flag to python.
    - `allreduce_bench.py`: line 175/179, remove `check_correctness` and give `PASS` to `mscclpp_check` and `nccl_check` directly.

# Setup multinode mpirun
- Inside both docker containers, open a port by running:
```shell
mkdir /run/sshd
/usr/sbin/sshd -p 20000
```
- Add node 1's ssh key into both nodes' `authorized_keys`.
- Edit `.ssh/config` at node 1 to direct all ssh to port 20000:
```
Host *
Port 20000
```
- If ssh still asks for password, try another port.
- Create hostfile `hosts.txt`
```
10.0.0.5 slots=8
10.0.0.4 slots=8
```
- `mpirun` should now specify `--hostfile /root/hosts.txt`.
- No conda is needed: `pip install -r requirements_cu12.txt`
- Run command
```shell
/usr/local/mpi/bin/mpirun -allow-run-as-root -np 16 --bind-to numa -hostfile /root/hosts.txt -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp-public/build:$LD_LIBRARY_PATH -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x NCCL_SOCKET_IFNAME=eth0 -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/topo.xml -x NCCL_NET_PLUGIN=none -x NCCL_IB_DISABLE=0 -x NCCL_MIN_NCHANNELS=32 -x NCCL_DEBUG=WARN -x NCCL_P2P_DISABLE=0 -x NCCL_SHM_DISABLE=0 -x MSCCLPP_HOME=/root/mscclpp-public -np 16 -npernode 8 python3 /root/mscclpp-public/python/mscclpp_benchmark/allreduce_bench.py
```
```shell
/usr/local/mpi/bin/mpirun -allow-run-as-root -np 16 --bind-to numa -hostfile /root/hosts.txt -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp-public/build:$LD_LIBRARY_PATH -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x NCCL_SOCKET_IFNAME=eth0 -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/topo.xml -x NCCL_NET_PLUGIN=none -x NCCL_IB_DISABLE=0 -x NCCL_MIN_NCHANNELS=32 -x NCCL_DEBUG=WARN -x NCCL_P2P_DISABLE=0 -x NCCL_SHM_DISABLE=0 -x MSCCLPP_HOME=/root/mscclpp-public -np 16 -npernode 8 python3 /root/mscclpp-public/python/test/pipeline_expt_2_node.py
```
```shell
ssh 10.0.0.4 ; stdbuf --output=L python ~/lock.py ssh -t 10.0.0.5 docker exec -it original-mscclpp /usr/local/mpi/bin/mpirun -allow-run-as-root -np 16 --bind-to numa -hostfile /root/hosts.txt -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp-public/build:$LD_LIBRARY_PATH -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x NCCL_SOCKET_IFNAME=eth0 -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/topo.xml -x NCCL_NET_PLUGIN=none -x NCCL_IB_DISABLE=0 -x NCCL_MIN_NCHANNELS=32 -x NCCL_DEBUG=WARN -x NCCL_P2P_DISABLE=0 -x NCCL_SHM_DISABLE=0 -x MSCCLPP_HOME=/root/mscclpp-public -np 16 -npernode 8 python3 /root/mscclpp-public/python/test/pipeline_expt_2_node.py 2>&1 | tee results.txt
```


# Notes
- Error `ibv_create_cq(cqe=4096) failed: Cannot allocate memory` is caused by not setting `max locked memory` to `unlimited`. One can check by running `ulimit -a`. The solution is to add `--ulimit memlock=-1:-1` when `docker run`. There seems to be a discrepancy between host and docker container in `ulimit -a` by default.
- Run `ib_send_bw -d mlx5_0 -i 1 -a -R --report_gbits` on both server and client sides to check IB bandwidth. (never successful yet on lambda)
    - Saeed's command: run `ib_write_bw -d mlx5_0` on server node and `ib_write_bw 10.0.0.5 -d mlx5_0` on client node with `10.0.0.5` be the address of server node.
- If report compile error for `#include <mscclpp/sm_channel_device.hpp>` not found, check if the `include_dir` in `KernelBuilder` from `mscclpp-public/python/mscclpp/utils.py` is as following:
```python
include_dir = os.path.join(self._current_file_dir, "../../include")
```
- Currently, running official `xxxxxx_test_perf` has the following error:
```shell
  what():  Test CUDA failure: /root/mscclpp-public/test/mscclpp-test/common.cc:209 'too many resources requested for launch'
```
Originally, running `test_pipeline` also has this error. However, after fixing the argument passed as `Plist` instead of pointer to reduce the size of arguments to the kernel, `test_pipeline` can be run properly.
- For performance numbers, see `single_node_results.txt`.
- With proxy channels, it is observed that when `k` or `ninstance` is too large (AR: k>4, AG: k>8), both allreduce and allgather can hang. Allgather generally hangs at larger `k`, probably because it requires less number of channels.
- For proxy channels, increasing the number of parallel channels does not improve performance; however, for sm channels, it does.
- To implement `k` sm channels per proxy channel, we can let `k` sm channels use the same scratch buffer but writing to different offsets to simplify logic.
- Ensure if `connection_types[b]=x` at `a`, then `connection_types[a]=x` at `b`.
- Proxy channel (within a node at least) requires higher `nelem_per_send` (1MB?); otherwise, performance is extremely poor.
- Number of proxy channels: [issue](https://github.com/microsoft/mscclpp/issues/242)
- Be careful about mlx nics. Some may be using ethernet instead of IB and need to be avoided by setting `MSCCLPP_HCA_DEVICES`.
- `cp.cuda.Device(MPI.COMM_WORLD.rank % 8).use()` is necessary; otherwise, all processes may use the same GPU.
- Proxy must be reconstructed `proxy_service = ProxyService()` for every data size; otherwise, program may hang due to proxy service being reused too many times.
- Right now, the best 2x A100 node performance is achieved by manually edge-split the topology into local rings and one-to-one inter-node connections. IB bw is set to 20GB/s with NVLink set to 300GB/s, forcing code to use IB as less as possible. The optimal tree for large data sizes appears to be is to generate symmetric k=1 tree and then ninstance=6, achieving 250GB/s algbw at 3GB.
- Flushing proxy channel more frequently seems to improve (allgather) performance. Turns out it is beneficial to do `putWithSignalAndFlush` at allgather root.
- `CUDA error code=701(b'CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES')` or `CUDA error code=701(b'CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES')` can be caused by compiling two different kernels with the same name in a single run, or feeding the wrong kernel name into `KernelBuilder`.

# MSCCL

Compile instructions:
```shell
git clone https://github.com/microsoft/msccl.git
cd msccl/
make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
cd ..
git clone https://github.com/nvidia/nccl-tests.git
cd nccl-tests/
make MPI=1 MPI_HOME=/usr/mpi/gcc/openmpi-4.1.5a1/ NCCL_HOME=/home/azureuser/liangyu/msccl/build/ -j
cd ..
```
Run command:
```shell
mpirun \
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
-x NCCL_ALGO=MSCCL,TREE \
-x NCCL_TOPO_FILE=/home/azureuser/liangyu/topo.xml \
-x MSCCL_XML_FILES=/home/azureuser/liangyu/mscclpp-public/pipeline_msccl_xml/msccl_allgather_k1_inst1_NVLINK300_IB25.xml \
/home/azureuser/liangyu/nccl-tests/build/all_gather_perf -b 256 -e 10M -f 2 -g 1 -z 0 -n 100 -w 10 -c 1 -a 2
```

# RCCL

```shell
git clone https://github.com/ROCmSoftwarePlatform/rccl.git
cd rccl
mkdir build
cd build
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ ..
make -j
```
```shell
git clone https://github.com/ROCmSoftwarePlatform/rccl-tests.git
cd rccl-tests
make MPI=1 MPI_HOME=/usr/mpi/gcc/openmpi-4.1.5a1 HIP_HOME=/opt/rocm/bin/hipcc RCCL_HOME=/home/amdautomation/liangyu/rccl/build
```
```shell
mpirun --allow-run-as-root -tag-output -map-by ppr:8:node -bind-to numa -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 -x LD_PRELOAD=/rccl/build/librccl.so:$LD_PRELOAD -x NCCL_DEBUG=WARN /rccl-tests/build/all_gather_perf -b 1 -e 16G -f 2 -g 1 -c 1 -n 1000 -w 20 -G 1
```