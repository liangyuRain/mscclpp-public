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
- Machine has cuda 12.2. Thus, the base docker image under `mscclpp-public/docker` has to be changed to `FROM nvidia/cuda:12.2.0-devel-ubuntu20.04`.
- Build an image called `liangyu-mscclpp`: 
```shell
docker build -t liangyu-mscclpp -f mscclpp-public/docker/base-x-cuda12.2.dockerfile .
```
- Create a container named `liangyu-mscclpp` using the image `liangyu-mscclpp` we just built:
```shell
docker run --gpus all -it --privileged --net=host --ipc=host -p 81:5001 -d --name liangyu-mscclpp --entrypoint bash liangyu-mscclpp
```
- Copy `mscclpp-public` into the container:
```shell
docker cp mscclpp-public/ liangyu-mscclpp:/root/
```
- Enter the container:
```shell
docker exec -it liangyu-mscclpp bash
```
- Install miniconda:
```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
```
- Create a conda environment:
```shell
conda create --name mscclpp python=3.8
conda activate mscclpp
```
- The `cuda-python==12.1.0` and `cupy-cuda12x` in `mscclpp-public/python/test/requirements_cu12.txt` cannot be found by conda. Remove them and run:
```shell
conda install --file mscclpp-public/python/requirements_cu12.txt -c conda-forge
```
- Because cuda version is 12.2, we install `cuda-python==12.2.0` instead:
```shell
pip install cuda-python==12.2.0
```
- Other necessary packages:
```shell
pip install cupy-cuda12x cmake==3.25.0
conda install networkx
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

# Notes
- Error `ibv_create_cq(cqe=4096) failed: Cannot allocate memory` is caused by not setting `max locked memory` to `unlimited`. One can check by running `ulimit -a`. The solution is to add `--ulimit memlock=-1:-1` when `docker run`. There seems to be a discrepancy between host and docker container in `ulimit -a` by default.
- Run `ib_send_bw -d mlx5_0 -i 1 -a -R --report_gbits` on both server and client sides to check IB bandwidth. (never successful yet on lambda)
- If report compile error for `#include <mscclpp/sm_channel_device.hpp>` not found, check if the `include_dir` in `KernelBuilder` from `mscclpp-public/python/mscclpp/utils.py` is as following:
```python
include_dir = os.path.join(self._current_file_dir, "../../include")
```
- Currently, running official `xxxxxx_test_perf` has the following error:
```shell
  what():  Test CUDA failure: /root/mscclpp-public/test/mscclpp-test/common.cc:209 'too many resources requested for launch'
```
Originally, running `test_pipeline` also has this error. However, after fixing the argument passed as `Plist` instead of pointer to reduce the size of arguments to the kernel, `test_pipeline` can be run properly.
- At this commit, if change `pipeline_expt` to take best result instead of averge, perfomance numbers for `pipeline_kernel.cu`:
```shell
############################################# Allgather #############################################
nranks=8
k=4, nelem_per_send=262144
check_iters=10, warmup_iters=10, iters=50
KERNEL_FILE=pipeline_kernel.cu

             size(B)        avg_time(us)        min_time(us)     avg_algbw(GB/s)     max_algbw(GB/s)
          1073741824             4376.92             3999.74              228.47              250.02

########################################### ReduceScatter ###########################################
nranks=8
k=8, nelem_per_send=262144, scratch_size=1048576
check_iters=10, warmup_iters=10, iters=50
KERNEL_FILE=pipeline_kernel.cu

             size(B)        avg_time(us)        min_time(us)     avg_algbw(GB/s)     max_algbw(GB/s)
          1073741824             5618.57             5124.10              177.98              195.16

############################################# Allreduce #############################################
nranks=8
k=4, nelem_per_send=32768, scratch_size=1048576
check_iters=10, warmup_iters=10, iters=50
KERNEL_FILE=pipeline_kernel.cu

             size(B)        avg_time(us)        min_time(us)     avg_algbw(GB/s)     max_algbw(GB/s)
          1073741824             8709.06             8655.87              114.82              115.53
```
For `pipeline_kernel_simplified.cu` (different `nelem_per_send`):
```shell
############################################# Allgather #############################################
nranks=8
k=4, nelem_per_send=262144
check_iters=10, warmup_iters=10, iters=50
KERNEL_FILE=pipeline_kernel_simplified.cu

             size(B)        avg_time(us)        min_time(us)     avg_algbw(GB/s)     max_algbw(GB/s)
          1073741824             4415.75             3999.74              226.46              250.02

########################################### ReduceScatter ###########################################
nranks=8
k=8, nelem_per_send=262144, scratch_size=1048576
check_iters=10, warmup_iters=10, iters=50
KERNEL_FILE=pipeline_kernel_simplified.cu

             size(B)        avg_time(us)        min_time(us)     avg_algbw(GB/s)     max_algbw(GB/s)
          1073741824             5637.39             5235.71              177.39              191.00

############################################# Allreduce #############################################
nranks=8
k=4, nelem_per_send=131072, scratch_size=1048576
check_iters=10, warmup_iters=10, iters=50
KERNEL_FILE=pipeline_kernel_simplified.cu

             size(B)        avg_time(us)        min_time(us)     avg_algbw(GB/s)     max_algbw(GB/s)
          1073741824             8976.12             8891.39              111.41              112.47
```
- With proxy channels, it is observed that when `k` or `ninstance` is too large (AR: k>4, AG: k>8), both allreduce and allgather can hang. Allgather generally hangs at larger `k`, probably because it requires less number of channels.
- For proxy channels, increasing the number of parallel channels does not improve performance; however, for sm channels, it does.
- To implement `k` sm channels per proxy channel, we can let `k` sm channels use the same scratch buffer but writing to different offsets to simplify logic.
- Ensure if `connection_types[b]=x` at `a`, then `connection_types[a]=x` at `b`.
- Proxy channel (within a node at least) requires higher `nelem_per_send` (1MB?); otherwise, performance is extremely poor.
- Number of proxy channels: [issue](https://github.com/microsoft/mscclpp/issues/242)
