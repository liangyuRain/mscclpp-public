# Setup on Lambda Machine
- BBN's lambda machine has cuda 12.2. Thus, the base docker image under `mscclpp-public/docker` has to be changed to `FROM nvidia/cuda:12.2.0-devel-ubuntu20.04`. The following env variables are also required for docker image on lambda machine to have internet access:
```docker
ENV HTTP_PROXY="http://proxy.bbn.com:3128"
ENV HTTPS_PROXY="http://proxy.bbn.com:3128"
ENV http_proxy="http://proxy.bbn.com:3128"
ENV https_proxy="http://proxy.bbn.com:3128"
```
- Build an image called `mscclpp-public`: 
```shell
docker build -t mscclpp-public -f mscclpp-public/docker/base-x-cuda12.2.dockerfile .
```
- Create a container named `mscclpp-public` using the image `mscclpp-public` we just built:
```shell
docker run --env HTTP_PROXY="http://proxy.bbn.com:3128" \
           --env HTTPS_PROXY="http://proxy.bbn.com:3128" \
           --env http_proxy="http://proxy.bbn.com:3128" \
           --env https_proxy="http://proxy.bbn.com:3128" -d --name mscclpp --gpus all mscclpp tail -f /dev/null
```
```shell
docker run --ulimit memlock=-1:-1 \
           --device=/dev/infiniband/uverbs0 \
           --device=/dev/infiniband/uverbs1 \
           --device=/dev/infiniband/uverbs2 \
           --device=/dev/infiniband/uverbs3 \
           --device=/dev/infiniband/uverbs4 \
           --device=/dev/infiniband/uverbs5 \
           --device=/dev/infiniband/uverbs6 \
           --device=/dev/infiniband/uverbs7 \
           --device=/dev/infiniband/uverbs8 \
           --device=/dev/infiniband/uverbs9 \
           --device=/dev/infiniband/rdma_cm \
           --env HTTP_PROXY="http://proxy.bbn.com:3128" \
           --env HTTPS_PROXY="http://proxy.bbn.com:3128" \
           --env http_proxy="http://proxy.bbn.com:3128" \
           --env https_proxy="http://proxy.bbn.com:3128" -d --name mscclpp-public --gpus all mscclpp-public tail -f /dev/null
```
- Copy `mscclpp-public` into the container:
```shell
docker cp mscclpp-public/ mscclpp-public:/root/
```
- Enter the container:
```shell
docker exec -it mscclpp-public bash
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

# Notes
- Error `ibv_create_cq(cqe=4096) failed: Cannot allocate memory` is caused by not setting `max locked memory` to `unlimited`. One can check by running `ulimit -a`. The solution is to add `--ulimit memlock=-1:-1` when `docker run`. There seems to be a discrepancy between host and docker container in `ulimit -a` by default.
- Run `ib_send_bw -d mlx5_0 -i 1 -a -R --report_gbits` on both server and client sides to check IB bandwidth. (never successful yet on lambda)
- If report compile error for `#include <mscclpp/sm_channel_device.hpp>` not found, check if the `include_dir` in `KernelBuilder` from `mscclpp-public/python/mscclpp/utils.py` is as following:
```python
include_dir = os.path.join(self._current_file_dir, "../../include")
```
