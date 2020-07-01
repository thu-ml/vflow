# VFlow: More Expressive Generative Flows with Variational Data Augmentation

This repository contains Tensorflow implementation of experiments from the paper [VFlow: More Expressive Generative Flows with Variational Data Augmentation](https://arxiv.org/abs/2002.09741). The implementation is based on [Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design](https://github.com/aravindsrinivas/flowpp).

# Main Dependencies

* Python >= 3.6 
* Tensorflow v1.14.0
* [horovod v0.18.2](https://github.com/uber/horovod)

[Horovod GPU setup instructions](https://github.com/uber/horovod/blob/master/docs/gpus.rst)

# Usage Instructions on One Machine

We trained our models with data-parallelism using Horovod. For simply usage in one machine (e.g. 8 GPUs):

## CIFAR 10 
```
horovodrun -np 8 python3 run_cifar.py
```

## ImageNet 

### Data for ImageNet Experiments: 
Script to create dataset [here](https://github.com/thu-ml/vflow/blob/master/flows_imagenet/create_imagenet_benchmark_datasets.py)

### ImageNet 32x32
```
horovodrun -np 8 python3 imagenet32.py
```
### ImageNet 64x64
```
horovodrun -np 8 python3 imagenet64.py
```

# Multi-Machine Usage

## Network with Infiniband

Create a file named as `hostfile`:
(Modify `$ip_for_another_machine_for_ib_interface`)
```
localhost slots=8
$ip_for_another_machine_for_ib_interface slots=8
```

Then create a script named as `run.sh`:
(Modify `$ib_network_interface`, `ib_device`, e.g. `ib0`, `mlx5_0`)
```
mpirun -np 16 \
 --hostfile hostfile \
 -bind-to none -map-by slot \
 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH  -x NCCL_SOCKET_IFNAME=$ib_network_interface \
 -mca pml ob1 -mca btl self,openib -mca btl_openib_if_include $ib_device \
 bash -c "python /dir_to_vflow/flows/run_cifar10.py"
```

## Network with TCP

Create a file named as `hostfile`:
(Modify `$ip_for_another_machine_for_tcp_interface`)
```
localhost slots=8
$ip_for_another_machine_for_tcp_interface slots=8
```

Then create a script named as `run.sh`:
(Modify `$tcp_network_interface`, e.g. `enp94s0f0`)
```
mpirun -np 16 \
 --hostfile hostfile \
 -bind-to none -map-by slot \
 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=$tcp_network_interface \
 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include $tcp_network_interface \
 bash -c "python /dir_to_vflow/flows/run_cifar10.py"
```


# Contact

Please open an issue.

# Cite
Please cite our paper if you use this code in your own work:

```
@inproceedings{chen2020vflow,
  title={VFlow: More Expressive Generative Flows with Variational Data Augmentation},
  author={Chen, Jianfei and Lu, Cheng and Chenli, Biqi and Zhu, Jun and Tian, Tian},
  booktitle={International Conference on Machine Learning},
  year={2020}
}
```
