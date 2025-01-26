#!/bin/bash

# Activate the conda environment
source activate hipporag_ref
export HF_HOME=$MY_HF_HOME

# Set the CUDA visible devices and run the command with nohup
CUDA_VISIBLE_DEVICES="6" nohup vllm serve osunlp/UGround-V1-2B --tensor_parallel_size 1 --port 6999 --disable-log-requests > logs/uground_backend.log 2>&1 &