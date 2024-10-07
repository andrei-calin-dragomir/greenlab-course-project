# Comparative Evaluation of Energy Efficiency in Large Language Models: *Analyzing Improvements Across Incremental Versions in Inference Tasks*

This experiment is a group project for the GreenLab course, under the Computer Science Master's programme at VU Amsterdam.

## Experiment Candidates

### 1. Alibaba Cloud’s QWen:
The versions that are going to be tested are incrementally as follows:
- [Qwen1-7B](https://huggingface.co/Qwen/Qwen-7B "‌")
- [Qwen1.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B "‌")
- [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-1.5B "‌")
- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B "‌")

### 2. Google’s Gemma
The versions that are going to be tested are incrementally as follows:
- [Gemma-2B-it](https://huggingface.co/google/gemma-2b-it "‌")
- [Gemma-1.1-2B-it](https://huggingface.co/google/gemma-1.1-2b-it "‌")
- [Gemma-2-2B-it](https://huggingface.co/google/gemma-2-2b-it "‌")

These versions are all instruct versions of the Gemma model. This will not affect our study because we are not drawing any comparative conclusions on the model performance between the LLM candidates.

### 3. Mistralai’s Mistral
The versions that are going to be tested are incrementally as follows:
- [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 "‌")
- [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 "‌")
- [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 "‌")

These versions are all instruct versions of the open-source Mistral model model. Just as for Gemma, this will not affect our study because we are not drawing any comparative conclusions on the model performance between the LLM candidates.

## Tool Selection

# Experiment Automation
We automated the experiment using the following framework: [experiment-runner](https://github.com/S2-group/experiment-runner).

The workflow of the experiment is defined as:
**TODO** Write here the direct translation of the experiment runner config steps. It should be matching with Section 5 of the paper.

# Metrics Extraction
- Energy Consumption (CPU/GPU):
    - [powerjoular](https://wiki.archlinux.org/title/Powertop) : 
        - **CPU Consumption (Joules)**
        - **GPU Consumption (Joules)**
- Resource Utilization:
    - [htop](https://linux.die.net/man/1/htop) : **CPU Utilization (%), Memory Utilization (Bytes/%)**
    - [nvidia-smi](https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf) : **GPU Utilization (%), GPU VMemory (Bytes/%)**
