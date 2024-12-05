# Comparative Evaluation of Energy Efficiency in Large Language Models: _Analyzing Improvements Across Incremental Versions in Inference Tasks_

## Overview

This project focuses on collecting, processing, and analyzing data based on various inference and loading tasks. The data is divided into multiple categories and classes, as described below.
**This experiment is a group project for the GreenLab course, under the Computer Science Master's programme at VU Amsterdam.**

---

## Subjects

### Alibaba Cloud’s QWen

| Version Name    | Description                     | Link                                             | Parameters |
| --------------- | ------------------------------- | ------------------------------------------------ | ---------- |
| Qwen1.5         | Incremental update              | [Qwen:7b](https://ollama.com/library/qwen:7b "‌") | 7.72b      |
| Qwen2           | Major update to Qwen            | [Qwen2](https://ollama.com/library/qwen2 "‌")     | 7.62b      |
| Qwen2.5         | Latest incremental update       | [Qwen2.5](https://ollama.com/library/qwen2.5 "‌") | 7.62b      |

---

### Google’s Gemma

| Version Name    | Description                     | Link                                          | Parameters |
| --------------- | ------------------------------- | --------------------------------------------- | ---------- |
| Gemma           | Initial version updated to v1.1 | [Gemma](https://ollama.com/library/gemma "‌")  | 8.54b      |
| Gemma2          | Incremental update              | [Gemma2](https://ollama.com/library/gemma2 "‌")| 9.24b      |

---

### Mistralai’s Mistral

| Version Name    | Description                         | Link                                                       | Parameters |
| --------------- | ----------------------------------- | ---------------------------------------------------------- | ---------- |
| Mistral-v0.1    | Initial instruct version of Mistral | [Mistral:v0.1](https://ollama.com/library/mistral:v0.1 "‌") | 7.24b      |
| Mistral-v0.2    | Incremental update to Mistral       | [Mistral:v0.2](https://ollama.com/library/mistral:v0.2 "‌") | 7.24b      |
| Mistral-v0.3    | Latest incremental update           | [Mistral:v0.3](https://ollama.com/library/mistral:v0.3 "‌") | 7.25b      |

### Meta's Llama

| Version Name    | Description                         | Link                                                       | Parameters |
| --------------- | ----------------------------------- | ---------------------------------------------------------- | ---------- |
| Llama2          |                                     | [Llama2](https://ollama.com/library/llama2 "‌")             | 6.74b      |
| Llama3          |                                     | [Llama3](https://ollama.com/library/llama3 "‌")             | 8.03b      |
| Llama3.1        |                                     | [Llama3.1](https://ollama.com/library/llama3 "‌")           | 8.03b      |

### Microsoft's Phi

| Version Name    | Description                         | Link                                                       | Parameters |
| --------------- | ----------------------------------- | ---------------------------------------------------------- | ---------- |
| Phi             | Actually version 2 of Phi           | [Phi](https://ollama.com/library/phi "‌")                   | 2.78b      |
| Phi3            |                                     | [Phi3](https://ollama.com/library/phi3 "‌")                 | 3.82b      |
| Phi3.5          |                                     | [Phi3.5](https://ollama.com/library/phi3.5 "‌")             | 3.82b      |

---

### Tool Selection

| Tool Name         | Purpose                             | Link                                                               |
| ----------------- | ----------------------------------- | ------------------------------------------------------------------ |
| Experiment-Runner | Automates the experiment process    | [Experiment-Runner](https://github.com/S2-group/experiment-runner) |
| Paramiko          | Executing commands on exp. machine  | [Paramiko](https://www.paramiko.org/)                              |
| Energibridge      | Collects all exp. machine metrics   | [Energibridge](https://github.com/tdurieux/EnergiBridge.git)       |
| Evaluate          | Provides model performance scores   | [Evaluate](https://huggingface.co/docs/evaluate/en/index)          |

---

## Data Collected Per Class

| **Category**           | **Percentages**                                  | **Values**                                                                  |
| ---------------------- | ------------------------------------------------ | --------------------------------------------------------------------------- |
| **Inference**          | `CPU_USAGE_0`, ..., `CPU_USAGE_31`, `GPU0_USAGE` | `GPU0_MEMORY_USED`, `USED_MEMORY`, `USED_SWAP`                              |
|                        |                                                  | `DRAM_ENERGY (J)`, `PACKAGE_ENERGY (J)`, `PP0_ENERGY (J)`, `PP1_ENERGY (J)` |
|                        |                                                  | `GPU0_ENERGY (mJ)`                                                       |
| **Additional Metrics** | `rouge_scores`, `bleu_scores`                              | `inference_time`                                                            |


---

## Machine Specifications

The following table describes the hardware specifications of the machine used for the experiments:

| **Specification** | **Details**      |
| ----------------- | ---------------- |
| **CPU**           | Intel i9-13900KF |
| **Memory Size**   | 64 GiB           |
| **GPU**           | GeForce RTX 4070 |
| **VRAM**          | 12 GiB           |

---

## Running the Experiment

### Prerequisites

- The experimental machine should have [ollama](https://ollama.com/) installed in order to run the models.

- You should create a `.env` file containing:
```python
HOST='ip of the experimental machine'
EXPERIMENTAL_MACHINE_USER='username of account on experimental machine'
PASSWORD='password of account on experimental machine'
```



### Installation


```bash
# On orchestrating machine
git clone --recursive https://github.com/andrei-calin-dragomir/greenlab-course-project.git
cd ./greenlab-course-project
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

```bash
# On experimental machine
git clone --recursive https://github.com/andrei-calin-dragomir/greenlab-course-project.git
cd ./greenlab-course-project/Energibridge

sudo chgrp -R <experimental_machine_user> /dev/cpu/*/msr;
sudo chmod g+r /dev/cpu/*/msr;
cargo build -r;
sudo setcap cap_sys_rawio=ep target/release/energibridge;
```

### Execution (from orchestration machine)

```bash
source ./venv/bin/activate
python experiment-runner/experiment-runner ./RunnerConfig.py
```
