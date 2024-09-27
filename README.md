# Greenlab Course Project

This experiment is a group project for the GreenLab course, under the Computer Science Master's programme at VU Amsterdam.

## Subject Candidates

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

## Experiment Data

The measurements collected for each run can be found under `data/run_table.csv` with the following entry types:
| Entry | Type | Note |
| :---: | :---: | :--- |
| run*number | str | format: r*x* where \_x* is iteration |
| candidate*family | str | `mistral` OR `qwen` OR `gemma` |
|performance_score_type | str | `ans_correctness` OR `sum_correctness` OR `context_relevancy` |
| task_type | str | `generation` OR `answering` OR `summarization` |
| input_type | str | `small` OR `large` |
| release_version | str | format: **v**y where \_y* represents the version number |
| gpu_utilization | \[(timestamp, float)\] | a set of timestamped datapoints |
| cpu_utilization | \[(timestamp, float)\] | a set of timestamped datapoints |
| memory_usage | \[(timestamp, int)\] | a set of timestamped datapoints |
| response_time | deltatime | time between request and response receival |
| performance_score | float | ranging from 0-1 |
| energy_consumption | \[(timestamp, int)\] | a set of timestamped datapoints |
| input_token_size | int | number of tokens of the input prompt |
| output_token_size | int | number of tokens of the model's response |
