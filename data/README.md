# Experiment Data

## Experimental Setup

**TODO** Write specifications of the machine we ran the experiment on.

## Data format

The measurements collected for each run can be found under `run_table.csv` with the following entry types:
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
