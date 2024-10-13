# Experiment Data

## Machine specifications
|Type   | Spec  |
| :---: | :---: |
| CPU   | Intel i9-13900KF |
| RAM   | 64GB  |
| GPU   | GeForce RTX 4070 | 
| VRAM  | 12GB |

## Data format

The measurements collected for each run can be found under `run_table.csv` with the following entry types:
| Entry | Type | Note |
| :---: | :---: | :--- |
| run_number | str | format: r*x* where _x_ is iteration |
| candidate_family | str | `mistral` OR `qwen` OR `gemma` |
| task_type | str | `generation` OR `answering` OR `summarization` |
| input_type | str | `short` OR `long` |
| model_version | str | format: **v**y where _y_ represents the version number |
|performance_score_type | str | `ans_correctness` OR `sum_correctness` OR `context_relevancy` |
| cpu_utilization | \[(timestamp, float)\] | a set of timestamped datapoints |
| ram_usage | \[(timestamp, int)\] | a set of timestamped datapoints |
| gpu_utilization | \[(timestamp, float)\] | a set of timestamped datapoints |
| vram_usage | \[(timestamp, int)\] | a set of timestamped datapoints |
| response_time | deltatime | time between request and response receival |
| performance_score | float | ranging from 0-1 |
| energy_consumption | \[(timestamp, int)\] | a set of timestamped datapoints |
| input_token_size | int | number of tokens of the input prompt |
| output_token_size | int | number of tokens of the model's response |
