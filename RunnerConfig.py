import os
import pandas as pd
import subprocess
import time
import json
import shlex
import signal
from transformers import AutoModelForCausalLM, AutoTokenizer
from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ExtendedTyping.Typing import SupportsStr
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
from deepeval.metrics import GEval, ContextualRelevancyMetric, SummarizationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from typing import Dict, Optional
from pathlib import Path


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Dictionary to store model configurations
model_configs = {
    "qwen-v1": {
        "model_name": "Qwen/Qwen-7B",
        "tokenizer_name": "Qwen/Qwen1.5-7B",
    },
    "qwen-v1.5": {
        "model_name": "Qwen/Qwen1.5-7B",
        "tokenizer_name": "Qwen/Qwen1.5-7B",
    },
    "qwen-v2": {
        "model_name": "Qwen/Qwen2-7B",
        "tokenizer_name": "Qwen/Qwen2-7B",
    },
    "qwen-v2.5": {
        "model_name": "Qwen/Qwen2.5-7B",
        "tokenizer_name": "Qwen/Qwen2.5-7B",
    },
    "gemma-v1": {
        "model_name": "google/gemma-2b-it",
        "tokenizer_name": "google/gemma-2b-it",
    },
    "gemma-v1.1": {
        "model_name": "google/gemma-1.1-2b-it",
        "tokenizer_name": "google/gemma-1.1-2b-it",
    },
    "gemma-v2": {
        "model_name": "google/gemma-2-2b-it",
        "tokenizer_name": "google/gemma-2-2b-it",
    },
    "mistral-v0.1": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
        "tokenizer_name": "mistralai/Mistral-7B-Instruct-v0.1",
    },
    "mistral-v0.2": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "tokenizer_name": "mistralai/Mistral-7B-Instruct-v0.2",
    },
    "mistral-v0.3": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "tokenizer_name": "mistralai/Mistral-7B-Instruct-v0.3",
    },
}

# List of prompts
prompts = {
    "generation": {
        "short": {
            "instruction": "Complete the sentence.",
            "content": "The weather today is",
        },
        "long": {
            "instruction": "Continue the paragraph with additional information that logically follows.",
            "content": "In various regions across the globe, the climate conditions can vary significantly depending on the season, geography, and local atmospheric factors. Some areas experience more frequent changes in weather patterns, while others remain stable for longer periods of time. When looking at today's forecast, one could observe...",
        },
    },
    "question_answering": {
        "short": {
            "instruction": "Provide the answer to the question.",
            "content": "What is the capital of France?",
            "expected_output": "Paris",
        },
        "long": {
            "instruction": "Based on the given information, provide a clear and concise answer to the question.",
            "content": "France, located in Western Europe, is a country with a rich history, culture, and diverse geography. It has played a major role in international politics, economics, and culture. One of the key aspects of any country is its capital, which often serves as the political, cultural, and economic hub. For France, what is its capital city?",
            "expected_output": "Paris",
        } 
    },
    "summarization": {
        "short": {
            "instruction": "Summarize the key points of the paragraph.",
            "content": "Global trade connects markets across continents, leading to the exchange of goods, services, and ideas. Technological advancements and faster transportation have driven exponential growth in international trade, creating new business opportunities and economic growth worldwide. However, challenges like trade imbalances, economic dependencies, and environmental concerns have arisen as a result. The role of international cooperation has become increasingly important to resolve disputes and manage these impacts.",
        },
        "long": {
            "instruction": "Summarize the key points of the paragraphs in a concise manner.",
            "content": "Global trade has evolved significantly over the past century, largely driven by advancements in transportation and communication technologies. This rapid growth has enabled businesses to access new markets and fostered international collaboration, leading to increased economic interdependence. However, with these benefits have come challenges, including increased competition and the risk of trade imbalances between nations. \n\nAt the same time, the rise of global trade has spurred significant changes in labor markets. Countries with access to cheaper labor have become manufacturing hubs, while higher-income nations have focused more on services and technology. This shift has led to wage disparities and political debates about the future of work in many economies. \n\nEnvironmental impacts of global trade have also become a pressing issue. Increased production and transportation contribute to higher greenhouse gas emissions and resource depletion. International efforts, such as environmental agreements, seek to mitigate these impacts, though balancing economic growth with sustainability remains a challenge. \n\nFinally, trade policies and agreements play a crucial role in shaping global trade dynamics. Countries enter into bilateral or multilateral agreements to reduce tariffs, promote free trade, or protect key industries. These agreements can boost economic ties but also lead to disputes over issues like intellectual property, market access, and labor standards.",
        },
    },
}

class RunnerConfig:
    ROOT_DIR = Path("../data/")  # Root directory for storing data
    name: str = "test_runner_experiment"  # Name of the experiment
    results_output_path: Path = ROOT_DIR / 'experiments'  # Path where results will be stored
    operation_type: OperationType = OperationType.AUTO  # Operation type for automatic execution
    time_between_runs_in_ms: int = 1000 * 60  # 60 seconds between runs
    repetitions: int = 30  # Number of repetitions for the experiment runs

    def __init__(self):
        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN, self.before_run),
            (RunnerEvents.START_RUN, self.start_run),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT, self.interact),
            (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
            (RunnerEvents.STOP_RUN, self.stop_run),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment)
        ])
        self.run_table_model = None
        self.run_data = {}

        self.model = None
        self.tokenizer = None

        self.power_profiler = None
        self.gpu_profiler = None
        self.cpu_profiler = None
        output.console_log("Custom config loaded")

    def load_model(self, context: RunnerContext):
        run_variation = context.run_variation["model_version"]
        if run_variation in model_configs:
            model_name = model_configs[run_variation]["model_name"]
            tokenizer_name = model_configs[run_variation]["tokenizer_name"]
            print(f"Loading model: {model_name}, tokenizer: {tokenizer_name}")
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            return model, tokenizer
        else:
            raise ValueError(f"Model configuration not found for run variation: {run_variation}")

    def create_run_table_model(self) -> RunTableModel:
        main_factor = FactorModel("model_version", list(model_configs.keys()))
        blocking_factor_1 = FactorModel("task_type", ['generation', 'question_answering', 'summarization'])
        co_factor = FactorModel("input_size", ['short', 'long'])
        self.run_table_model = RunTableModel(
            factors=[main_factor, blocking_factor_1, co_factor],
            repetitions=self.repetitions,
            data_columns=['cpu_utilization', 'ram_usage',
                          'gpu_utilization', 'vram_usage',
                          'performance_score', 'response_time',
                          'input_token_size', 'output_token_size',
                          'energy_consumption']
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        output.console_log("Config.before_experiment() called!")
        self.experiment_start_time = time.time()
        output.console_log("Experiment started.")

    def before_run(self) -> None:
        output.console_log("Config.before_run() called!")

    def start_run(self, context: RunnerContext) -> None:
        self.run_start_time = time.time()
        self.run_data = {}
        self.model, self.tokenizer = self.load_model(context)
        output.console_log("Config.start_run() called!")

    def start_measurement(self, context: RunnerContext) -> None:
        output.console_log("Config.start_measurement() called!")

        # Commands for GPU, power, and CPU profiling
        gpu_profiler_cmd = [
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
            '--format=csv,nounits', '-l', '1'
        ]
        power_profiler_cmd = [
            'powerjoular', '-l', '-f', str(context.run_dir / "powerjoular.csv")
        ]
        cpu_profiler_cmd = [
            'top', '-b', '-d', '1', '-u', os.getenv('USER')
        ]

        try:
            # Open the files in Python and redirect output there.
            gpu_output_file = open(context.run_dir / "nvidia-smi.csv", "w")
            power_output_file = open(context.run_dir / "powerjoular.csv", "w")
            cpu_output_file = open(context.run_dir / "top-output.txt", "w")

            # Start the profilers, redirecting their output to respective files.
            self.gpu_profiler = subprocess.Popen(
                gpu_profiler_cmd, stdout=gpu_output_file, stderr=subprocess.DEVNULL
            )
            self.power_profiler = subprocess.Popen(
                power_profiler_cmd, stdout=power_output_file, stderr=subprocess.DEVNULL
            )
            self.cpu_profiler = subprocess.Popen(
                cpu_profiler_cmd, stdout=cpu_output_file, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            output.console_log(f"Error starting profilers: {e}")
            self.cleanup()

    def interact(self, context: RunnerContext) -> None:
        output.console_log("Config.interact() called!")

        # Prepare input text and tokenize
        input_text = prompts[context.run_variation['task_type']][context.run_variation['input_size']]['content']
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)

        # Extract attention mask
        attention_mask = inputs['attention_mask']

        start_time = time.time()
        # Generate output with attention mask to avoid the error
        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=attention_mask,
            max_new_tokens=100
        )
        end_time = time.time()

        self.run_data["response_time"] = end_time - start_time
        self.run_data["output_text"] = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        output.console_log(f"Generated output: {self.run_data['output_text']}")

    def stop_measurement(self, context: RunnerContext) -> None:
        output.console_log("Config.stop_measurement called!")
        try:
            if self.power_profiler:
                os.kill(self.power_profiler.pid, signal.SIGINT)
                self.power_profiler.wait()
            if self.gpu_profiler:
                os.kill(self.gpu_profiler.pid, signal.SIGINT)
                self.gpu_profiler.wait()
            if self.cpu_profiler:
                os.kill(self.cpu_profiler.pid, signal.SIGINT)
                self.cpu_profiler.wait()
        except Exception as e:
            output.console_log(f"Error stopping profilers: {e}")

    def stop_run(self, context: RunnerContext) -> None:
        output.console_log("Config.stop_run() called!")
        run_end_time = time.time()
        total_run_time = run_end_time - self.run_start_time
        estimated_total_time = ((total_run_time + 60) * self.repetitions) / 60 / 60

        self.model = None
        self.tokenizer = None

        output.console_log(f"Run completed in {total_run_time:.2f} seconds.")
        output.console_log(f"Estimated total time to completion: {estimated_total_time:.2f} hours")

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, SupportsStr]]:
        try:
            # Reading power profiling data
            try:
                power_df = pd.read_csv(context.run_dir / "powerjoular.csv")
            except FileNotFoundError as e:
                output.console_log(f"Power profiling data not found: {e}")
                return None
            except pd.errors.EmptyDataError as e:
                output.console_log(f"Power profiling data is empty: {e}")
                return None

            # Reading GPU profiling data
            try:
                gpu_df = pd.read_csv(context.run_dir / "nvidia-smi.csv")
                # Strip whitespace and handle potential column name variations
                gpu_df.columns = gpu_df.columns.str.strip()
                output.console_log(f"GPU profiling data columns: {gpu_df.columns}")

                # Extract the GPU utilization and VRAM usage
                gpu_utilization = gpu_df.get('utilization.gpu [%]', gpu_df.get('utilization.gpu')).to_list()
                vram_usage = gpu_df.get('memory.used [MiB]', gpu_df.get('memory.used')).to_list()

            except FileNotFoundError as e:
                output.console_log(f"GPU profiling data not found: {e}")
                return None
            except pd.errors.EmptyDataError as e:
                output.console_log(f"GPU profiling data is empty: {e}")
                return None
            except Exception as e:
                output.console_log(f"Unexpected error reading GPU profiling data: {e}")
                return None

            # Reading CPU profiling data from top-output.txt
            cpu_usage = []
            memory_usage = []
            try:
                with open(context.run_dir / 'top-output.txt', 'r') as file:
                    for line in file:
                        columns = line.split()
                        if len(columns) > 8:
                            # Append CPU usage (9th column) and RES memory (6th column) to the respective lists
                            cpu_usage.append(columns[8])
                            memory_usage.append(columns[5])
            except FileNotFoundError as e:
                output.console_log(f"CPU profiling data file not found: {e}")
                return None
            except Exception as e:
                output.console_log(f"Unexpected error reading CPU profiling data: {e}")
                return None

            # Evaluate performance using Deepeval
            try:
                task_type = context.run_variation['task_type']
                input_size = context.run_variation['input_size']
                prompt = prompts[task_type][input_size]
                retrieval_context = [prompt['content']]

                if task_type == 'question_answering' and 'expected_output' in prompt:
                    expected_output = prompt['expected_output']
                    test_case = LLMTestCase(
                        input=prompt['content'],
                        actual_output=self.run_data["output_text"],
                        expected_output=expected_output,  # Correctly provide expected output
                        retrieval_context=retrieval_context
                    )
                    metric = GEval(
                        name="Correctness",
                        model="gpt-4",
                        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                        evaluation_steps=[
                            "Check whether the facts in 'actual output' contradict any facts in 'expected_output'"
                        ]
                    )
                elif task_type == 'generation':
                    test_case = LLMTestCase(
                        input=prompt['content'],
                        actual_output=self.run_data["output_text"],
                        retrieval_context=retrieval_context
                    )
                    metric = ContextualRelevancyMetric(
                        threshold=0.7,
                        model="gpt-4",
                    )
                elif task_type == 'summarization':
                    test_case = LLMTestCase(
                        input=prompt['content'],
                        actual_output=self.run_data["output_text"],
                        retrieval_context=retrieval_context
                    )
                    metric = SummarizationMetric(
                        threshold=0.5,
                        model="gpt-4",
                        assessment_questions=[
                            "Is the coverage score based on a percentage of 'yes' answers?",
                            "Does the score ensure the summary's accuracy with the source?",
                            "Does a higher score mean a more comprehensive summary?"
                        ]
                    )
                else:
                    raise ValueError(f"Unknown task type: {task_type}")

                # Measuring the metric score
                metric.measure(test_case)
                self.run_data["performance_score"] = metric.score

            except KeyError as e:
                output.console_log(f"KeyError when accessing prompts or run data: {e}")
                return None
            except ValueError as e:
                output.console_log(f"ValueError in performance evaluation: {e}")
                return None
            except Exception as e:
                output.console_log(f"Unexpected error during performance evaluation: {e}")
                return None

            # Return run data
            try:
                return {
                    "cpu_utilization": cpu_usage,
                    "ram_usage": memory_usage,
                    "gpu_utilization": gpu_utilization,
                    "vram_usage": vram_usage,
                    "response_time": self.run_data['response_time'],
                    "performance_score": self.run_data['performance_score'],
                    "energy_consumption": power_df['Total Power'].to_list() if 'Total Power' in power_df.columns else []
                }
            except KeyError as e:
                output.console_log(f"KeyError when preparing run data to return: {e}")
                return None
            except Exception as e:
                output.console_log(f"Unexpected error preparing run data: {e}")
                return None

        except Exception as e:
            import traceback
            error_message = f"Error populating run data: {e}\n{traceback.format_exc()}"
            output.console_log(error_message)
            return None



    def after_experiment(self) -> None:
        experiment_end_time = time.time()
        total_experiment_duration = experiment_end_time - self.experiment_start_time
        hours, remainder = divmod(total_experiment_duration, 3600)
        minutes, _ = divmod(remainder, 60)
        output.console_log(f"Total experiment duration: {int(hours)} hours and {int(minutes)} minutes.")
        output.console_log("Config.after_experiment() called!")

    def cleanup(self):
        try:
            if self.power_profiler and self.power_profiler.poll() is None:
                self.power_profiler.terminate()
            if self.gpu_profiler and self.gpu_profiler.poll() is None:
                self.gpu_profiler.terminate()
            if self.cpu_profiler and self.cpu_profiler.poll() is None:
                self.cpu_profiler.terminate()
        except Exception as e:
            output.console_log(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        config = RunnerConfig()
        config.create_run_table_model()
    except Exception as e:
        config.cleanup()
        output.console_log(f"Error occurred during execution: {e}")
