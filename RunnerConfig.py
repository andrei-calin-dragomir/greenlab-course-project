import os
import pandas as pd
import subprocess
import time
import json
import shlex
import signal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ExtendedTyping.Typing import SupportsStr
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
from deepeval.metrics import ContextualRelevancyMetric, SummarizationMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from typing import Dict, Optional
from pathlib import Path
from pydantic import BaseModel

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN is not set. Please set it as an environment variable.")


if os.geteuid() != 0:
    raise PermissionError("This script must be run as root (with sudo) to use powerjoular.")


model_configs = {
    "qwen-v1": {
        "model_name": "Qwen/Qwen-7B",
        "tokenizer_name": "Qwen/Qwen1.5-7B",
        "device": "cuda"
    },
    "qwen-v1.5": {
        "model_name": "Qwen/Qwen1.5-7B",
        "tokenizer_name": "Qwen/Qwen1.5-7B",
        "device": "cuda"
    },
    "qwen-v2": {
        "model_name": "Qwen/Qwen2-7B",
        "tokenizer_name": "Qwen/Qwen2-7B",
        "device": "cuda"
    },
    "qwen-v2.5": {
        "model_name": "Qwen/Qwen2.5-7B",
        "tokenizer_name": "Qwen/Qwen2.5-7B",
        "device": "cuda"
    },
    "gemma-v1": {
        "model_name": "google/gemma-2b-it",
        "tokenizer_name": "google/gemma-2b-it",
        "device": "cuda"
    },
    "gemma-v1.1": {
        "model_name": "google/gemma-1.1-2b-it",
        "tokenizer_name": "google/gemma-1.1-2b-it",
        "device": "cuda"
    },
    "gemma-v2": {
        "model_name": "google/gemma-2-2b-it",
        "tokenizer_name": "google/gemma-2-2b-it",
        "device": "cuda"
    },
    "mistral-v0.1": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
        "tokenizer_name": "mistralai/Mistral-7B-Instruct-v0.1",
        "device": "cuda"
    },
    "mistral-v0.2": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "tokenizer_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "device": "cuda"
    },
    "mistral-v0.3": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "tokenizer_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "device": "cuda"
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
            "content": "Global trade has evolved significantly over the past century, largely driven by advancements in transportation and communication technologies. This rapid growth has enabled businesses to access new markets and fostered international collaboration, leading to increased economic interdependence. However, with these benefits have come challenges, including increased competition and the risk of trade imbalances between nations.\n\nAt the same time, the rise of global trade has spurred significant changes in labor markets. Countries with access to cheaper labor have become manufacturing hubs, while higher-income nations have focused more on services and technology. This shift has led to wage disparities and political debates about the future of work in many economies.\n\nEnvironmental impacts of global trade have also become a pressing issue. Increased production and transportation contribute to higher greenhouse gas emissions and resource depletion. International efforts, such as environmental agreements, seek to mitigate these impacts, though balancing economic growth with sustainability remains a challenge.\n\nFinally, trade policies and agreements play a crucial role in shaping global trade dynamics. Countries enter into bilateral or multilateral agreements to reduce tariffs, promote free trade, or protect key industries. These agreements can boost economic ties but also lead to disputes over issues like intellectual property, market access, and labor standards.",
        },
    },
}


class CustomLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str, tokenizer_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True  # Ensuring the custom code from the model's repository is trusted.
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str:
        gen_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            max_length=250,
            num_return_sequences=1
        )

        output = gen_pipeline(prompt)[0]["generated_text"]
        return output

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Custom LLM"
    
class RunnerConfig:
    ROOT_DIR = Path("../data/")
    name: str = "test_runner_experiment"
    results_output_path: Path = ROOT_DIR / 'experiments'
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 1000 * 60
    repetitions: int = 30

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
            device = model_configs[run_variation]["device"]
            print(f"Loading model: {model_name}, tokenizer: {tokenizer_name} on device: {device}")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=HUGGINGFACE_API_TOKEN,
                    trust_remote_code=True,  # Added this line to avoid the need for user input
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                )
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=HUGGINGFACE_API_TOKEN, trust_remote_code=True)
                return model, tokenizer
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    output.console_log("CUDA out of memory error. Trying CPU fallback...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        token=HUGGINGFACE_API_TOKEN,
                        trust_remote_code=True
                    ).to("cpu")
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=HUGGINGFACE_API_TOKEN, trust_remote_code=True)
                    return model, tokenizer
                else:
                    raise RuntimeError(f"Failed to load model/tokenizer: {e}")
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

        gpu_profiler_cmd = [
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
            '--format=csv,nounits', '-l', '1'
        ]
        power_profiler_cmd = [
            'sudo', '/usr/bin/powerjoular', '-l', '-f', str(context.run_dir / "powerjoular.csv")
        ]
        cpu_profiler_cmd = [
            'top', '-b', '-d', '1', '-u', os.getenv('USER')
        ]

        try:
            gpu_output_file = open(context.run_dir / "nvidia-smi.csv", "w")
            power_output_file = open(context.run_dir / "powerjoular.csv", "w")
            cpu_output_file = open(context.run_dir / "top-output.txt", "w")

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

        input_text = prompts[context.run_variation['task_type']][context.run_variation['input_size']]['content']
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)


        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        try:
            start_time = time.time()
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', None),
                max_new_tokens=100
            )
            end_time = time.time()

            self.run_data["response_time"] = end_time - start_time
            self.run_data["output_text"] = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            output.console_log(f"Generated output: {self.run_data['output_text']}")

        except RuntimeError as e:
            output.console_log(f"RuntimeError during generation: {e}")

    def stop_measurement(self, context: RunnerContext) -> None:
        output.console_log("Config.stop_measurement called!")
        try:
            for profiler in [self.power_profiler, self.gpu_profiler, self.cpu_profiler]:
                if profiler and profiler.poll() is None:
                    profiler.terminate()
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
            # Power profiling
            try:
                power_df = pd.read_csv(context.run_dir / "powerjoular.csv")
            except FileNotFoundError:
                output.console_log("Power profiling data not found.")
                power_df = pd.DataFrame()

            # GPU profiling
            try:
                gpu_df = pd.read_csv(context.run_dir / "nvidia-smi.csv")
                gpu_df.columns = gpu_df.columns.str.strip()
                gpu_utilization = gpu_df.get('utilization.gpu [%]', []).tolist()
                vram_usage = gpu_df.get('memory.used [MiB]', []).tolist()
            except FileNotFoundError:
                output.console_log("GPU profiling data not found.")
                gpu_utilization, vram_usage = [], []

            # CPU profiling
            cpu_usage = []
            memory_usage = []
            try:
                with open(context.run_dir / 'top-output.txt', 'r') as file:
                    for line in file:
                        columns = line.split()
                        if len(columns) > 8:
                            cpu_usage.append(columns[8])
                            memory_usage.append(columns[5])
            except FileNotFoundError:
                output.console_log("CPU profiling data file not found.")

            # Performance evaluation using custom model
            try:
                task_type = context.run_variation['task_type']
                input_size = context.run_variation['input_size']
                prompt = prompts[task_type][input_size]

                test_case = LLMTestCase(
                    input=prompt['content'],
                    actual_output=self.run_data["output_text"],
                    retrieval_context=[prompt['content']]
                )

                # Use the CustomLLM for evaluation instead of GPT-4
                model_name = model_configs[context.run_variation['model_version']]["model_name"]
                tokenizer_name = model_configs[context.run_variation['model_version']]["tokenizer_name"]
                custom_llm = CustomLLM(model_name, tokenizer_name)

                # Choose the appropriate metric and evaluate
                if task_type == 'generation':
                    metric = ContextualRelevancyMetric(model=custom_llm, threshold=0.7)
                else:
                    metric = SummarizationMetric(model=custom_llm, threshold=0.5)

                metric.measure(test_case)
                self.run_data["performance_score"] = metric.score

            except Exception as e:
                output.console_log(f"Unexpected error during performance evaluation: {e}")
                return None

            return {
                "cpu_utilization": cpu_usage,
                "ram_usage": memory_usage,
                "gpu_utilization": gpu_utilization,
                "vram_usage": vram_usage,
                "response_time": self.run_data['response_time'],
                "performance_score": self.run_data['performance_score'],
                "energy_consumption": power_df['Total Power'].to_list() if 'Total Power' in power_df.columns else []
            }

        except Exception as e:
            output.console_log(f"Error populating run data: {e}")
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
            for profiler in [self.power_profiler, self.gpu_profiler, self.cpu_profiler]:
                if profiler and profiler.poll() is None:
                    profiler.terminate()
        except Exception as e:
            output.console_log(f"Error during cleanup: {e}")
            
if __name__ == "__main__":
    try:
        config = RunnerConfig()
        config.create_run_table_model()
    except Exception as e:
        config.cleanup()
        output.console_log(f"Error occurred during execution: {e}")
