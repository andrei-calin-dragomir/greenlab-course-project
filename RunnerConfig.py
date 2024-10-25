import os
import pandas as pd
import subprocess
import time
import json
import shlex
import signal
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ExtendedTyping.Typing import SupportsStr
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
from typing import Dict, Optional
from pathlib import Path
from pydantic import BaseModel
from evaluate import load

# Set environment variable for CUDA memory allocation settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load Huggingface API token from environment variables
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN is not set. Please set it as an environment variable.")

# Check if script is run as root for powerjoular profiling
if os.geteuid() != 0:
    raise PermissionError("This script must be run as root (with sudo) to use powerjoular.")

# Model configurations for various versions of Gemma, Mistral, and Qwen models
model_configs = {
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
    "qwen-v1": {
        "model_name": "Qwen/Qwen-7B",
        "tokenizer_name": "Qwen/Qwen-7B",
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
}

# Prompts for different NLP tasks (generation, question-answering, summarization)
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

class RunnerConfig:
    # Class for runner configuration, including paths, event subscriptions, and metrics
    ROOT_DIR = Path("../data/")
    name: str = "test_runner_experiment"
    results_output_path: Path = ROOT_DIR / 'experiments'
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 1000 * 60  # Delay between runs
    repetitions: int = 30  # Number of experiment repetitions

    def __init__(self):
        # Subscribe to multiple runner events
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
        # Initialize models and profiling variables
        self.run_table_model = None
        self.run_data = {}
        self.bleu_metric = load("bleu")
        self.rouge_metric = load("rouge")
        self.model = None
        self.tokenizer = None
        self.power_profiler = None
        self.gpu_profiler = None
        self.cpu_profiler = None

        output.console_log("Custom config loaded")

    # Load Gemma model based on the provided model and tokenizer names
    def load_gemma_model(self, model_name: str, tokenizer_name: str):
        torch.cuda.empty_cache()
        gc.collect()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            token=os.getenv("HUGGINGFACE_API_TOKEN")
        )
        model.gradient_checkpointing = True  # Enable gradient checkpointing
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            token=os.getenv("HUGGINGFACE_API_TOKEN"),
            trust_remote_code=True
        )
        return model, tokenizer

    # Load Mistral model
    def load_mistral_model(self, model_name: str, tokenizer_name: str):
        torch.cuda.empty_cache()
        gc.collect()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            token=os.getenv("HUGGINGFACE_API_TOKEN")
        )
        model.gradient_checkpointing = True
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            token=os.getenv("HUGGINGFACE_API_TOKEN"),
            trust_remote_code=True
        )
        return model, tokenizer

    # Load Qwen model
    def load_qwen_model(self, model_name: str, tokenizer_name: str):
        torch.cuda.empty_cache()
        gc.collect()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            offload_buffers=True
        )
        model.gradient_checkpointing = True
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        return model, tokenizer

    # Load model based on context and model version
    def load_model(self, context: RunnerContext):
        run_variation = context.run_variation["model_version"]

        if run_variation in model_configs:
            model_name = model_configs[run_variation]["model_name"]
            tokenizer_name = model_configs[run_variation]["tokenizer_name"]
            print(f"Loading model: {model_name}, tokenizer: {tokenizer_name} on GPU")

            if "gemma" in run_variation:
                return self.load_gemma_model(model_name, tokenizer_name)
            elif "mistral" in run_variation:
                return self.load_mistral_model(model_name, tokenizer_name)
            elif "qwen" in run_variation:
                return self.load_qwen_model(model_name, tokenizer_name)
            else:
                raise ValueError(f"Model family not recognized for run variation: {run_variation}")

        else:
            raise ValueError(f"Model configuration not found for run variation: {run_variation}")

    # Create the run table model based on experiment factors
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

    # Event methods
    def before_experiment(self) -> None:
        output.console_log("Config.before_experiment() called!")
        self.experiment_start_time = time.time()
        output.console_log("Experiment started.")

    def before_run(self) -> None:
        output.console_log("Config.before_run() called!")

    # Start a run, load model, and initialize profiling
    def start_run(self, context: RunnerContext) -> None:
        self.run_start_time = time.time()
        self.run_data = {}
        self.model, self.tokenizer = self.load_model(context)
        output.console_log("Config.start_run() called!")

    def start_measurement(self, context: RunnerContext) -> None:
        output.console_log("Config.start_measurement() called!")

        # Start profiling commands for GPU, power, and CPU
        gpu_profiler_cmd = [
            'sudo', 'nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
            '--format=csv,nounits', '-l', '1'
        ]
        power_profiler_cmd = [
            'sudo', '/usr/bin/powerjoular', '-l', '-f', str(context.run_dir / "powerjoular.csv")
        ]
        cpu_profiler_cmd = [
            'sudo', 'top', '-b', '-d', '1', '-p', str(os.getpid()), '|', 'grep', f"\'{os.getpid()}\'", '--line-buffered'
        ]

        try:
            # Start GPU profiler
            gpu_output_file = open(context.run_dir / "nvidia-smi.csv", "w")
            self.gpu_profiler = subprocess.Popen(
                gpu_profiler_cmd, stdout=gpu_output_file, stderr=subprocess.DEVNULL
            )
            output.console_log("GPU profiler started successfully.")
        except Exception as e:
            output.console_log(f"Error starting GPU profiler: {e}")
            self.cleanup()

        # Start power profiler
        try:
            self.power_profiler = subprocess.Popen(
                power_profiler_cmd, stderr=subprocess.DEVNULL
            )
            output.console_log("Power profiler started successfully.")
        except Exception as e:
            output.console_log(f"Error starting power profiler: {e}")
            self.cleanup()

        # Start CPU profiler
        try:
            cpu_output_file = open(context.run_dir / "top-output.txt", "w")
            self.cpu_profiler = subprocess.Popen(
                cpu_profiler_cmd, stdout=cpu_output_file, stderr=subprocess.DEVNULL, shell=True
            )
            output.console_log("CPU profiler started successfully.")
        except Exception as e:
            output.console_log(f"Error starting CPU profiler: {e}")
            self.cleanup()

    # Interact with the loaded model using a prompt
    def interact(self, context: RunnerContext) -> None:
        output.console_log("Config.interact() called!")

        input_text = prompts[context.run_variation['task_type']][context.run_variation['input_size']]['content']
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=False, truncation=True)

        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        try:
            # Generate output and calculate response time
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

    # Stop measurement and terminate profiling processes
    def stop_measurement(self, context: RunnerContext) -> None:
        output.console_log("Config.stop_measurement called!")
        try:
            for profiler in [self.power_profiler, self.gpu_profiler, self.cpu_profiler]:
                if profiler and profiler.poll() is None:
                    profiler.terminate()
        except Exception as e:
            output.console_log(f"Error stopping profilers: {e}")

    # Stop a run, clean up resources, and log run completion time
    def stop_run(self, context: RunnerContext) -> None:
        output.console_log("Config.stop_run() called!")
        run_end_time = time.time()
        total_run_time = run_end_time - self.run_start_time
        estimated_total_time = ((total_run_time + 60) * self.repetitions) / 60 / 60

        self.model = None
        self.tokenizer = None

        output.console_log(f"Run completed in {total_run_time:.2f} seconds.")
        output.console_log(f"Estimated total time to completion: {estimated_total_time:.2f} hours")

    # Populate run data with profiling results and compute performance metrics
    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, SupportsStr]]:
        try:
            # Power profiling data
            try:
                power_df = pd.read_csv(context.run_dir / "powerjoular.csv")
            except FileNotFoundError:
                output.console_log("Power profiling data not found.")
                power_df = pd.DataFrame()

            # GPU profiling data
            try:
                gpu_df = pd.read_csv(context.run_dir / "nvidia-smi.csv")
                gpu_df.columns = gpu_df.columns.str.strip()
                gpu_utilization = gpu_df.get('utilization.gpu [%]', []).tolist()
                vram_usage = gpu_df.get('memory.used [MiB]', []).tolist()
            except FileNotFoundError:
                output.console_log("GPU profiling data not found.")
                gpu_utilization, vram_usage = [], []

            # CPU profiling data
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

            run_variation = context.run_variation.get("model_version")
            if not run_variation or run_variation not in model_configs:
                raise ValueError(f"Model configuration not found for run variation: {run_variation}")

            # Load model and tokenizer for evaluation
            model_name = model_configs[run_variation].get("model_name")
            tokenizer_name = model_configs[run_variation].get("tokenizer_name")

            if model_name is None or tokenizer_name is None:
                raise ValueError(f"Invalid model or tokenizer configuration for run variation: {run_variation}")

            output.console_log(f"Model name: {model_name}, Tokenizer name: {tokenizer_name}")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

            # Get task type and input size, and prepare prompt
            task_type = context.run_variation['task_type']
            input_size = context.run_variation['input_size']
            prompt = prompts[task_type][input_size]

            device = next(model.parameters()).device
            inputs = tokenizer(prompt['content'], return_tensors="pt", padding=False, truncation=True).to(device)

            # Generate output and calculate response time
            start_time = time.time()
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', None),
                max_new_tokens=100
            )
            end_time = time.time()

            generated_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.run_data["response_time"] = end_time - start_time
            self.run_data["output_text"] = generated_output

            # Evaluate performance using BLEU or ROUGE depending on task type
            expected_output = prompt.get('expected_output', None)

            if expected_output:
                if task_type == 'generation' or task_type == 'summarization':
                    # Using ROUGE for generation or summarization tasks
                    rouge_result = self.rouge_metric.compute(predictions=[generated_output], references=[expected_output])
                    self.run_data["performance_score"] = rouge_result['rougeL'].fmeasure
                    output.console_log(f"ROUGE score: {rouge_result['rougeL'].fmeasure:.4f}")
                elif task_type == 'question_answering':
                    # Using BLEU for QA tasks
                    bleu_result = self.bleu_metric.compute(predictions=[generated_output], references=[[expected_output]])
                    self.run_data["performance_score"] = bleu_result['bleu']
                    output.console_log(f"BLEU score: {bleu_result['bleu']:.4f}")
            else:
                output.console_log("No expected output found for evaluation. Skipping performance score calculation.")
                self.run_data["performance_score"] = None  # Set to None if no expected output is available

            # Return collected run data for reporting
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

    # After experiment is completed, log the total experiment duration
    def after_experiment(self) -> None:
        experiment_end_time = time.time()
        total_experiment_duration = experiment_end_time - self.experiment_start_time
        hours, remainder = divmod(total_experiment_duration, 3600)
        minutes, _ = divmod(remainder, 60)
        output.console_log(f"Total experiment duration: {int(hours)} hours and {int(minutes)} minutes.")
        output.console_log("Config.after_experiment() called!")

    # Cleanup method to terminate profiling processes if needed
    def cleanup(self):
        try:
            for profiler in [self.power_profiler, self.gpu_profiler, self.cpu_profiler]:
                if profiler and profiler.poll() is None:
                    profiler.terminate()
        except Exception as e:
            output.console_log(f"Error during cleanup: {e}")

# Main script entry point
if __name__ == "__main__":
    try:
        config = RunnerConfig()
        config.create_run_table_model()
    except Exception as e:
        config.cleanup()
        output.console_log(f"Error occurred during execution: {e}")
