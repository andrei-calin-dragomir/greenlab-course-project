import os
import subprocess
import psutil
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
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

class RunnerConfig:
    ROOT_DIR = Path("../data/")  # Root directory for storing data
    name: str = "test_runner_experiment"  # Name of the experiment
    results_output_path: Path = ROOT_DIR / 'experiments'  # Path where results will be stored
    operation_type: OperationType = OperationType.AUTO  # Operation type for automatic execution
    time_between_runs_in_ms: int = 1000 * 5  # 5 seconds between runs
    repetitions: int = 30  # Number of repetitions for the experiment runs
    
    def __init__(self):
        # Subscribe to different lifecycle events of the experiment
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
        self.run_table_model = None  # Placeholder for run table model, to be initialized later
        output.console_log("Custom config loaded")

    def before_experiment(self) -> None:
        # Install dependencies before starting the experiment
        print("Installing dependencies...")
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
        self.experiment_start_time = time.time()  # Track total experiment start time
        output.console_log("Experiment started.")
        output.console_log("Config.before_experiment() called!")

    def before_run(self) -> None:
        # Called before each run of the experiment
        output.console_log("Config.before_run() called!")

    def start_run(self, context: RunnerContext) -> None:
        # Initialize data for each run and track the start time
        self.run_start_time = time.time()  # Track the start time for each individual run
        context.run_data = {}  # Initialize run_data as an empty dictionary
        output.console_log("Config.start_run() called!")

    def start_measurement(self, context: RunnerContext) -> None:
        # Start measurement by querying GPU usage and memory utilization
        output.console_log("Config.start_measurement() called!")
        context.run_data['gpu_utilization'] = subprocess.getoutput("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
        context.run_data['vram_usage'] = subprocess.getoutput("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")

    def create_run_table_model(self) -> RunTableModel:
        # Create a model to represent the configuration of different experiment runs
        main_factor = FactorModel("model_version", list(model_configs.keys()))  # Main factor representing all model versions
        blocking_factor_1 = FactorModel("candidate_family", ['qwen', 'gemma', 'mistral'])  # Blocking factor to specify candidate family
        blocking_factor_2 = FactorModel("task_type", ['generation'])  # Blocking factor to specify task type
        co_factor = FactorModel("input_size", ['short'])  # Co-factor to specify input size
        # Defining the run table with repetitions and the data columns to collect
        self.run_table_model = RunTableModel(
            factors=[main_factor, blocking_factor_1, blocking_factor_2, co_factor],
            repetitions=self.repetitions,
            data_columns=['cpu_utilization', 'ram_usage', 
                          'gpu_utilization', 'vram_usage',
                          'performance_score', 'performance_score_type',
                          'response_time', 'input_token_size', 'output_token_size',
                          'energy_consumption']
        )
        return self.run_table_model

    def load_model(self, context: RunnerContext):
        # Load the model and tokenizer based on the current run configuration
        run_variation = context.run_variation["model_version"]  # Get the run variation, e.g., "qwen-v1"

        # Fetch model and tokenizer configurations from the dictionary
        if run_variation in model_configs:
            model_name = model_configs[run_variation]["model_name"]
            tokenizer_name = model_configs[run_variation]["tokenizer_name"]
            print(f"Loading model: {model_name}, tokenizer: {tokenizer_name}")
            # Load the model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            return model, tokenizer
        else:
            # Raise an error if the model configuration is not found
            raise ValueError(f"Model configuration not found for run variation: {run_variation}")

    def interact(self, context: RunnerContext) -> None:
        # Perform interaction with the model by providing an input text
        output.console_log("Config.interact() called!")
        input_text = "What is the capital of France?"
        model, tokenizer = self.load_model(context)  # Load model and tokenizer
        
        inputs = tokenizer(input_text, return_tensors="pt")  # Tokenize the input text
        start_time = time.time()  # Track the start time for generating output
        outputs = model.generate(**inputs, max_length=50)  # Generate output using the model
        end_time = time.time()  # Track the end time for generating output
        
        # Store the response time and the generated output in the run data
        context.run_data["response_time"] = end_time - start_time
        context.run_data["output_text"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output.console_log(f"Generated output: {context.run_data['output_text']}")

    def stop_measurement(self, context: RunnerContext) -> None:
        # Called to stop any measurements after the run
        output.console_log("Config.stop_measurement called!")

    def stop_run(self, context: RunnerContext) -> None:
        # Called after completing each run, calculates total run time
        run_end_time = time.time()
        total_run_time = run_end_time - self.run_start_time
        # Estimate total time required for all runs based on the time taken for this run
        estimated_total_time = ((total_run_time + 60) * self.repetitions) / 60 / 60  # Estimated hours
        
        # Calculate performance score based on response time and GPU utilization
        gpu_utilization = float(context.run_data.get('gpu_utilization', 0))
        response_time = context.run_data.get('response_time', 1)
        context.run_data['performance_score'] = 1000 / (response_time * (gpu_utilization / 100 + 1))
        
        # Estimate energy consumption (assuming power ratings for GPU and CPU)
        gpu_power = 300  # Watts for GPU
        cpu_power = 100  # Watts for CPU
        gpu_utilization_percent = float(context.run_data.get('gpu_utilization', 0)) / 100
        cpu_utilization_percent = psutil.cpu_percent() / 100
        context.run_data['energy_consumption'] = (
            ((gpu_power * gpu_utilization_percent) + (cpu_power * cpu_utilization_percent)) * total_run_time
        ) / 3600  # in kWh
        
        output.console_log(f"Run completed in {total_run_time:.2f} seconds.")
        output.console_log(f"Estimated total time to completion: {estimated_total_time:.2f} hours")
        output.console_log(f"Performance score: {context.run_data['performance_score']:.2f}")
        output.console_log(f"Estimated energy consumption: {context.run_data['energy_consumption']:.4f} kWh")
        output.console_log("Config.stop_run() called!")

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, SupportsStr]]:
        # Populate data collected during the run for further analysis
        cpu_usage = psutil.cpu_percent()  # Get current CPU usage
        ram_usage = psutil.virtual_memory().percent  # Get current RAM usage
        return {
            "cpu_utilization": cpu_usage,
            "ram_usage": ram_usage,
            "gpu_utilization": context.run_data.get('gpu_utilization'),
            "vram_usage": context.run_data.get('vram_usage'),
            "response_time": context.run_data.get('response_time'),
            "performance_score": context.run_data.get('performance_score'),
            "energy_consumption": context.run_data.get('energy_consumption')
        }

    def after_experiment(self) -> None:
        # Called after the entire experiment is completed to log the total duration
        experiment_end_time = time.time()
        total_experiment_duration = experiment_end_time - self.experiment_start_time
        hours, remainder = divmod(total_experiment_duration, 3600)
        minutes, _ = divmod(remainder, 60)
        output.console_log(f"Total experiment duration: {int(hours)} hours and {int(minutes)} minutes.")
        output.console_log("Config.after_experiment() called!")

if __name__ == "__main__":
    # Instantiate the RunnerConfig and create the run table model
    config = RunnerConfig()
    config.create_run_table_model()
