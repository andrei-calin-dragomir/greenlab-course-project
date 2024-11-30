from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ExtendedTyping.Typing import SupportsStr
from ProgressManager.Output.OutputProcedure import OutputProcedure as output

from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath

import os
import gc
import time
import torch
import subprocess
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from evaluate import load as load_evaluation
from concurrent.futures import ThreadPoolExecutor

class WarmUp:
    def __init__(self, duration=60, matrix_size=1000):
        """
        Warms up the machine by performing computationally intensive tasks on both CPU and GPU simultaneously.
        
        Args:
            duration (int): Duration to run the warm-up (in seconds).
            matrix_size (int): Size of the square matrices for multiplication.
        """
        with ThreadPoolExecutor() as executor:
            tasks = []
            # Schedule CPU warm-up
            tasks.append(executor.submit(self.cpu_warm_up, matrix_size, duration))
            # Schedule GPU warm-up
            tasks.append(executor.submit(self.gpu_warm_up, matrix_size, duration))
            
            # Wait for all tasks to complete
            for task in tasks:
                task.result()
        
        print(f"Warm-up complete. Ran for {duration} seconds.")

    def cpu_warm_up(matrix_size, duration):
        """
        Warm up the CPU by performing matrix multiplication for the given duration.
        """
        start_time = time.time()
        cpu_matrix_a = np.random.rand(matrix_size, matrix_size)
        cpu_matrix_b = np.random.rand(matrix_size, matrix_size)
        while time.time() - start_time < duration:
            _ = np.dot(cpu_matrix_a, cpu_matrix_b)

    def gpu_warm_up(matrix_size, duration):
        """
        Warm up the GPU by performing matrix multiplication for the given duration.
        """
        if not torch.cuda.is_available():
            print("No GPU available. Skipping GPU warm-up.")
            return
        
        gpu_matrix_a = torch.rand(matrix_size, matrix_size, device='cuda')
        gpu_matrix_b = torch.rand(matrix_size, matrix_size, device='cuda')
        start_time = time.time()
        while time.time() - start_time < duration:
            _ = torch.mm(gpu_matrix_a, gpu_matrix_b)

class Model:
    # Load model based on context and model version
    def __init__(self, model: RunnerContext):
        # Model configurations
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

        model_name = model_configs[model]["model_name"]
        tokenizer_name = model_configs[model]["tokenizer_name"]
        self.model, self.tokenizer = None, None

        start_time = time.time()
        try:
            if "gemma" in model_name:
                self.model, self.tokenizer =  self._load_gemma_model(model_name, tokenizer_name)
            elif "mistral" in model_name:
                self.model, self.tokenizer =  self._load_mistral_model(model_name, tokenizer_name)
            elif "qwen" in model_name:
                self.model, self.tokenizer =  self._load_qwen_model(model_name, tokenizer_name)
            return start_time - time.time()
        # TODO Get right exception
        except RuntimeError as e:
            output.console_log_FAIL(f"RuntimeError during generation: {e}")
    
    # Finish this and return interaction metrics
    def run_inference(self, inference_task: dict):
        instruction = inference_task['instruction']
        content = inference_task['content']

        start_time = time.time()
        tokenized_content = self.tokenizer(content, return_tensors="pt", padding=False, truncation=True)
        tokenization_time = time.time()

        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        start_time = time.time()

        # Generate output
        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask', None),
            max_new_tokens=100
        )
        inference_time = start_time - time.time()

        return inference_time, tokenization_time, outputs

    def score_output(self, inference_output: str, inference_type: str, expected_output: str):
        pass

    # Load Gemma model based on the provided model and tokenizer names
    def _load_gemma_model(self, model_name: str, tokenizer_name: str):
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
    def _load_mistral_model(self, model_name: str, tokenizer_name: str):
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
    def _load_qwen_model(self, model_name: str, tokenizer_name: str):
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

def parse_energibridge_output(file_path, label: str = None):
    # Define target columns
    target_columns = [
        'GPU0_MEMORY_USED', 'GPU0_USAGE', 'USED_MEMORY', 'USED_SWAP',
        'TOTAL_MEMORY', 'TOTAL_SWAP', 'GPU0_MEMORY_TOTAL',
    ] + [f'CPU_USAGE_{i}' for i in range(32)]

    delta_target_columns = [
        'DRAM_ENERGY (J)', 'PACKAGE_ENERGY (J)', 'PP0_ENERGY (J)', 'PP1_ENERGY (J)', 'GPU0_POWER (mWatts)'
    ]

    # Read the file into a pandas DataFrame
    df = pd.read_csv(file_path).apply(pd.to_numeric, errors='coerce')

    # Calculate column-wise averages, ignoring NaN values and deltas from start of experiment to finish
    averages = df[target_columns].mean().to_dict()
    deltas = {f'{label}_{column}' if label else column : df[column].iloc[-1] - df[column].iloc[0]  for column in delta_target_columns}

    return dict(averages.items() | deltas.items())

class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name:                       str             = "llm_inference_experiment"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path:        Path            = ROOT_DIR / 'experiments'

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type:             OperationType   = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms:    int             = 60000

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
    def __init__(self):
        """Executes immediately after program start, on config load"""
        load_dotenv()
        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN       , self.before_run       ),
            (RunnerEvents.START_RUN        , self.start_run        ),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT         , self.interact         ),
            (RunnerEvents.STOP_MEASUREMENT , self.stop_measurement ),
            (RunnerEvents.STOP_RUN         , self.stop_run         ),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT , self.after_experiment )
        ])
        self.run_table_model = None  # Initialized later

        self.warmup_time                : int   = 60    # Seconds
        self.post_warmup_cooldown_time  : int   = 30    # Seconds

        # Prompts for different NLP tasks (generation, question-answering, summarization)
        self.input_prompts = {
            "generation": {
                "short": {
                    "instruction": "Generate a coherent and contextually appropriate completion for the sentence.",
                    "content": "Artificial intelligence has transformed industries by improving...",
                },
                "long": {
                    "instruction": "Expand upon the given paragraph with logical, evidence-based details or related concepts.",
                    "content": "The Industrial Revolution marked a pivotal moment in human history, with profound impacts on economies, societies, and the environment. One of the lasting consequences of this era is the rise in greenhouse gas emissions, contributing to global warming. Over the years, various international efforts, such as the Kyoto Protocol and the Paris Agreement, have aimed to address this issue. Continuing this discussion, provide a summary of the economic and technological advancements that have emerged as part of the response to climate change.",
                }
            },
            "question_answering": {
                "short": {
                    "instruction": "Provide a precise answer to the following factual question.",
                    "content": "What is the capital of France?",
                    "expected_output": "Paris",
                },
                "long": {
                    "instruction": "Analyze the provided context to generate an accurate and well-structured answer.",
                    "content": "Climate change is driven by the accumulation of greenhouse gases in the atmosphere, with carbon dioxide being the most significant contributor due to fossil fuel combustion. Other gases like methane and nitrous oxide also play substantial roles. What are the primary sources of these emissions, and how do they vary across different industries?",
                    "expected_output": "Primary sources include energy production, agriculture, transportation, and industrial processes, with variations depending on regional practices and technologies.",
                } 
            },
            "summarization": {
                "short": {
                    "instruction": "Summarize the main points from the following brief article.",
                    "content": "The rise of artificial intelligence (AI) in healthcare has opened new frontiers in diagnostics and treatment planning. Machine learning models trained on medical datasets can now predict patient outcomes with unprecedented accuracy. However, challenges remain, including ethical concerns about data privacy, potential biases in AI algorithms, and the need for robust regulatory frameworks. Addressing these issues is crucial for integrating AI into mainstream clinical practice.",
                },
                "long": {
                    "instruction": "Provide a concise summary of the key insights from the provided technical paper.",
                    "content": "The adoption of renewable energy sources has been a cornerstone of global strategies to combat climate change. Solar and wind power have seen remarkable growth due to technological advancements and decreasing costs. However, the intermittency of these sources poses a challenge for energy systems, necessitating the development of energy storage technologies and grid integration strategies. Policymakers have implemented incentives, such as tax credits and feed-in tariffs, to accelerate the transition. Nevertheless, achieving carbon neutrality will require a holistic approach, incorporating energy efficiency, sustainable infrastructure development, and international collaboration.",
                }
            }
        }
        
        self.run_model = None
        self.run_tokenizer = None

        output.console_log("Custom config loaded")

    def create_run_table_model(self) -> RunTableModel:
        """Create and return the run_table model here. A run_table is a List (rows) of tuples (columns),
        representing each run performed"""
        main_factor = FactorModel("model_version", ["gemma-v1", "gemma-v1.1", "gemma-v2", 
                                                    "mistral-v0.1", "mistral-v0.2", "mistral-v0.3", 
                                                    "qwen-v1", "qwen-v1.5", "qwen-v2", "qwen-v2.5"])
        blocking_factor_1 = FactorModel("task_type", ['generation', 'question_answering', 'summarization'])
        co_factor = FactorModel("input_size", ['short', 'long'])
        self.run_table_model = RunTableModel(
            factors=[main_factor, blocking_factor_1, co_factor],
            repetitions=20,
            exclude_variations=[],
            data_columns=[
                'inf_time', 'inf_GPU0_MEMORY_USED', 'inf_GPU0_USAGE', 'inf_USED_MEMORY', 'inf_USED_SWAP',
                'inf_DRAM_ENERGY (J)', 'inf_PACKAGE_ENERGY (J)', 'inf_PP0_ENERGY (J)', 'inf_PP1_ENERGY (J)', 'inf_GPU0_POWER (mWatts)',
                ] + [
                    f'inf_CPU_USAGE_{i}' for i in range(32)
                    ] + [
                        'load_time', 'load_GPU0_MEMORY_USED', 'load_GPU0_USAGE', 'load_USED_MEMORY', 'load_USED_SWAP',
                        'load_DRAM_ENERGY (J)', 'load_PACKAGE_ENERGY (J)', 'load_PP0_ENERGY (J)', 'load_PP1_ENERGY (J)', 'load_GPU0_POWER (mWatts)',
                        ] + [
                            f'load_CPU_USAGE_{i}' for i in range(32)
                            ] + [
                                'input_token_size', 'input_token_time', 'output', 'performance_score'
                                ] + [
                                    'TOTAL_MEMORY', 'TOTAL_SWAP', 'GPU0_MEMORY_TOTAL'
                                ]
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment here
        Invoked only once during the lifetime of the program."""
        output.console_log("Config.before_experiment() called!")

        #Run machine warmup based on specified interval for both CPU and GPU
        WarmUp(duration=self.warmup_time)

        # Cooldown machine
        time.sleep(self.post_warmup_cooldown_time)

        output.console_log_OK("Warmup finished. Experiment is starting now!")

    def before_run(self) -> None:
        """Perform any activity required before starting a run.
        No context is available here as the run is not yet active (BEFORE RUN)"""
        self.auxiliary_data = {
            'load_time' : 0,
            'inference_time' : 0,
            'performance_score' : 0,
            'input_token_size' : 0,
            'output' : '',
        }

        output.console_log("Config.before_run() called!")

    def start_run(self, context: RunnerContext) -> None:
        """Perform any activity required for starting the run here.
        For example, starting the target system to measure.
        Activities after starting the run should also be performed here."""
        output.console_log("Config.start_run() called!")

        # Run the command in the background
        self.energibridge_command = f"echo {os.getenv('PASSWORD')} | sudo -S energibridge -g --interval 200 --summary --output {context.run_dir}/energibridge_load.csv --command-output {context.run_dir}/output.txt sleep 100000"
        self.energibridge_pid = subprocess.Popen(self.energibridge_command).pid
        output.console_log(f"Energibridge for model loading started...")
        
        output.console_log_bold(f"Loading model: {context.run_variation['model_version']}")
        start_time = time.time()

        self.run_model = Model(context.run_variation["model_version"])

        self.auxiliary_data['load_time'] = start_time - time.time()
        #Stop energibridge
        subprocess.Popen(f"echo {os.getenv('PASSWORD')} | sudo -S kill {self.energibridge_pid}")
        output.console_log_OK(f'Model loading time: {self.auxiliary_data['load_time']}s')
        output.console_log("Energibridge collection stopped.")

    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""
        output.console_log("Config.start_measurement() called!")

        # Run the command in the background
        self.energibridge_command = f"echo {os.getenv('PASSWORD')} | sudo -S energibridge -g --interval 200 --summary --output {context.run_dir}/energibridge_inference.csv --command-output {context.run_dir}/output.txt sleep 100000"
        self.energibridge_pid = subprocess.Popen(self.energibridge_command).pid
        output.console_log(f"Energibridge collection for inference started...")

    def interact(self, context: RunnerContext) -> None:
        """Perform any interaction with the running target system here, or block here until the target finishes."""
        output.console_log("Config.interact() called!")
                
        inference_time, tokenization_time, output = self.run_model.run_inference(self.input_prompts[context.run_variation['task_type']][context.run_variation['input_size']])

        self.auxiliary_data['inference_time'] = inference_time
        self.auxiliary_data['tokenization_time'] = tokenization_time


        output.console_log_bold(f"Model tokenization time: {self.auxiliary_data['tokenization_time']}s")
        output.console_log_bold(f'Model inference time: {self.auxiliary_data['inference_time']}s')
        
        output.console_log_OK("Inference finished!")


    def stop_measurement(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping measurements."""
        output.console_log("Config.stop_measurement called!")

        #Stop energibridge
        subprocess.Popen(f"echo {os.getenv('PASSWORD')} | sudo -S kill {self.energibridge_pid}")
        output.console_log("Energibridge collection stopped.")

    def stop_run(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping the run.
        Activities after stopping the run should also be performed here."""
        output.console_log("Config.stop_run() called!")

        del self.run_model
        gc.collect()
        torch.cuda.empty_cache()

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, SupportsStr]]:
        """Parse and process any measurement data here.
        You can also store the raw measurement data under `context.run_dir`
        Returns a dictionary with keys `self.run_table_model.data_columns` and their values populated"""
        output.console_log("Config.populate_run_data() called!")

        load_data = parse_energibridge_output(f'{context.run_dir}/energibridge_load.csv', 'load')
        inf_data = parse_energibridge_output(f'{context.run_dir}/energibridge_inference.csv', 'inf')

        return dict(load_data.items() | inf_data.items() | self.auxiliary_data.items())

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""

        output.console_log("Config.after_experiment() called!")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None
