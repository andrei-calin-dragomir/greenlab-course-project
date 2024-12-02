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

import paramiko
import pandas as pd
from scp import SCPClient
from os import getenv
from dotenv import load_dotenv
from evaluate import load as load_evaluation
import time
load_dotenv()

class ExternalMachineAPI:
    def __init__(self):

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.stdin = None
        self.stdout = None
        self.stderr = None
        
        try:
            self.ssh.connect(hostname=getenv('HOSTNAME'), username=getenv('USERNAME'), password=getenv('PASSWORD'))
        except paramiko.SSHException:
            print('Failed to send run command to machine!')

    def execute_remote_command(self, command : str = '', overwrite_channels : bool = True):
        try:
            # Execute the command
            if overwrite_channels:
                self.stdin, self.stdout, self.stderr = self.ssh.exec_command(command)
            else:
                self.ssh.exec_command(command)
        except paramiko.SSHException:
            print('Failed to send run command to machine!')
        except TimeoutError:
            print('Timeout reached while waiting for command output.')

    def copy_file_from_remote(self, remote_path, local_path):
        # Create SSH client and SCP client
        with SCPClient(self.ssh.get_transport()) as scp:
            # Copy the file from remote to local
            scp.get(remote_path, local_path, recursive=True)
        print(f"Copied {remote_path} to {local_path}")

    def __del__(self):
        self.stdin.close()
        self.stdout.close()
        self.stderr.close()
        self.ssh.close()

def parse_energibridge_output(file_path):
    # Define target columns
    target_columns = [
        'GPU0_MEMORY_USED', 'GPU0_USAGE', 'USED_MEMORY', 'USED_SWAP',
    ] + [f'CPU_USAGE_{i}' for i in range(32)]

    delta_target_columns = [
        'DRAM_ENERGY (J)', 'PACKAGE_ENERGY (J)', 'PP0_ENERGY (J)', 'PP1_ENERGY (J)', 'GPU0_POWER (mWatts)'
    ]

    # Read the file into a pandas DataFrame
    df = pd.read_csv(file_path).apply(pd.to_numeric, errors='coerce')

    # Calculate column-wise averages, ignoring NaN values and deltas from start of experiment to finish
    averages = df[target_columns].mean().to_dict()
    deltas = {column : df[column].iloc[-1] - df[column].iloc[0]  for column in delta_target_columns}

    return dict(averages.items() | deltas.items())

def score_inference_output(task_type : str, inference_output : str, expected_outputs : List[str]):
    evaluation = load_evaluation(task_type)

    scores = evaluation.compute(predictions=[inference_output], references=[expected_outputs])

    score = scores[task_type] if task_type == 'bleu' else scores['rouge1']

    if score <= 0.4:
        output.console_log_FAIL(f"Performance Score: {score:.4f}")
    elif 0.4 < score <= 0.8:
        output.console_log_bold(f"Performance Score: {score:.4f}")
    else:
        output.console_log_OK(f"Performance Score: {score:.4f}")

    return scores

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
    time_between_runs_in_ms:    int             = 1000 #60000

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
    def __init__(self):
        """Executes immediately after program start, on config load"""
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
        self.project_name = 'greenlab-course-project'
        self.input_prompts = {
            "generation": {
                "short": {
                    "instruction": "Generate a coherent and contextually appropriate completion for the sentence.",
                    "content": "Artificial intelligence has transformed industries by improving...",
                    "metric": "bleu",
                    "output_length": 50,
                    "expected_outputs": [
                        "Artificial intelligence has transformed industries by improving efficiency and enabling new innovations in fields like healthcare and transportation.",
                        "Industries have been transformed by artificial intelligence, which has boosted efficiency and innovation across various sectors.",
                        "AI has led to significant improvements in efficiency and innovation within industries such as healthcare and logistics."
                    ]
                },
                "long": {
                    "instruction": "Expand upon the given paragraph with logical, evidence-based details or related concepts.",
                    "content": "The Industrial Revolution marked a pivotal moment in human history, with profound impacts on economies, societies, and the environment. One of the lasting consequences of this era is the rise in greenhouse gas emissions, contributing to global warming. Over the years, various international efforts, such as the Kyoto Protocol and the Paris Agreement, have aimed to address this issue. Continuing this discussion, provide a summary of the economic and technological advancements that have emerged as part of the response to climate change.",
                    "metric": "bleu",
                    "output_length": 50,
                    "expected_outputs": [
                        "Economic advancements include growth in renewable energy industries, innovations in carbon capture technologies, and green jobs. Technological progress includes breakthroughs in solar power, wind energy, and sustainable transport solutions.",
                        "The response to climate change has spurred technological advancements such as renewable energy systems and energy storage, alongside economic measures like carbon pricing and green infrastructure investment.",
                        "Advancements include renewable energy technologies, energy efficiency measures, and policies encouraging carbon reduction, such as emissions trading systems."
                    ]
                }
            },
            "question_answering": {
                "short": {
                    "instruction": "Provide a precise answer to the following factual question.",
                    "content": "What are the capitals of all european countries?",
                    "metric": "bleu",
                    "output_length": 150,
                    "expected_outputs": ['''The capitals of European countries are:
                                        Andorra - Andorra la Vella, Albania - Tirana, Austria - Vienna, Belarus - Minsk, Belgium - Brussels, Bosnia and Herzegovina - Sarajevo, Bulgaria - Sofia, Croatia - Zagreb, Cyprus - Nicosia, Czechia - Prague, Denmark - Copenhagen, Estonia - Tallinn, Finland - Helsinki, France - Paris.
                                        Georgia - Tbilisi, Germany - Berlin, Greece - Athens, Hungary - Budapest, Iceland - Reykjavik, Ireland - Dublin, Italy - Rome, Kosovo - Pristina, Latvia - Riga, Liechtenstein - Vaduz, Lithuania - Vilnius, Luxembourg - Luxembourg, Malta - Valletta, Moldova - Chisinau, Monaco - Monaco.
                                        Montenegro - Podgorica, Netherlands - Amsterdam, North Macedonia - Skopje, Norway - Oslo, Poland - Warsaw, Portugal - Lisbon, Romania - Bucharest, Russia - Moscow, San Marino - San Marino, Serbia - Belgrade, Slovakia - Bratislava, Slovenia - Ljubljana, Spain - Madrid, Sweden - Stockholm, Switzerland - Bern, Ukraine - Kyiv, UK - London, Vatican - Vatican City.''']
                },
                "long": {
                    "instruction": "Analyze the provided context to generate an accurate and well-structured answer.",
                    "content": "Climate change is driven by the accumulation of greenhouse gases in the atmosphere, with carbon dioxide being the most significant contributor due to fossil fuel combustion. Other gases like methane and nitrous oxide also play substantial roles. What are the primary sources of these emissions, and how do they vary across different industries?",
                    "metric": "rouge",
                    "output_length": 150,
                    "expected_outputs": [
                        "Greenhouse gases primarily come from energy production, agriculture, transportation, and industrial processes, with variations based on regional energy systems and practices.",
                        "Key sources include fossil fuel combustion in energy production, methane emissions in agriculture, and transportation emissions from vehicles. Industrial processes also contribute, varying regionally.",
                        "Primary emissions sources are fossil fuel energy, agricultural practices like livestock farming, and transportation, all varying depending on local technologies and economies."
                    ]
                }
            },
            "summarization": {
                "short": {
                    "instruction": "Summarize the main points from the following brief article.",
                    "content": "The rise of artificial intelligence (AI) in healthcare has opened new frontiers in diagnostics and treatment planning. Machine learning models trained on medical datasets can now predict patient outcomes with unprecedented accuracy. However, challenges remain, including ethical concerns about data privacy, potential biases in AI algorithms, and the need for robust regulatory frameworks. Addressing these issues is crucial for integrating AI into mainstream clinical practice.",
                    "metric": "rouge",
                    "output_length": 50,
                    "expected_outputs": [
                        "AI in healthcare enhances diagnostics and treatment with accurate predictions but faces challenges in ethics, bias, and regulation.",
                        "Machine learning in healthcare improves outcome prediction but raises ethical, bias, and privacy concerns needing regulation.",
                        "AI is transforming healthcare diagnostics and treatment planning, though challenges like data privacy and algorithmic biases remain."
                    ]
                },
                "long": {
                    "instruction": "Provide a concise summary of the key insights from the provided technical paper.",
                    "content": "The adoption of renewable energy sources has been a cornerstone of global strategies to combat climate change. Solar and wind power have seen remarkable growth due to technological advancements and decreasing costs. However, the intermittency of these sources poses a challenge for energy systems, necessitating the development of energy storage technologies and grid integration strategies. Policymakers have implemented incentives, such as tax credits and feed-in tariffs, to accelerate the transition. Nevertheless, achieving carbon neutrality will require a holistic approach, incorporating energy efficiency, sustainable infrastructure development, and international collaboration.",
                    "metric": "rouge",
                    "output_length": 50,
                    "expected_outputs": [
                        "Renewable energy growth has been driven by lower costs and technology, but challenges like intermittency require storage and grid strategies. Policies like tax credits help, and carbon neutrality needs global collaboration.",
                        "Solar and wind energy are growing but require solutions for intermittency, such as storage and integration. Incentives like feed-in tariffs support this, while carbon neutrality demands international action.",
                        "The transition to renewable energy is advancing with policy incentives, storage innovations, and global collaboration, yet challenges like intermittency remain key to address."
                    ]
                }
            }
        }
        self.models = ["qwen:7b", "qwen2:7b", "qwen2.5:7b", "mistral:v0.1", "mistral:v0.2", "mistral:v0.3", "qwen:7b", "qwen2", "qwen2.5"]
        
        self.warmup_time                : int   = 10 #60    # Seconds
        self.post_warmup_cooldown_time  : int   = 10 #30   # Seconds
        self.metric_capturing_interval  : int   = 100 #Miliseconds

        self.gpu_clock : int = 300 #Mhz
        self.gpu_power_cap : int = 100 #Watts

        output.console_log("Custom config loaded")

    def create_run_table_model(self) -> RunTableModel:
        """Create and return the run_table model here. A run_table is a List (rows) of tuples (columns),
        representing each run performed"""
        main_factor = FactorModel("model_version", self.models)
        blocking_factor_1 = FactorModel("task_type", ['generation', 'question_answering', 'summarization'])
        co_factor = FactorModel("input_size", ['short', 'long'])
        self.run_table_model = RunTableModel(
            factors=[main_factor, blocking_factor_1, co_factor],
            repetitions=1, #20
            data_columns=[
                'GPU0_MEMORY_USED', 'GPU0_USAGE', 'USED_MEMORY', 'USED_SWAP',
                'DRAM_ENERGY (J)', 'PACKAGE_ENERGY (J)', 'PP0_ENERGY (J)', 'PP1_ENERGY (J)', 'GPU0_POWER (mWatts)',
                ] + [f'CPU_USAGE_{i}' for i in range(32)] + ['performance_scores', 'inference_time']
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment here
        Invoked only once during the lifetime of the program."""
        output.console_log("Config.before_experiment() called!")
        ssh = ExternalMachineAPI()
        
        output.console_log(f'Setting up GPU frequency at {self.gpu_clock}Mhz and maximum power draw at {self.gpu_power_cap}W...')
        # Set persistence 
        ssh.execute_remote_command(f"echo {getenv('PASSWORD')} | sudo -S nvidia-smi -i 0 -pm 1")
        # Set GPU frequency during usage
        ssh.execute_remote_command(f"echo {getenv('PASSWORD')} | sudo -S nvidia-smi -i 0 -lgc {self.gpu_clock}")
        # Set GPU maximum power draw
        ssh.execute_remote_command(f"echo {getenv('PASSWORD')} | sudo -S nvidia-smi -i 0 -pl {self.gpu_power_cap}")
        output.console_log_OK(f'GPU configuration completed!')

        output.console_log('Installing models...')
        ssh.execute_remote_command(f"./{self.project_name}/install_models.sh {','.join(self.models)}")
        machine_output = None
        while not machine_output == 'Model installation process completed!':
            machine_output = ssh.stdout.readline()
            output.console_log(f'Installation: {machine_output}...')
        output.console_log_OK('Model installation process completed!')
        
        #Run machine warmup based on specified interval for both CPU and GPU
        output.console_log(f'Warming up machine using warmup script for {self.warmup_time}s...')
        ssh.execute_remote_command(f'./{self.project_name}/warmup.sh {self.warmup_time}')

        #Wait for warmup to finish and then cooldown machine
        time.sleep(self.warmup_time + self.post_warmup_cooldown_time)
        output.console_log_OK('Warmup complete.')

    def before_run(self) -> None:
        """Perform any activity required before starting a run.
        No context is available here as the run is not yet active (BEFORE RUN)"""
        output.console_log("Config.before_run() called!")
        self.inference_time = 0,
        self.inference_output = ''
        

    def start_run(self, context: RunnerContext) -> None:
        """Perform any activity required for starting the run here.
        For example, starting the target system to measure.
        Activities after starting the run should also be performed here."""
        output.console_log("Config.start_run() called!")
        ssh = ExternalMachineAPI()
        

        # Make directory of run on experimental machine
        self.external_run_dir = f'./{self.project_name}/experiments/{self.name}/{context.run_dir.name}'
        ssh.execute_remote_command(f"echo {getenv('PASSWORD')} | sudo -S mkdir -p {self.external_run_dir}")
        output.console_log(f'Run directory on experimental machine: {self.external_run_dir}')

        # Loading the model of the run
        ssh.execute_remote_command(f"ollama run {context.run_variation['model_version']}")
        output.console_log_bold(f'Loaded model: {context.run_variation['model_version']}')

    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""
        output.console_log("Config.start_measurement() called!")

        # Run the energibridge command in the background
        ssh = ExternalMachineAPI()
        ssh.execute_remote_command(f"echo {getenv('PASSWORD')} | sudo -S energibridge --interval {self.metric_capturing_interval} --output {self.external_run_dir}/energibridge.csv sleep 600 & echo $!")
        self.energibridge_pid = int(ssh.stdout.readline())

        output.console_log(f"Energibridge collection for inference started...")

    def interact(self, context: RunnerContext) -> None:
        """Perform any interaction with the running target system here, or block here until the target finishes."""
        output.console_log("Config.interact() called!")

        input_size = context.run_variation['input_size']
        task_type = context.run_variation['task_type']

        output.console_log(f"Running inference for a {task_type} task with {input_size} input...")        
        prompting_data = self.input_prompts[task_type][input_size]
        maximum_output_prompt = f"You must respond in {prompting_data['output_length']} word(s)."

        prompt = prompting_data['instruction'] + prompting_data['content'] + maximum_output_prompt

        ssh = ExternalMachineAPI()
        # Running inference task
        start_time = time.time()
        ssh.execute_remote_command(f"echo {prompt} | ollama run qwen:7b")
        raw_output = ssh.stdout.readlines()
        self.inference_time = time.time() - start_time
        self.inference_output = ''.join(raw_output)
        output.console_log_OK(f"Inference finished in {self.auxiliary_data['inference_time']}!")


    def stop_measurement(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping measurements."""
        output.console_log("Config.stop_measurement called!")
        ssh = ExternalMachineAPI()

        # Stop energibridge
        ssh.execute_remote_command(f"echo {getenv('PASSWORD')} | sudo -S kill {self.energibridge_pid}")
        output.console_log("Energibridge collection stopped.")

    def stop_run(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping the run.
        Activities after stopping the run should also be performed here."""
        output.console_log("Config.stop_run() called!")
        ssh = ExternalMachineAPI()

        # Stop current model
        output.console_log_bold('Stopping run model...')
        ssh.execute_remote_command(f"ollama stop {context.run_variation['model_version']}")

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, SupportsStr]]:
        """Parse and process any measurement data here.
        You can also store the raw measurement data under `context.run_dir`
        Returns a dictionary with keys `self.run_table_model.data_columns` and their values populated"""
        output.console_log("Config.populate_run_data() called!")
        ssh = ExternalMachineAPI()
        ssh.copy_file_from_remote(f'{self.external_run_dir}/energibridge.csv', context.run_dir)

        # Store output in a file
        with open(f"{context.run_dir}/output.txt", "w") as file:
            file.write(self.inference_output)

        task_type = context.run_variation['task_type']
        expected_outputs = self.input_prompts[task_type][context.run_variation['input_size']]['expected_outputs']

        inference_scores = score_inference_output(task_type, self.inference_output, expected_outputs)
        load_data = parse_energibridge_output(f'{context.run_dir}/energibridge.csv')

        return dict(load_data.items() | {'inference_time' : self.inference_time, 'performance_scores' : inference_scores})

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""

        # Reset GPU with sudo nvidia-smi -i 0 -rgc and sudo nvidia-smi -pl 200

        output.console_log("Config.after_experiment() called!")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None
