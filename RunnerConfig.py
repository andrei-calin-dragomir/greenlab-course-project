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
        if self.stdin:
            self.stdin.close()
        if self.stdout:
            self.stdout.close()
        if self.stderr:
            self.stderr.close()
        self.ssh.close()

def parse_energibridge_output(file_path):
    # Define target columns
    target_columns = [
        'GPU0_MEMORY_USED', 'GPU0_USAGE', 'USED_MEMORY', 'USED_SWAP',
    ] + [f'CPU_USAGE_{i}' for i in range(32)]

    delta_target_columns = [
        'DRAM_ENERGY (J)', 'PACKAGE_ENERGY (J)', 'PP0_ENERGY (J)', 'PP1_ENERGY (J)', 'GPU0_ENERGY (mJ)'
    ]

    # Read the file into a pandas DataFrame
    df = pd.read_csv(file_path).apply(pd.to_numeric, errors='coerce')

    # Calculate column-wise averages, ignoring NaN values and deltas from start of experiment to finish
    averages = df[target_columns].mean().to_dict()
    deltas = {column : df[column].iloc[-1] - df[column].iloc[0]  for column in delta_target_columns}

    return dict(averages.items() | deltas.items())

def score_inference_output(score_type : str, inference_output : str, expected_outputs : List[str]):
    evaluation = load_evaluation(score_type)
    scores = evaluation.compute(predictions=[inference_output], references=[expected_outputs])

    score = next(iter(scores.values()))

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
    name:                       str             = "inference_experiment"

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
                    "output_length": 100,
                    "expected_outputs": [
                        "Artificial intelligence has transformed industries by improving efficiency and enabling new innovations in fields like healthcare and transportation.",
                        "Industries have been transformed by artificial intelligence, which has boosted efficiency and innovation across various sectors.",
                        "AI has led to significant improvements in efficiency and innovation within industries such as healthcare and logistics."
                    ]
                },
                "long": {
                    "instruction": "Expand upon the given paragraph with logical, evidence-based details or related concepts.",
                    "content": "The Industrial Revolution marked a pivotal moment in human history, with profound impacts on economies, societies, and the environment. One of the lasting consequences of this era is the rise in greenhouse gas emissions, contributing to global warming. Over the years, various international efforts, such as the Kyoto Protocol and the Paris Agreement, have aimed to address this issue. Continuing this discussion, provide a summary of the economic and technological advancements that have emerged as part of the response to climate change.",
                    "output_length": 150,
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
                    "output_length": 150,
                    "expected_outputs": ['''The capitals of European countries are:
                                        Andorra - Andorra la Vella, Albania - Tirana, Austria - Vienna, Belarus - Minsk, Belgium - Brussels, Bosnia and Herzegovina - Sarajevo, Bulgaria - Sofia, Croatia - Zagreb, Cyprus - Nicosia, Czechia - Prague, Denmark - Copenhagen, Estonia - Tallinn, Finland - Helsinki, France - Paris.
                                        Georgia - Tbilisi, Germany - Berlin, Greece - Athens, Hungary - Budapest, Iceland - Reykjavik, Ireland - Dublin, Italy - Rome, Kosovo - Pristina, Latvia - Riga, Liechtenstein - Vaduz, Lithuania - Vilnius, Luxembourg - Luxembourg, Malta - Valletta, Moldova - Chisinau, Monaco - Monaco.
                                        Montenegro - Podgorica, Netherlands - Amsterdam, North Macedonia - Skopje, Norway - Oslo, Poland - Warsaw, Portugal - Lisbon, Romania - Bucharest, Russia - Moscow, San Marino - San Marino, Serbia - Belgrade, Slovakia - Bratislava, Slovenia - Ljubljana, Spain - Madrid, Sweden - Stockholm, Switzerland - Bern, Ukraine - Kyiv, UK - London, Vatican - Vatican City.''']
                },
                "long": {
                    "instruction": "Analyze the provided context to generate an accurate and well-structured answer.",
                    "content": "Climate change is driven by the accumulation of greenhouse gases in the atmosphere, with carbon dioxide being the most significant contributor due to fossil fuel combustion. Other gases like methane and nitrous oxide also play substantial roles. What are the primary sources of these emissions, and how do they vary across different industries?",
                    "output_length": 200,
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
                    "output_length": 50,
                    "content": "The adoption of renewable energy sources has been a cornerstone of global strategies to combat climate change. Solar and wind power have seen remarkable growth due to technological advancements and decreasing costs. However, the intermittency of these sources poses a challenge for energy systems, necessitating the development of energy storage technologies and grid integration strategies. Policymakers have implemented incentives, such as tax credits and feed-in tariffs, to accelerate the transition. Nevertheless, achieving carbon neutrality will require a holistic approach, incorporating energy efficiency, sustainable infrastructure development, and international collaboration.",
                    "expected_outputs": [
                        "AI in healthcare enhances diagnostics and treatment with accurate predictions but faces challenges in ethics, bias, and regulation.",
                        "Machine learning in healthcare improves outcome prediction but raises ethical, bias, and privacy concerns needing regulation.",
                        "AI is transforming healthcare diagnostics and treatment planning, though challenges like data privacy and algorithmic biases remain."
                    ]
                },
                "long": {
                    "instruction": "Provide a concise summary of the key insights from the provided technical paper.",
                    "content": "Artificial intelligence (AI), in its broadest sense, is intelligence exhibited by machines, particularly computer systems.  It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.[1] Such machines may be called AIs. Some high-profile applications of AI include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT, and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go). However, many AI applications are not perceived as AI: A lot of cutting edge AI has filtered into general applications, often without being called AI because once something becomes useful enough and common enough its not labeled AI anymore.[2][3] The various subfields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and support for robotics.[a] General intelligence—the ability to complete any task performable by a human on an at least equal level—is among the fields long-term goals.[4] To reach these goals, AI researchers have adapted and integrated a wide range of techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics.[b] AI also draws upon psychology, linguistics, philosophy, neuroscience, and other fields.[5] Artificial intelligence was founded as an academic discipline in 1956,[6] and the field went through multiple cycles of optimism,[7][8] followed by periods of disappointment and loss of funding, known as AI winter.[9][10] Funding and interest vastly increased after 2012 when deep learning outperformed previous AI techniques.[11] This growth accelerated further after 2017 with the transformer architecture,[12] and by the early 2020s hundreds of billions of dollars were being invested in AI (known as the AI boom). The widespread use of AI in the 21st century exposed several unintended consequences and harms in the present and raised concerns about its risks and long-term effects in the future, prompting discussions about regulatory policies to ensure the safety and benefits of the technology.",
                    "output_length": 200,
                    "expected_outputs": [
                        "Renewable energy growth has been driven by lower costs and technology, but challenges like intermittency require storage and grid strategies. Policies like tax credits help, and carbon neutrality needs global collaboration.",
                        "Solar and wind energy are growing but require solutions for intermittency, such as storage and integration. Incentives like feed-in tariffs support this, while carbon neutrality demands international action.",
                        "The transition to renewable energy is advancing with policy incentives, storage innovations, and global collaboration, yet challenges like intermittency remain key to address."
                    ]
                }
            }
        }
        self.models = ["llama2", "llama3", "llama3.1", "mistral:v0.1", "mistral:v0.2", "mistral:v0.3", "qwen:7b", "qwen2", "qwen2.5", "phi", "phi3", "phi3.5", "gemma", "gemma2"]
        
        self.metric_capturing_interval  : int   = 200 # Miliseconds

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
            repetitions=20,
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
        output.console_log(ssh.stdout.readline())
        # Set GPU frequency during usage
        ssh.execute_remote_command(f"echo {getenv('PASSWORD')} | sudo -S nvidia-smi -i 0 -lgc {self.gpu_clock}")
        output.console_log(ssh.stdout.readline())
        # Set GPU maximum power draw
        ssh.execute_remote_command(f"echo {getenv('PASSWORD')} | sudo -S nvidia-smi -i 0 -pl {self.gpu_power_cap}")
        output.console_log(ssh.stdout.readline())
        output.console_log_OK(f'GPU configuration completed!')

        output.console_log('Installing models...')
        ssh.execute_remote_command(f"./{self.project_name}/install_models.sh {','.join(self.models)}")
        machine_output = ''
        while 'Model installation process completed!' not in machine_output:
            machine_output = ssh.stdout.readline()
            output.console_log(f'Installation: {machine_output}...')
        output.console_log_OK('Model installation process completed!')

    def before_run(self) -> None:
        """Perform any activity required before starting a run.
        No context is available here as the run is not yet active (BEFORE RUN)"""
        output.console_log("Config.before_run() called!")
        self.inference_time = 0
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
        ssh.execute_remote_command(f"echo Respond with LOADED | ollama run {context.run_variation['model_version']}")
        output.console_log_bold(f"{ssh.stdout.readline().strip()} model: {context.run_variation['model_version']}")

    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""
        output.console_log("Config.start_measurement() called!")

        # Run the energibridge command in the background
        ssh = ExternalMachineAPI()
        energibridge_path = f'./{self.project_name}/EnergiBridge/target/release/energibridge'
        ssh.execute_remote_command(f"echo {getenv('PASSWORD')} | sudo -S {energibridge_path} -g --interval {self.metric_capturing_interval} --output {self.external_run_dir}/energibridge.csv sleep 600 & echo $!")
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
        output.console_log(prompt)
        ssh = ExternalMachineAPI()
        # Running inference task
        start_time = time.time()
        ssh.execute_remote_command(f'echo "{prompt}" | ollama run {context.run_variation["model_version"]}')
        raw_output = ssh.stdout.readlines()
        self.inference_time = time.time() - start_time
        self.inference_output = ''.join(raw_output)
        output.console_log(self.inference_output)
        output.console_log_OK(f"Inference finished in {self.inference_time}s")


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

        bleu_scores = score_inference_output('bleu', self.inference_output, expected_outputs)
        rouge_scores = score_inference_output('rouge', self.inference_output, expected_outputs)
        
        run_data = parse_energibridge_output(f'{context.run_dir}/energibridge.csv')
        run_data['inference_time'] = self.inference_time
        run_data['rouge_scores'] = rouge_scores
        run_data['bleu_scores'] = bleu_scores

        return run_data

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""

        ssh = ExternalMachineAPI()
        ssh.execute_remote_command(f"echo {getenv('PASSWORD')} | sudo nvidia-smi -i 0 -rgc")
        ssh.execute_remote_command(f"echo {getenv('PASSWORD')} | sudo nvidia-smi -pl 200")

        output.console_log("Config.after_experiment() called!")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None
