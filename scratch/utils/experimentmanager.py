import configparser
import argparse
import os
from datetime import datetime


class ExperimentManager:
    '''
    Singleton pattern class used to manage experiment settings.
    '''

    _instance = None  # used to define Singleton pattern.

    def __init__(self) -> None:

        # Set the absolute path to the main directory
        self._main_dir_name = "work"  # NOTE: change to 'work' for DL-28
        self._main_dir_path = os.path.dirname(os.path.realpath(__file__))
        depth = 0
        while os.path.basename(self._main_dir_path) != self._main_dir_name:
            depth += 1
            if depth == 10:
                raise Exception("Main folder not found.")
            self._main_dir_path = os.path.dirname(self._main_dir_path)

        # Creates an environment parser and read the config file
        env_cfg_path = os.path.join(
            self._main_dir_path, 'Configs/init/environment.cfg')

        assert os.path.isfile(
            env_cfg_path), f"{os.path.realpath(env_cfg_path)} is not a valid file!"

        self.env_parser = configparser.ConfigParser()
        self.env_parser.read(env_cfg_path)

        # init time
        self._init_time = self.currentTime()

        # Handling modifications received from the command line;
        self._argparser = argparse.ArgumentParser()
        self._argparser.add_argument('-s', '--strategy')
        self._argparser.add_argument('-b', '--benchmark')
        self._argparser.add_argument('--training_epochs', type=int, default=6)
        
        self._argparser.add_argument('--batch_size', type=int, default=32)
        self._argparser.add_argument('--weight_decay', type=float, default=1e-4)
        self._argparser.add_argument('--learning_rate', type=float, default=1e-3)
        self._argparser.add_argument('--plasticity_factor', type=float, default=0.9)
        self._argparser.add_argument('--save', type=bool, default=True)

        self._args = self._argparser.parse_args()

        self.save_experiment = False  # TODO: turn into a property
        if self._args.save:
            self.save_experiment = True

    def __new__(cls):
        '''
        Implements the Singleton pattern.
        '''
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def currentTime(self) -> str:
        return str(datetime.now())[:-7]  # Ignore miliseconds

    def read_experiment(self, cfg_path: str) -> None:
        '''
            Creates an experiment parser from the specified cfg_path.
        '''

        assert os.path.isfile(
            cfg_path), f"{os.path.realpath(cfg_path)} is not a valid file!"

        self.exp_parser = configparser.ConfigParser()
        self.exp_parser.read(cfg_path)
    
        # reads strategy from experiment config file or from argument line command

        #commenting line to adequate to designed architecture
        '''
        if self._args.strategy:
            self._read_strategy_from_file(self._args.strategy)
        else:
            self._read_strategy_from_file(
                self.exp_parser.get('experiment', 'strategy'))
        '''
                

        # changes the number of epochs if specified on the command line.
        if not self.exp_parser.has_section('training'):
            self.exp_parser.add_section('training')

        if self._args.training_epochs:
            self.exp_parser.set('training', 'epochs',
                                str(self._args.training_epochs))
            
        if self._args.training_epochs:
            self.exp_parser.set('training', 'plasticity_factor',
                                str(self._args.plasticity_factor))

        if self._args.batch_size:
            self.exp_parser.set('training', 'batch_size',
                                str(self._args.batch_size))
            
        if self._args.weight_decay:
            self.exp_parser.set('training', 'weight_decay',
                                str(self._args.weight_decay))
            
        if self._args.learning_rate:
            self.exp_parser.set('training', 'learning_rate',
                                str(self._args.learning_rate))
            
        
        # reads benchmark from experiment config file or from argument line command
            
        #commenting line to adequate to designed architecture
        '''
        if self._args.benchmark:
            self._read_benchmark_from_file(self._args.benchmark)
        else:
            self._read_benchmark_from_file(
                self.exp_parser.get('experiment', 'benchmark'))
        '''

        # create folders to save experiment info
        if self._args.save:
            self._init_exp_folder()

        return

    def _read_strategy_from_file(self, strategy: str) -> None:
        # read experiment's strategy from a cfg file.

        strategy_path = os.path.join(self.get_dir_path(
            'strategies'), f"{strategy}.cfg")
        assert os.path.isfile(
            strategy_path), f'{strategy_path} is not a valid config file!'

        self.exp_parser.read(os.path.join(self.get_dir_path(
            'strategies'), strategy_path))
        return

    def _read_benchmark_from_file(self, benchmark: str) -> None:
        # read experiment's benchmark from a cfg file.

        benchmark_path = os.path.join(self.get_dir_path(
            'benchmarks'), f"{benchmark}.cfg")
        assert os.path.isfile(
            benchmark_path), f'{benchmark_path} is not a valid config file!'

        self.exp_parser.read(os.path.join(self.get_dir_path(
            'strategies'), benchmark_path))
        return

    def _init_exp_folder(self):
        self._exp_folder = os.path.join(self.get_dir_path("results"),
                                        self.exp_parser.get(
            "benchmark", "name"),
            self.exp_parser.get(
            "strategy", "name"),
            self._init_time,)

        os.makedirs(self._exp_folder)

        # Create the directories to save the results
        # os.makedirs(os.path.join(results_path, self.env_parser.get(
        #     "results", "graphs_directory")))
        # os.makedirs(os.path.join(
        #     results_path, self.env_parser.get("results", "logs_directory")))
        # os.makedirs(os.path.join(
        #     results_path, self.env_parser.get("results", "model_directory")))

        # # Save the experiment settings
        # cfg_path = os.path.join(results_path, self.getstr("exp_settings_file"))
        # with open(cfg_path, 'w') as cfgfile:
        #         self.__tag.write(cfgfile)

    def set_scenario_params(self, param_dict):
        if not self.exp_parser.has_section('scenario_configs'):
            self.exp_parser.add_section('scenario_configs')

        for i in param_dict:
            self.exp_parser.set(section='scenario_configs', option=i,
                            value= str(param_dict[i]))
            
    def set_dataset_params(self, dataset_cfg):
        if not self.exp_parser.has_section('dataset_configs'):
            self.exp_parser.add_section('dataset_configs')

        cfgparser = self.new_parser("pamap_parser")
        cfgparser.optionxform = str
        
        cfgparser.read(os.path.join(self.get_dir_path("preprocessing"), dataset_cfg))
        
        #base parameterss
        for i in cfgparser['base_parameters']:
            self.exp_parser.set(section='dataset_configs', option=i,
                            value= str(cfgparser['base_parameters'][i]))
       
       #use columns
        self.exp_parser.set(section='dataset_configs', option='used_sensors',
                        value= str([i for i in cfgparser['use_cols'] if cfgparser['use_cols'].getboolean(i)]))


    @property
    def exp_folder(self):
        assert 'exp_parser' in self.__dict__, "You must setup an experiment. Use read_experiment method."

        return self._exp_folder

    # def saveTag(self, name: str):
        # path = os.path.join(self.getDirectoryPath("TAGS_DIRECTORY"), f'{name}.cfg')
        # if os.path.isfile(path):
        #     overwrite = input(f"The tag {name} already exists - overwrite?\n[y/n] ")
        #     if overwrite == 'y' or overwrite == 'Y':
        #         with open(path, 'w') as cfgfile:
        #             self.__tag.write(cfgfile)
        #         print(f"The tag {name} has been updated.")
        #     else:
        #         print("The tag was not saved.")
        # else:
        #     with open(path, 'w') as cfgfile:
        #         self.__tag.write(cfgfile)
        #     print(f"The tag {name} has been saved.")
        # pass

    def new_parser(self, name: str, cfg_path=None):
        '''
        Creates a new configuration parser.

        Returns the created parser.
        '''

        if name in ["env_parser", "tag_parser", "exp_parser"]:
            raise Exception(f"The name '{name}' is a reserved name.")

        if name in self.__dict__:
            raise Exception(f"There is already a parser named '{name}'.")

        new_parser = configparser.ConfigParser()

        if cfg_path != None:
            abs_path = os.path.join(self.main_dir_path, cfg_path)

            assert os.path.isfile(
                abs_path), f"{os.path.realpath(abs_path)} is not a valid file!"

            new_parser.read(abs_path)

        self.__dict__[name] = new_parser

        return new_parser

    def print_attributes(self, parser_name=None):
        '''
            Prints the attributes of the provided parser.
            If parser_name is equal to None, print all configuration parsers.
        '''
        if parser_name != None:
            print(f"{parser_name}:\n")
            parser = self.__dict__[parser_name]
            for s in parser:
                print(f"[{s}]")
                for attr, value in parser[s].items():
                    print(f'{attr}: {value}')
                print()
        else:
            # Print all configuration parsers
            for name, v in self.__dict__.items():
                if type(v) == configparser.ConfigParser:
                    print(f"{name}:\n")
                    parser = v
                    for s in parser:
                        print(f"[{s}]")
                        for attr, value in parser[s].items():
                            print(f'{attr}: {value}')
                        print()

    def get_dir_path(self, dir_name: str):
        '''
        Returns the absolute path to the directory dir_name. 

        The requested directory should be specified in the [directories] section 
        of the default environment configuration file.
        '''

        return os.path.join(self.main_dir_path, self.env_parser["directories"][dir_name])

    def get_plugins_list(self) -> list:
        try:
            plugins = self.exp_parser.get('strategy', 'plugins').split(', ')
        except AttributeError:
            print("You must define a experiment config file.")
            plugins = []

        return plugins
    

    @property
    def main_dir_path(self):
        return self._main_dir_path
