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
        self.__main_dir_name = "work"  # TODO change to 'work' for DL-28
        self.__main_dir_path = os.path.dirname(os.path.realpath(__file__))
        depth = 0
        while os.path.basename(self.__main_dir_path) != self.__main_dir_name:
            depth += 1
            if depth == 10:
                raise Exception("Main folder not found.")
            self.__main_dir_path = os.path.dirname(self.__main_dir_path)

        self.__save_exp = False  # Default values

        # Creates an environment parser and read the config file
        env_cfg_path = os.path.join(
            self.__main_dir_path, 'Configs/init/environment.cfg')

        assert os.path.isfile(
            env_cfg_path), f"{os.path.realpath(env_cfg_path)} is not a valid file!"

        self.env_parser = configparser.ConfigParser()
        self.env_parser.read(env_cfg_path)

        # init time
        #self.__init_time = self.currentTime()

        # Handling modifications received from the command line;
        #self.__argparser = argparse.ArgumentParser()
        # self.__argparser.add_argument('-t', '--tag') # TODO: Handle commented arguments
        # self.__argparser.add_argument('-d', '--dataset')
        # self.__argparser.add_argument('-s', '--strategy')
        # self.__argparser.add_argument('-e', '--epochs')
        # self.__argparser.add_argument('--save_tag')
        #self.__argparser.add_argument('--save_exp', action='store_true')

        #self.__args = self.__argparser.parse_args()

        # if self.__args.tag:
        #     self.readFromFile(os.path.join(
        #                         os.path.join(self.main_dir_path,
        #                         self.getstr("TAGS_DIRECTORY")), f"{self.__args.tag}.cfg"))
        # if self.__args.dataset:
        #     self.updateAttr('dataset', self.__args.dataset)
        # if self.__args.strategy:
        #     self.updateAttr('strategy', self.__args.strategy)
        # if self.__args.epochs:
        #     self.updateAttr('epochs', self.__args.epochs)
        #if self.__args.save_exp:
        #    self.__save_exp = True

    def __new__(cls):
        '''
        Implements the Singleton pattern.
        '''
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __del__(self):
        # set end_time
        self.__end_time = self.currentTime()

        # Save experiment
        if self.__save_exp:
            self.__save_experiment()

        # Save tag
        # if self.__args.save_tag:
            # self.saveTag(self.__args.save_tag)

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

        return

    def save_experiment(self):
        '''
        The method that specifies that the class should save experiment information upon completion.
        '''
        self.__save_exp = True

    def __save_experiment(self):
        results_path = os.path.join(self.get_dir_path("results"),
                                    self.exp_parser.get("DEFAULT", "dataset"),
                                    self.exp_parser.get("DEFAULT", "strategy"),
                                    self.__init_time,)

        # Create the directories to save the results
        os.makedirs(os.path.join(results_path, self.env_parser.get(
            "results", "graphs_directory")))
        os.makedirs(os.path.join(
            results_path, self.env_parser.get("results", "logs_directory")))
        os.makedirs(os.path.join(
            results_path, self.env_parser.get("results", "model_directory")))

        # # Save the experiment settings
        # cfg_path = os.path.join(results_path, self.getstr("exp_settings_file"))
        # with open(cfg_path, 'w') as cfgfile:
        #         self.__tag.write(cfgfile)

    def saveTag(self, name: str):
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
        pass

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

    @property
    def main_dir_path(self):
        return self.__main_dir_path
