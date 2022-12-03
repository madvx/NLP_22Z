"""
KEEP THIS FILE IN THE SAME DIRECTORY AS 'main.py'!
"""
import ast
import os
from configparser import ConfigParser
from configupdater import ConfigUpdater

config_filepath = "config.ini"  # config filepath relative to main.py / file_manager.py
parser = ConfigParser(comment_prefixes='#')
updater = ConfigUpdater()


def get_filepath(file_name):
    """
    returns absolute filepath to file no matter where the main.py script was run from
    :param file_name: filepath relative to main.py / file_manager.py file
    :return: absolute filepath to file
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)


# Odczyt pliku konfiguracyjnego
def __get_config(section):
    """
    get dictionary of format {option_name: option_value} with data from given config file section
    :param section: section name of config file
    :return: dictionary of format {option_name: option_value} with data from given config file section
    """
    conf = {}
    try:
        parser.read(get_filepath(config_filepath))
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                conf[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, get_filepath(config_filepath)))
    except Exception as error:
        raise error
    return conf


def __set_config(section, config):
    """
    function to edit specific section in config file
    :param section: config file section which is to be edited
    :param config: dictionary with data in format {option_name: option_value}
    :return: returns True if config save was successful
    """
    try:
        updater.read(get_filepath(config_filepath))
        if updater.has_section(section):
            for key, value in config.items():
                updater.set(section, key, value)
            with open(get_filepath(config_filepath), "w") as config_file:
                updater.write(config_file)
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, get_filepath(config_filepath)))
    except Exception as error:
        raise error
    return True


def get_config(section: str):
    return parse_config(__get_config(section=section))

def parse_config(config: dict[str]):
    """
    parses str-only config to target python data-structures using `ast.literal_eval`
    """
    parsed_dict = {}
    for key, value in config.items():
        parsed_dict[key] = ast.literal_eval(value)
    return parsed_dict


def format_config(config: dict):
    formatted_dict = {}
    for key, value in config.items():
        if isinstance(value, str):
            formatted_dict[key] = f'"{value}"'
        else:
            formatted_dict[key] = str(value)
    return formatted_dict


if __name__ == "__main__":
    conf = __get_config(section="scrapping")
    print(conf, "\n\n")

    conf = parse_config(conf)
    print(conf, "\n\n")

    print(format_config(conf), "\n\n")
    # __set_config(section="scrapping", config=format_config(conf))


