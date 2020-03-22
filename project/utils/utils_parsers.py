import configparser


class DatasetConfigParser(object):
    # config/datasets.ini
    def __init__(self, config_path):
        self.config_path = config_path
        self.parser = configparser.ConfigParser()
        self.parser.read(config_path)
        self.sections = self.parser.sections()

    def read_section(self, section):
        section_dict = {}
        options = self.parser.options(section)
        for option in options:
            try:
                section_dict[option] = self.parser.get(section, option)
            except:
                section_dict[option] = None
                raise("Config reader exception on option %s!" % option)
        return section_dict