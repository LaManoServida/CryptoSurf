from configparser import ConfigParser

config = ConfigParser()

# open config file and try to open the custom one if it exists
config.read('config.ini')
config.read('config-local.ini')

# extract constants
api_key = config['binance']['api_key']
api_secret = config['binance']['api_secret']
default_dataset_directory = config['training']['default_dataset_directory']
