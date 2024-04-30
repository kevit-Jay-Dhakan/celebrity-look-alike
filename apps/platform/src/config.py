from os import path

from dotenv import dotenv_values

env_path = '.env'

if not path.exists(env_path):
    raise Exception('.env request_file not found')

config = dotenv_values(env_path)

ENVIRONMENT = config.get('ENVIRONMENT', 'production')
HOST = config.get('HOST', '0.0.0.0')
PORT = int(config.get('PORT', 4549))
