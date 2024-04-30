from os import getcwd, path
from os.path import join

from dotenv import dotenv_values

env_path = '.env'

if not path.exists(env_path):
    raise Exception('.env request_file not found')

config = dotenv_values(env_path)

WEB_APP_PORT = str(config.get('WEB_APP_PORT', 4550))
INPUT_IMAGE_DOWNLOAD_PATH = config.get(
    "INPUT_IMAGE_DOWNLOAD_PATH", 'libs/utils/ml_model/user_images'
)
INPUT_IMAGE_DOWNLOAD_PATH = join(getcwd(), INPUT_IMAGE_DOWNLOAD_PATH)
