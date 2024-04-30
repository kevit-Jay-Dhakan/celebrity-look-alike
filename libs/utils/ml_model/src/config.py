from os import makedirs, path

from dotenv import dotenv_values

env_path = ".env"
if not path.exists(env_path):
    raise Exception(".env file not found")

config = dotenv_values(env_path)

INPUT_IMAGE_DOWNLOAD_PATH = config.get(
    "INPUT_IMAGE_DOWNLOAD_PATH", 'libs/utils/ml_model/user_images'
)
makedirs(INPUT_IMAGE_DOWNLOAD_PATH, exist_ok=True)
TRAIN_IMAGES_FOLDER_PATH = 'libs/utils/ml_model/train_images'
makedirs(TRAIN_IMAGES_FOLDER_PATH, exist_ok=True)
