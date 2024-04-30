import subprocess
from os import getcwd
from os.path import join

from apps.platform_web.src.config import WEB_APP_PORT

if __name__ == '__main__':
    FILE_PATH = join(getcwd(), 'apps/platform_web/src/service.py')
    command = [
        "streamlit", "run", "--server.port", WEB_APP_PORT, FILE_PATH
    ]
    subprocess.run(command)
