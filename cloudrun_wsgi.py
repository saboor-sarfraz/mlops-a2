import logging

from cloudrun_api import create_app

logging.basicConfig(level=logging.INFO)
app = create_app()
