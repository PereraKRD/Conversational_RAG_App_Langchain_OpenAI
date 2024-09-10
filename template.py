import os
import logging
from pathlib import Path

list_of_files = [
    "store_index.py",
    "static/.gitkeep",
    "templates/chat.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Directory {filedir} created successfully for file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"File {filepath} created successfully")
    else:
        logging.info(f"File {filepath} already exists")