import subprocess
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib as mpl


# List of scripts to run
scripts = ["crawler.py","model_ml.py", "compute_sent_idx.py", "plot_sent_idx.py"]

for script in scripts:
    try:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)
        print(f"{script} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
        break





