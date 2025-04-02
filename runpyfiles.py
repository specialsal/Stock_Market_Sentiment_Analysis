import subprocess
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib as mpl


# List of scripts to run
scripts = ["model_ml.py", "compute_sent_idx.py", "plot_sent_idx.py"]

for script in scripts:
    try:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)
        print(f"{script} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
        break


# https://guba.eastmoney.com/list,zssh000001.html
def get_guba_data():
    url = "https://guba.eastmoney.com/list,zssh000001.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    response = requests.get(url, headers=headers)