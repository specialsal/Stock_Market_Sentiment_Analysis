import subprocess

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