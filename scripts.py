import subprocess

# List of scripts to run
scripts = [
    "./script/inthewild_script.py",
]

# Run each script in a separate process
processes = [subprocess.Popen(["python", script]) for script in scripts]

# Wait for all processes to complete
for process in processes:
    process.wait()