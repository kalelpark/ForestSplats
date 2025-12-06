import os
import subprocess

DATA_DIR = "/home/data/cv_datasets/Wongi_Data/wildset/phototourism"
RES_DIR = "./result/phototourism"
scenes = ["brandenburg-gate"] 

for scene in scenes:
    command = [
        "taskset", "-c", "11,12,13,14,15",
        "/home/XXXX/anaconda3/envs/ForestSplats/bin/python", "-m", "ForestSplats.train",
        "--data", os.path.join(DATA_DIR, scene),
        "--output", f"./final_25/phototourism/{scene}/",
        "--iteration", "100000",
        "--dataset-type", "phototourism",
        "--savepath", f"./final/{scene}/",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    subprocess.run(command, env=env)
    