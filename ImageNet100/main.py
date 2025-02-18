import json
import os

with open("Labels.json") as f:
    dir_names = json.loads(f.read())

def rename_dir(old_name, new_name):
    os.rename(old_name, new_name)

for old_name in dir_names:
    rename_dir(f"train.X/{old_name}", f"train.X/{dir_names[old_name]}")
    rename_dir(f"val.X/{old_name}", f"val.X/{dir_names[old_name]}")


# rename_dir("t.tt/tt", "t.tt/pp")
