import os
import json
import random

root = "./dataset/mivia/split_resized/"
dest = "./json/mivia.json"

json_dict = {}

for f in os.listdir(root):
    label = 0 if f[0] == 'n' else 1
    json_dict[f] = label

json_dump = json.dumps(json_dict, indent=4)

with open(dest, "w+") as f:
    f.write(json_dump)
