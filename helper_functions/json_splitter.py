import os
import json

root = "./dataset/split_resized_dataset/"
dest = "./json/"

files = os.listdir(root)

for i in range(len(files)):
    f_json = {}

    f = files[i]
    label = 0 if f[0] == 'n' else 1

    f_json[f] = label

    dumped_json = json.dumps(f_json, indent=4)
    
    with open(dest + str(i) + ".json", "w+") as f:
        f.write(dumped_json)

