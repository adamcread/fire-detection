import os
import json
import random

root = "./dataset/split_resized_dataset/"
dest = "./json/"

test_split = 0.80
ub = 100

train_files = {}
test_files = {}

all_files = [f for f in os.listdir(root)]
random.shuffle(all_files)


for file in all_files:
    label = 0 if file[0] == 'n' else 1
    classifier = random.randint(1, ub)

    if classifier <= round(ub * test_split):
        train_files[file] = label
    else:
        test_files[file] = label

train_json = json.dumps(train_files, indent=4)
test_json = json.dumps(test_files, indent=4)

with open(dest + "train.json", "w+") as f:
    f.write(train_json)

with open(dest + "val.json", "w+") as f:
    f.write(test_json)
