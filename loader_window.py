import torch
import torch.utils.data as data_utl

import lintel
import cv2
import numpy as np
import json
import os

class DS(data_utl.Dataset):
    def __init__(self, split_file, root, length=16):
        with open(split_file, 'r') as f:
            self.data = json.load(f)
        
        self.vids = [k for k in self.data.keys()]

        self.split_file = split_file
        self.root = root
        self.length = length

    def __getitem__(self, index):
        vid = self.vids[index]
        classification = self.data[vid]

        print(vid)

        vid_path = os.path.join(self.root, vid)

        with open(vid_path, 'rb') as f:
            enc_vid = f.read()

        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        full_window = []
        for mult in range(0, total_frames//self.length):
            start_frame = mult*self.length

            frame_nums = [x+start_frame for x in range(self.length)] 
            df, width, height = lintel.loadvid_frame_nums(
                            enc_vid,
                            frame_nums = frame_nums,
                            should_seek = True
            )

            df = np.frombuffer(df, dtype=np.uint8)
            df = np.reshape(df, newshape=(self.length, height, width, 3))

            df = 1-2*(df.astype(np.float32)/255)
            df = df.transpose([3,0,1,2])

            full_window.append(df)
        
        return full_window, classification
    
    def __len__(self):
        return len(self.data.keys())

train = "./json/train.json" # json containing videos for training
val = "./json/val.json" # json containing videos for evaluation
root = "./dataset/split_resized_dataset/" # path to videos

# load training videos into object
dataset_tr = DS(
        split_file=train, # videos selected for loading
        root=root, # root dir to find videos
        length=16 # number of videos?
) 

# load training videos into object
dataset_val = DS(
        split_file=val, # videos selected for loading
        root=root, # root dir to find videos
        length=16 # number of videos?
) 

print("training")
for i in range(len(dataset_tr)):
    x = dataset_tr[i] 
    print("tr", i)

print("val")
for j in range(len(dataset_val)):
    x = dataset_val[i]
    print("val", j)
