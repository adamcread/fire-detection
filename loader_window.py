import torch
import torch.utils.data as data_utl

import lintel
import cv2
import numpy as np
import json
import os

print("hello")

class DS(data_utl.Dataset):
    def __init__(self, split_file, root, length=32):
        with open(split_file, 'r') as f:
            self.data = json.load(f)
        
        self.vids = [k for k in self.data.keys()]
        print(self.vids)

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

print("running")

dataset = DS(
        split_file = "./json/loader_test.json",
        root="./dataset/test_dataset/",
        length=16
)

print("data loaded")

for i in range(len(dataset)):
    print(i)
    x = dataset[i]


