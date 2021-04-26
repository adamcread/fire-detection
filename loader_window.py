import torch
import torch.utils.data as data_utl

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

        vid_path = os.path.join(self.root, vid)
                
        return vid_path, classification
    
    def __len__(self):
        return len(self.data.keys())
