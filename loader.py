import torch
import torch.utils.data as data_utl

import numpy as np
import random

import os
import lintel

import json


class DS(data_utl.Dataset):
    def __init__(self, split_file, root, length=32):
        with open(split_file, 'r') as f: 
            self.data = json.load(f) # get video paths from json

        self.vids = [k for k in self.data.keys()] # store video paths from json in list
        
        # save init variables
        self.split_file = split_file
        self.root = root
        self.length = length # length of videos

    def __getitem__(self, index):
        vid = self.vids[index] # get video at correct index
        classification = self.data[vid] # get label from correct vid
        
        # open path now video must exist
        with open(os.path.join(self.root, vid), 'rb') as f:
            enc_vid = f.read() # read file binary
        
        # loading vid into lintel
        # obtaining dataframe width and height of video
        # df, w, h, _ = lintel.loadvid(enc_vid, should_random_seek=self.random, num_frames=self.length*2)
        df, width, height, _ = lintel.loadvid(
                        enc_vid, 
                        num_frames=self.length
        )

        # interpret buffer as 1 dimensional array
        # convert from buffer to numpy array with int
        df = np.frombuffer(df, dtype=np.uint8) 
        df = np.reshape(df, newshape=(self.length, height, width, 3))

        # normalise data
        df = 1-2*(df.astype(np.float32)/255)
        
        # 3d -> return CxTxHxW
        df = df.transpose([3,0,1,2])
        return df, classification
        
    def __len__(self):
        return len(self.data.keys())