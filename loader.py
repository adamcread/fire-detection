import torch
import torch.utils.data as data_utl

import numpy as np
import random

import os
import lintel

import json


class DS(data_utl.Dataset):
    def __init__(self, split_file, root, length, mode='rgb', random=True, model='2d', size=24):
        with open(split_file, 'r') as f: 
            self.data = json.load(f) # get video paths from json

        self.vids = [k for k in self.data.keys()] # store video paths from json in list
        
        # save init variables
        self.split_file = split_file
        self.root = root
        self.mode = mode
        self.model = model
        self.length = length # length of videos
        self.random = random
        self.size = size

    def __getitem__(self, index):
        vid = self.vids[index] # get video at correct index
        classification = self.data[vid] # get label from correct vid
        
        # open path now video must exist
        with open(os.path.join(self.root, vid), 'rb') as f:
            enc_vid = f.read() # read file binary
            # binary data = raw data => read file raw data
        
        # loading vid into lintel
        # obtaining dataframe width and height of video
        # df, w, h, _ = lintel.loadvid(enc_vid, should_random_seek=self.random, num_frames=self.length*2)
        df, w, h, _ = lintel.loadvid(enc_vid, should_random_seek=self.random, num_frames=self.length*2) # ! GET RID OF RANDOM

        # interpret buffer as 1 dimensional array
        # convert from buffer to numpy array with int
        df = np.frombuffer(df, dtype=np.uint8) 

        if not self.random:
            i = int(round((h-self.size)/2.))
            j = int(round((w-self.size)/2.))
            df = np.reshape(df, newshape=(self.length*2, h, w, 3))[::2, i:-i, j:-j, :]
        else:
            th = self.size
            tw = self.size
            #print(h, th, h-th)
            i = random.randint(0, h - th) if h!=th else 0
            j = random.randint(0, w - tw) if w!=tw else 0
            df = np.reshape(df, newshape=(self.length*2, h, w, 3))[::2, i:i+th, j:j+tw, :]

            # randomly flip
            if random.random() < 0.5:
                df = np.flip(df, axis=2).copy()
            
        # normalise data
        df = 1-2*(df.astype(np.float32)/255)

        if self.model == '2d':
            # 2d -> return TxCxHxW
            return df.transpose([0,3,1,2]), classification
        
        # 3d -> return CxTxHxW
        df = df.transpose([3,0,1,2])
        return df, classification
        
    def __len__(self):
        return len(self.data.keys())