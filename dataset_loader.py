import torch
import torch.utils.data as data_utl

import numpy as np
import random

import os
import lintel

import json


class DS(data_utl.Dataset):
    def __init__(self, split_file, root, mode='rgb', length=64, random=True, model='2d', size=112):
        with open(split_file, 'r') as f: 
            self.data = json.load(f) # get video paths from json
        self.vids = [k for k in self.data.keys()] # store video paths from json in list

        if mode == 'flow': # if flow do what?
            new_data = {}

            self.vids = ['flow'+v[3:] for v in self.vids] # remap videos to be flow...{vid_path}

            for v in self.data.keys():
                new_data['flow'+v[3:]] = self.data[v] # add labels to flow

            self.data = new_data
        
        # save init variables
        self.split_file = split_file
        self.root = root
        self.mode = mode
        self.model = model
        self.length = length
        self.random = random
        self.size = size

    def __getitem__(self, index):
        vid = self.vids[index] # get video at correct index
        cls = self.data[vid] # get label from correct vid

        if not os.path.exists(os.path.join(self.root, vid)): # if vid cannot be found as it's flow?
            if self.mode == 'flow' and self.model == '2d':
                return np.zeros((3, 20, self.size, self.size), dtype=np.float32), 0 # return zeros if not found
            elif self.mode == 'flow' and self.model == '3d':
                return np.zeros((2, self.length, self.size, self.size), dtype=np.float32), 0 # return zeros if not found
        
        # open path now video must exist
        with open(os.path.join(self.root, vid), 'rb') as f:
            enc_vid = f.read() # read file binary? 
            # binary data = raw data => read file raw data
        
        # loading vid into lintel
        # obtaining dataframe width and height of video
        df, w, h, _ = lintel.loadvid(enc_vid, should_random_seek=self.random, num_frames=self.length*2)
        # print("width", w, "height", h) # raw video dimensions (unedited)
     
        # interpret buffer as 1 dimensional array
        df = np.frombuffer(df, dtype=np.uint8) # unsigned 8 bit integer

        if w < 128 or h < 128 or h > 512 or w > 512: # if video too big or too small
             # crop df to 128x128
            df = np.zeros(
                            (self.length*2, # number of frames?
                            128, # height
                            128, # width
                            3), # colour channels?
                            dtype=np.uint8
            )

            w = h = 128
            cls = 0

        # center crop 
        # applying random croppings to help improve performance slightly
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

            if random.random() < 0.5:
                df = np.flip(df, axis=2).copy()

        if self.mode == 'flow':
            #print(df[:,:,:,1:].mean())
            #exit()
            # only take the 2 channels corresponding to flow (x,y)
            # t+1, ..., t
            df = df[:,:,:,1:]
            if self.model == '2d':
                # this should be redone...
                # stack 10 along channel axis
                df = np.asarray([df[:10],df[2:12],df[4:14]]) # gives 3x10xHxWx2
                df = df.transpose(0,1,4,2,3).reshape(3,20,self.size,self.size).transpose(0,2,3,1)
            
        # normalise data
        df = 1-2*(df.astype(np.float32)/255)

        if self.model == '2d':
            # 2d -> return TxCxHxW
            return df.transpose([0,3,1,2]), cls
        
        # 3d -> return CxTxHxW
        return df.transpose([3,0,1,2]), cls
        
    def __len__(self):
        return len(self.data.keys())

