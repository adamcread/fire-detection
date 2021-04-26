import os
import argparse
import json

from statistics import mode
import cv2
import lintel
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-length', type=int, default=16)
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-clip', type=float, default=0.1, help='gradient clipping')

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

#import model
import flow_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = flow_model.resnet_3d_v1(
    resnet_depth=200, # taken from resnet_3d_v1 definition
    num_classes=2
)

model = nn.DataParallel(model).to(device)
batch_size = args.batch_size

from loader_window import DS

train = "./json/train.json" # json containing videos for training
val = "./json/val.json" # json containing videos for evaluation
root = "./dataset/split_resized_dataset/" # path to videos

# load training videos into object
dataset_tr = DS(
        split_file=train, # videos selected for loading
        root=root, # root dir to find videos
        length=args.length # number of videos?
) 
dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

# load evaluation videos into object
dataset_val = DS(
        split_file=val, 
        root=root, 
        length=args.length
) 
vdl = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

dataloader = {'train':dl, 'val':vdl} # dictionary to contain training and validation videos loaded
print("DATA LOADED")
    
params = [p for p in model.parameters()]
# stochastic gradient descent
solver = optim.SGD(
    [{'params':params}], # model.classifier's parameters
    lr=args.lr, 
    weight_decay=1e-6, 
    momentum=0.9
) # model.base's parameters

# changes learning rate based on current descent
lr_sched = optim.lr_scheduler.ReduceLROnPlateau(solver, patience=7)
num_epochs = int(1e30) # iterations
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        train = (phase=='train') # enable grad or not
        if train: # train model
            model.train()
        else: # evaluate model
            model.eval()
            
        tloss = 0. # time loss for each iteration?
        acc = 0. # accuracy
        tot = 0 #total?
        c = 0 # iteration

        with torch.set_grad_enabled(train):
            for vid_path, classification in dataloader[phase]:
                print("mode:", phase)
                print("epoch {} video {}".format(epoch, c*batch_size))
                classification = classification.to(device) # video prediction

                with open(vid_path[0], 'rb') as f:
                    enc_vid = f.read()

                cap = cv2.VideoCapture(vid_path[0])
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                vid_preds = []
                for mult in range(0, total_frames//args.length):
                    start_frame = mult*args.length

                    frame_nums = [x+start_frame for x in range(args.length)] 
                    df, width, height = lintel.loadvid_frame_nums(
                                    enc_vid,
                                    frame_nums = frame_nums,
                                    should_seek = True
                    )

                    df = np.frombuffer(df, dtype=np.uint8)
                    df = np.reshape(df, newshape=(args.length, height, width, 3))

                    df = 1-2*(df.astype(np.float32)/255)
                    vid = df.transpose([3,0,1,2])

                    vid = vid.to(device)
    
                    outputs = model(vid)
                    outputs = outputs.squeeze(3).squeeze(2)

                    pred = torch.max(outputs, dim=1)[1] 
                    
                    if train:
                        corr = torch.sum((pred == classification).int()) # number of correct videos
                        acc += corr.item() # running tot of correctly classified
                        tot += vid.size(0) # running tot of num of videos
                        loss = F.cross_entropy(outputs, classification)

                        solver.zero_grad()
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        solver.step()

                        tloss += loss.item()
                        c += 1
                    else:
                        vid_preds.append(pred)
                
                if not train:
                    try:
                        video_pred = mode(vid_preds)
                    except:
                        video_pred = vid_preds[-1]

                    corr = torch.sum((video_pred == classification).int()) # number of correct videos
                    acc += corr.item() # running tot of correctly classified
                    tot += vid.size(0) # running tot of num of videos
                    loss = F.cross_entropy(outputs, classification)

                    tloss += loss.item()
                    c += 1
                
                print("epoch {} video {}".format(epoch, c*batch_size))

        if phase == 'train':
            print('train loss', tloss/c, 'acc', acc/tot)
        else:
            print('val loss', tloss/c, 'acc', acc/tot)
            lr_sched.step(tloss/c)
    
    lr_sched.step(loss/c)
