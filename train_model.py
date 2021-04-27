import os
import argparse
import json
import time

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-length', type=int, default=16)
parser.add_argument('-lr', type=float, default=0.01, help="learning rate")
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
    resnet_depth=18, # taken from resnet_3d_v1 definition
    num_classes=2
)

model = nn.DataParallel(model).to(device)
batch_size = args.batch_size

from loader import DS

train = "./json/train.json" # json containing videos for training
val = "./json/val.json" # json containing videos for evaluation
root = "./dataset/split_resized_dataset/" # path to videos

# load training videos into object
dataset_tr = DS(
        split_file=train, # videos selected for loading
        root=root, # root dir to find videos
        length=args.length, # number of videos?
) 
dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

# load evaluation videos into object
dataset_val = DS(
        split_file=val, 
        root=root, 
        length=args.length, 
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
        print("epoch:", epoch, "phase:", phase)
        start = time.time()

        train = (phase=='train') # enable grad or not
        if train: # train model
            model.train()
        else: # evaluate model
            model.eval()
            
        tloss = 0. # total loss
        acc = 0. # accuracy
        tot = 0 #total videos
        c = 0 # batch counter

        # 0 - false negative 
        # 1 - true negative
        # 2 - false positive
        # 3 - true positive
        quant_results = [0, 0, 0, 0]

        with torch.set_grad_enabled(train):
            for vid, classification in dataloader[phase]:
                vid = vid.to(device)
   
                classification = classification.to(device)
                outputs = model(vid)
                outputs = outputs.squeeze(3).squeeze(2)

                pred = torch.max(outputs, dim=1)[1] 
                corr = torch.sum((pred == classification).int())
                acc += corr.item()
                tot += vid.size(0)

                bin_conversion = 2*pred.item() + corr.item()
                quant_results[bin_conversion] += 1

                loss = F.cross_entropy(outputs, classification)
                
                if phase == 'train':
                    solver.zero_grad()
                    loss.backward()

                    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    solver.step()
                
                tloss += loss.item()
                c += 1
            
        if phase == 'train':
            print('train loss', tloss/c, 'acc', acc/tot)
        else:
            print('val loss', tloss/c, 'acc', acc/tot)
            lr_sched.step(tloss/c)

        print("False negative:", quant_results[0])
        print("True negative:", quant_results[1])
        print("False positive:", quant_results[2])
        print("True positive:", quant_results[3])
        print("Total:", tot)
        print("Time:", time.time() - start)
