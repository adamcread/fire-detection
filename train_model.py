import os
import argparse
import json
import time
from statistics import mode

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=1, help="batch size")
parser.add_argument('-length', type=int, default=16, help="num of frames considered in each train")
parser.add_argument('-lr', type=float, default=0.01, help="learning rate")
parser.add_argument('-clip', type=float, default=0.1, help="gradient clipping")
parser.add_argument('-train_mode', type=str, required=True, help="select 'start', 'random' or 'window'")
parser.add_argument('-fof', type=bool, default=False, help="use flow of flow")
parser.add_argument('-data_aug', type=bool, default=False, help="use data augmentation")
parser.add_argument('-resnet_depth', type=int, required=True, help="select 18, 34, 50, 101, 152 or 200")

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

print("daug", args.data_aug)
print("fof", args.fof)

#import model
import flow_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = flow_model.resnet_3d_v1(
    resnet_depth=args.resnet_depth, # taken from resnet_3d_v1 definition
    num_classes=2,
    fof=args.fof
)

model = nn.DataParallel(model).to(device)
batch_size = args.batch_size

from loader import DS

train = "./json/kim-lee-2019.json" # json containing videos for training
root = "./dataset/kim-lee-2019/split_resized/" # path to videos
state_path = "./state_dicts/"

# load training videos into object
dataset_tr = DS(
        split_file=train, # videos selected for loading
        root=root, # root dir to find videos
        length=args.length, # number of videos?
        mode=args.train_mode,
        data_aug=args.data_aug
) 
dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

dataloader = {'train':dl} # dictionary to contain training and validation videos loaded
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
    for phase in ['train']:
        print("epoch:", epoch, "phase:", phase)
        start = time.time()

        train = (phase=='train') # enable grad or not
        if train: # train model
            model.train()
        else: # evaluate model
            model.eval()
            
        tloss = 0. # time loss for each iteration?
        acc = 0. # accuracy
        tot = 0 #total?
        c = 0 # iteration

        # 0 - false negative 
        # 1 - true negative
        # 2 - false positive
        # 3 - true positive
        quant_results = [0, 0, 0, 0]

        with torch.set_grad_enabled(train):
            for vids, classification in dataloader[phase]:
                classification = classification.to(device)

                vid_preds = []
                for vid in vids:
                    vid = vid.to(device)
    
                    outputs = model(vid)
                    outputs = outputs.squeeze(3).squeeze(2)

                    if train:
                        loss = F.cross_entropy(outputs, classification)

                        solver.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        solver.step()
                    
                    pred = torch.max(outputs, dim=1)[1] 
                    vid_preds.append(pred)

                try:
                    video_pred = mode(vid_preds)
                except:
                    video_pred = vid_preds[-1]

                corr = torch.sum((video_pred == classification).int()) # number of correct videos
                acc += corr.item() # running tot of correctly classified
                tot += vid.size(0) # running tot of num of videos
                loss = F.cross_entropy(outputs, classification)

                bin_conversion = 2*video_pred.item() + corr.item()
                quant_results[bin_conversion] += 1

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

        path = '{}-{}-epoch{}'.format(args.train_mode, args.resnet_depth, epoch)
        if args.data_aug:
            path += "-daug"
        if args.fof:
            path += "-fof"

        torch.save(model.state_dict(), os.path.join(state_path, path+".pt"))