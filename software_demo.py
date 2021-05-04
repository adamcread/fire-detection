import os
import argparse
import json
import time
from statistics import mode

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=1, help="batch size")
parser.add_argument('-length', type=int, default=16, help="num of frames considered in each train")
parser.add_argument('-train_mode', type=str, required=True, help="select 'start', 'random' or 'window'")
parser.add_argument('-model_path', type=str, required=True, help="path to trained model")
parser.add_argument('-resnet_depth', type=int, required=True, help="select 18, 34, 50, 101, 152 or 200\n")
parser.add_argument('-demo_classification', type=str, required=True)

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
    resnet_depth=args.resnet_depth, # taken from resnet_3d_v1 definition
    num_classes=2
)

model = nn.DataParallel(model).to(device)

state_path = "./state_dicts/" + args.model_path
model.load_state_dict(torch.load(state_path, map_location=torch.device('cpu')))

batch_size = args.batch_size

from loader import DS

val_root = "./json/demo-{}.json".format(args.demo_classification)
data_root = "./dataset/software_demo/"


# load training videos into object
dataset_val = DS(
        split_file=val_root,
        root=data_root,
        length=args.length, # number of videos?
        mode=args.train_mode
) 
vdl = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

dataloader = {'val':vdl} # dictionary to contain training and validation videos loaded

for phase in ['val']:
    model.eval()

    with torch.set_grad_enabled(False):
        for vids, classification in dataloader[phase]:
            classification = classification.to(device)

            vid_preds = []
            for vid in vids:
                vid = vid.to(device)

                outputs = model(vid)
                outputs = outputs.squeeze(3).squeeze(2)
                
                pred = torch.max(outputs, dim=1)[1] 
                vid_preds.append(pred)

            try:
                video_pred = mode(vid_preds)
            except:
                video_pred = vid_preds[-1]
        
        print("VIDEO PREDICTIONS:", vid_preds)
        print("OVERALL PREDICTION:", video_pred)
                

