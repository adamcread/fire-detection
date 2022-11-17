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

val_root = "./json/" # json containing videos for evaluation
data_root = "./dataset/" # path to videos

files = ["mivia", "furg-fire-dataset", "chenebert-2011", "dunnings-2018"]

for f in files:
    # load training videos into object
    dataset_val = DS(
            split_file=val_root + f + ".json", # videos selected for loading
            root=data_root + f + "/split_resized/", # root dir to find videos
            length=args.length, # number of videos?
            mode=args.train_mode,
            data_aug=False
    ) 
    vdl = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    dataloader = {'val':vdl} # dictionary to contain training and validation videos loaded
    
    num_epochs = int(1) # iterations
    for epoch in range(num_epochs):
        for phase in ['val']:
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
            
            print("False negative:", quant_results[0])
            print("True negative:", quant_results[1])
            print("False positive:", quant_results[2])
            print("True positive:", quant_results[3])
            print("Total:", tot)
            print("Model:", args.model_path, "Val set:", f)
            print("Time:", time.time() - start)
