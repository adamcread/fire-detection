import os
import sys
import argparse
import inspect
import datetime
import json

import time

parser = argparse.ArgumentParser()
parser.add_argument('-system', type=str, default='fire')
parser.add_argument('-mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('-model', type=str, default='3d')
parser.add_argument('-exp_name', type=str)
parser.add_argument('-batch_size', type=int, default=24)
parser.add_argument('-length', type=int, default=16)
parser.add_argument('-learnable', type=str, default='[0,0,0,0,0]')
parser.add_argument('-niter', type=int)
parser.add_argument('-clip', type=float, default=0.01, help='gradient clipping')

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

#import models
import flow_2p1d_resnets
import kinetics_2p1d_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##################
#
# Create model, dataset, and training setup
#
##################
# model = flow_2p1d_resnets.resnet50(pretrained=False, mode=args.mode, n_iter=args.niter, learnable=eval(args.learnable), num_classes=400)
model = kinetics_2p1d_model.resnet_3d_v1(
    resnet_depth=50, # taken from resnet_3d_v1 definition
    num_classes=2
)

model = nn.DataParallel(model).to(device)
batch_size = args.batch_size

if args.system == 'fire':
    train = 'data/fire_train_temp.json' # json containing videos for training
    val = 'data/fire_val.json' # json containing videos for evaluation
    root = './fire' # path to videos

    from minikinetics_dataset import MK
    # load training videos into object
    dataset_tr = MK(
                split_file=train, # videos selected for loading
                root=root, # root dir to find videos
                length=args.length, # number of videos?
                model=args.model, # 2d/3d
                mode=args.mode # rgb or flow?
    ) 

    dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dataset = MK(val, root, length=args.length, model=args.model, mode=args.mode) # load evaluation videos into object

    vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


    dataloader = {'train':dl, 'val':vdl} # dictionary to contain training and validation videos loaded

    
# scale lr for flow layer
params = model.parameters()
params = [p for p in params]
other = []
ln = eval(args.learnable)
if ln[0] == 1:
    other += [p for p in params if (p.sum() == model.module.flow_layer.img_grad.sum()).all() and p.size() == model.module.flow_layer.img_grad.size()]
    other += [p for p in params if (p.sum() == model.module.flow_layer.img_grad2.sum()).all() and p.size() == model.module.flow_layer.img_grad2.size()]
    params = [p for p in params if (p.sum() != model.module.flow_layer.img_grad.sum()).all() or p.size() != model.module.flow_layer.img_grad.size()]
    params = [p for p in params if (p.sum() != model.module.flow_layer.img_grad2.sum()).all() or p.size() != model.module.flow_layer.img_grad2.size()]

if ln[1] == 1:
    other += [p for p in params if (p.sum() == model.module.flow_layer.f_grad.sum()).all() and p.size() == model.module.flow_layer.f_grad.size()]
    other += [p for p in params if (p.sum() == model.module.flow_layer.f_grad2.sum()).all() and p.size() == model.module.flow_layer.f_grad2.size()]
    params = [p for p in params if (p.sum() != model.module.flow_layer.f_grad.sum()).all() or p.size() != model.module.flow_layer.f_grad.size()]
    params = [p for p in params if (p.sum() != model.module.flow_layer.f_grad2.sum()).all() or p.size() != model.module.flow_layer.f_grad2.size()]

if ln[2] == 1:
    other += [p for p in params if (p.sum() == model.module.flow_layer.t.sum()).all() and p.size() == model.module.flow_layer.t.size()]
    params = [p for p in params if (p.sum() != model.module.flow_layer.t.sum()).all() or p.size() != model.module.flow_layer.t.size()]

if ln[3] == 1:
    other += [p for p in params if (p.sum() == model.module.flow_layer.l.sum()).all() and p.size() == model.module.flow_layer.l.size()]
    params = [p for p in params if (p.sum() != model.module.flow_layer.l.sum()).all() or p.size() != model.module.flow_layer.l.size()]

if ln[4] == 1:
    other += [p for p in params if (p.sum() == model.module.flow_layer.a.sum()).all() and p.size() == model.module.flow_layer.a.size()]
    params = [p for p in params if (p.sum() != model.module.flow_layer.a.sum()).all() or p.size() != model.module.flow_layer.a.size()]
    
#print([p for p in model.parameters() if (p == model.module.flow_layer.t).all()])
#print(other)
# print(len(params), len(other))
#exit()

lr = 0.01
# stochastic gradient descent
solver = optim.SGD(
    [{'params':params}, {'params':other, 'lr':0.01*lr}], # model.classifier's parameters
    lr=lr, weight_decay=1e-6, momentum=0.9) # model.base's parameters

# changes learning rate based on current descent
# ? gain more intuition about this
lr_sched = optim.lr_scheduler.ReduceLROnPlateau(solver, patience=7)


#################
#
# Setup logs, store model code
# hyper-parameters, etc...
#
#################
# log_name = datetime.datetime.today().strftime('%m-%d-%H%M%S') # +'-'+args.exp_name
# log_path = os.path.join('logs/',log_name)
# os.mkdir(log_path)
# os.system('cp * logs/'+log_name+'/')

# # # deal with hyper-params...
# with open(os.path.join(log_path,'params.json'), 'w') as out:
#     hyper = vars(args)
#     json.dump(hyper, out)
# log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[]}

    

###############
#
# Train the model and save everything
#
###############
num_epochs = 50 # iterations
for epoch in range(num_epochs):
    for phase in ['train']:
        train = (phase=='train') # enable grad or not
        if phase == 'train': # train model
            model.train()
        else: # evaluate model
            model.eval()
            
        tloss = 0. # time loss for each iteration?
        acc = 0. # accuracy/
        tot = 0 #total?
        c = 0 # iteration

        with torch.set_grad_enabled(train):
            # ! try one by one
            for vid, cls in dataloader[phase]:
                print("mode:", phase)
                print("epoch {} video {}".format(epoch, c*batch_size))
                vid = vid.to(device) # assign video to device?

                cls = cls.to(device)
                
                outputs = model(vid)
                outputs = outputs.squeeze(3).squeeze(2)
                # ! normalize? 

                # exploding gradients
                pred = torch.max(outputs, dim=1)[1] # ! potentially change to softmax

                # num of correct 
                corr = torch.sum((pred == cls).int())

                acc += corr.item()
                tot += vid.size(0)

                loss = F.cross_entropy(outputs, cls)
                
                if phase == 'train':
                    solver.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    solver.step()

                tloss += loss.item()
                c += 1
            
        if phase == 'train':
            # log['epoch'].append(tloss/c)
            # log['train_acc'].append(acc/tot)
            print('train loss', tloss/c, 'acc', acc/tot)
        else:
            # log['validation'].append(tloss/c)
            # log['val_acc'].append(acc/tot)
            print('val loss', tloss, 'acc', acc/tot)
            lr_sched.step(tloss/c)
    
    # with open(os.path.join(log_path,'log.json'), 'w') as out:
    #     json.dump(log, out)
    # torch.save(model.state_dict(), os.path.join(log_path, 'hmdb_flow-of-flow_2p1d.pt'))

    # lr_sched.step(loss/c)
