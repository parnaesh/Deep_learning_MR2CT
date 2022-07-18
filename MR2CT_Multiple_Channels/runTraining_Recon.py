# from __future__ import print_function
import argparse, os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch
import torch.utils.data as data_utils
from net3d import NetS
import time
import random
import loadImages
from loadImages import load_images
from loadImages import extract_batch_single_subject
import SimpleITK as sitk

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MR2CT")

parser.add_argument("--gpuID", type=int, default=3, help="which gpu to use")
parser.add_argument("--whichLoss", type=int, default=1, help="which loss to use: 1. LossL1, 2. lossRTL1, 3. MSE (default)")
parser.add_argument("--batchSize", type=int, default=10, help="training batch size")
parser.add_argument("--numofIters", type=int, default=55000000, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=1000, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=1000, help="number of iterations to save the model")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--decLREvery", type=int, default=49980, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=49980")
parser.add_argument("--prefixModelName", default="/tmp/t_", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--training_path", default="/tmp", type=str, help="training path")

parser.add_argument("--pretrained_model", default=1, type=int, help="starting from a pre-trained model")

##########control which information to use
parser.add_argument("--using_r1", default=0, type=int, help="using r1 for training 1(yes) 0(no)")
parser.add_argument("--using_r2", default=0, type=int, help="using r2 for training 1(yes) 0(no)")
parser.add_argument("--using_t1", default=0, type=int, help="using t1 for training 1(yes) 0(no)")

parser.add_argument("--using_fa3e1",  default=0, type=int, help="using fa3e1  for training 1(yes) 0(no)")
parser.add_argument("--using_fa3e2",  default=0, type=int, help="using fa3e2  for training 1(yes) 0(no)")
parser.add_argument("--using_fa15e1", default=0, type=int, help="using fa15e1 for training 1(yes) 0(no)")
parser.add_argument("--using_fa15e2", default=0, type=int, help="using fa15e2 for training 1(yes) 0(no)")

parser.add_argument("--using_dixon1",  default=0, type=int, help="using dixon1  for training 1(yes) 0(no)")
parser.add_argument("--using_dixon2",  default=0, type=int, help="using dixom2  for training 1(yes) 0(no)")

parser.add_argument("--ndf", default=32, type=int, help="number of features, default 32")

parser.add_argument("--tissue_type_balanced_sampling", default=1, type=int, help="tissue type balanced sampling, default 1")
parser.add_argument("--using_fixed_learning_rate", default=1, type=int, help="using fixed learning rate")

global opt, model
opt = parser.parse_args()

d1=64
d2=64
d3=64
dPatch  =[d1,d2,d3] # size of patches of input data
dTarget =[64,64,64] # size of pathes of label data
print(d1)

ss1 = d1//4
es1 = d1//4*3

ss2 = d2//4
es2 = d2//4*3

ss3 = d3//4
es3 = d3//4*3

#############################################################################################################################
def main():
    opt=parser.parse_args()
    print(opt)

    num_channels = opt.using_r1+opt.using_r2+opt.using_t1+opt.using_fa3e1+opt.using_fa3e2+opt.using_fa15e1+opt.using_fa15e2+opt.using_dixon1+opt.using_dixon2
    print('total number of channels %d'%(num_channels))

    net = NetS(1, num_channels, opt.ndf)
    net.cuda()

    starting_iter=0

    if opt.pretrained_model>0:
        model_name = opt.prefixModelName+"%d.pt"%(opt.pretrained_model)

        starting_iter = opt.pretrained_model*opt.saveModelEvery+1

        if opt.using_fixed_learning_rate == 0:
            lr_range = np.floor(starting_iter/opt.decLREvery)
            opt.lr = opt.lr*pow(0.5, lr_range+1)

            if opt.lr < 0.00001:
                opt.lr=0.00001
   
        print('current iter %d, lr %f'%(starting_iter, opt.lr))

        if os.path.exists(model_name):
            #checkpoint = torch.load(model_name)
            #net.load_state_dict(checkpoint['model'])
            net.load_state_dict(torch.load(model_name))
        else:
            print('%s does not exist!\n'%(model_name)) 
            exit()

    total_params = sum(p.numel() for p in net.parameters())
    total_trainables = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("total_params %d, total_trainables %d"%(total_params, total_trainables))

    optimizer = optim.Adam(net.parameters(),lr=opt.lr)
    criterion_L2 = nn.MSELoss()
    criterion_L1 = nn.L1Loss(reduction='elementwise_mean')

    criterion_L2 = criterion_L2.cuda()
    criterion_L1 = criterion_L1.cuda()

    ##########################################################################################
    path_train=opt.training_path

    ct_train, r1_train, r2_train, t1_train, fa3e1_train, fa3e2_train, fa15e1_train, fa15e2_train, dixon1_train, dixon2_train, mk_train, l1_train_air, l2_train_air, l3_train_air, l1_train_brain, l2_train_brain, l3_train_brain, l1_train_bone, l2_train_bone, l3_train_bone = load_images(path_train, dPatch, dTarget, opt)
    num_subjects = len(ct_train)

    print("num_subjects, r1, r2, t1, fa3e1, fa3e2, fa15e1, fa15e2, dixon1, dixon2, mk, air brain bone ", len(ct_train), len(r1_train), len(r2_train), len(t1_train), len(fa3e1_train), len(fa3e2_train), len(fa15e1_train), len(fa15e2_train), len(dixon1_train), len(dixon2_train), len(mk_train), len(l1_train_air), len(l1_train_brain), len(l1_train_bone))
  
    ##########################################################################################
    index_subject = random.randint(0, num_subjects-1)
    
    running_loss = 0.0
    start = time.time()

    for iter in range(starting_iter, opt.numofIters):

        if iter%4 == 0:
            index_subject = random.randint(0, num_subjects-1) 

        if iter%4 ==0:
            ct_train_batch, r1_train_batch, r2_train_batch, t1_train_batch, fa3e1_train_batch, fa3e2_train_batch, fa15e1_train_batch, fa15e2_train_batch, dixon1_train_batch, dixon2_train_batch = extract_batch_single_subject(ct_train, r1_train, r2_train, t1_train, fa3e1_train, fa3e2_train, fa15e1_train, fa15e2_train, dixon1_train, dixon2_train, l1_train_air, l2_train_air, l3_train_air, index_subject, opt.batchSize, dPatch, dTarget)

        if iter%4 ==1:
            ct_train_batch, r1_train_batch, r2_train_batch, t1_train_batch, fa3e1_train_batch, fa3e2_train_batch, fa15e1_train_batch, fa15e2_train_batch, dixon1_train_batch, dixon2_train_batch = extract_batch_single_subject(ct_train, r1_train, r2_train, t1_train, fa3e1_train, fa3e2_train, fa15e1_train, fa15e2_train, dixon1_train, dixon2_train, l1_train_brain, l2_train_brain, l3_train_brain, index_subject, opt.batchSize, dPatch, dTarget)

        if iter%4 ==2:
            ct_train_batch, r1_train_batch, r2_train_batch, t1_train_batch, fa3e1_train_batch, fa3e2_train_batch, fa15e1_train_batch, fa15e2_train_batch, dixon1_train_batch, dixon2_train_batch = extract_batch_single_subject(ct_train, r1_train, r2_train, t1_train, fa3e1_train, fa3e2_train, fa15e1_train, fa15e2_train, dixon1_train, dixon2_train, l1_train_bone, l2_train_bone, l3_train_bone, index_subject, opt.batchSize, dPatch, dTarget)
        if iter%4 ==3:
            ct_train_batch, r1_train_batch, r2_train_batch, t1_train_batch, fa3e1_train_batch, fa3e2_train_batch, fa15e1_train_batch, fa15e2_train_batch, dixon1_train_batch, dixon2_train_batch = extract_batch_single_subject(ct_train, r1_train, r2_train, t1_train, fa3e1_train, fa3e2_train, fa15e1_train, fa15e2_train, dixon1_train, dixon2_train, l1_train_bone, l2_train_bone, l3_train_bone, index_subject, opt.batchSize, dPatch, dTarget)

        ##########################################################
        if opt.using_r1 == 1:
            r1_train_batch = r1_train_batch.astype(float)
            r1_train_batch = torch.from_numpy(r1_train_batch)
            r1_train_batch = r1_train_batch.float()

        if opt.using_r2 == 1:
            r2_train_batch = r2_train_batch.astype(float)
            r2_train_batch = torch.from_numpy(r2_train_batch)
            r2_train_batch = r2_train_batch.float()

        if opt.using_t1 == 1:
            t1_train_batch = t1_train_batch.astype(float)
            t1_train_batch = torch.from_numpy(t1_train_batch)
            t1_train_batch = t1_train_batch.float()

        if opt.using_fa3e1 == 1:
            fa3e1_train_batch = fa3e1_train_batch.astype(float)
            fa3e1_train_batch = torch.from_numpy(fa3e1_train_batch)
            fa3e1_train_batch = fa3e1_train_batch.float()

        if opt.using_fa3e2 == 1:
            fa3e2_train_batch = fa3e2_train_batch.astype(float)
            fa3e2_train_batch = torch.from_numpy(fa3e2_train_batch)
            fa3e2_train_batch = fa3e2_train_batch.float()

        if opt.using_fa15e1 == 1:
            fa15e1_train_batch = fa15e1_train_batch.astype(float)
            fa15e1_train_batch = torch.from_numpy(fa15e1_train_batch)
            fa15e1_train_batch = fa15e1_train_batch.float()

        if opt.using_fa15e2 == 1:
            fa15e2_train_batch  = fa15e2_train_batch.astype(float)
            fa15e2_train_batch = torch.from_numpy(fa15e2_train_batch)
            fa15e2_train_batch = fa15e2_train_batch.float()

        if opt.using_dixon1 == 1:
            dixon1_train_batch = dixon1_train_batch.astype(float)
            dixon1_train_batch = torch.from_numpy(dixon1_train_batch)
            dixon1_train_batch = dixon1_train_batch.float()

        if opt.using_dixon2 == 1:
            dixon2_train_batch = dixon2_train_batch.astype(float)
            dixon2_train_batch = torch.from_numpy(dixon2_train_batch)
            dixon2_train_batch = dixon2_train_batch.float()

        ct_train_batch = ct_train_batch.astype(float)
        ct_train_batch = torch.from_numpy(ct_train_batch)
        ct_train_batch = ct_train_batch.float()

        ##########################################################
        if num_channels ==1 :
            if opt.using_t1==1:
                source = t1_train_batch

            if opt.using_r1==1:
                source = r1_train_batch

            if opt.using_r2==1:
                source = r2_train_batch

        ##########################################################
        if num_channels ==2 :
            if opt.using_dixon1==1 and opt.using_dixon2==1:
                source = torch.cat((dixon1_train_batch, dixon2_train_batch), dim=1)

            if opt.using_fa3e1==1 and opt.using_fa15e1==1:
                source = torch.cat((fa3e1_train_batch, fa15e1_train_batch), dim=1)

        ##########################################################
        if num_channels ==3 :
            if opt.using_r2==1 and opt.using_fa15e1==1 and opt.using_fa15e2==1:
                source = torch.cat((r2_train_batch, fa15e1_train_batch), dim=1)
                source = torch.cat((source, fa15e2_train_batch), dim=1)

            if opt.using_r1==1 and opt.using_fa3e1==1 and opt.using_fa15e1==1:
                source = torch.cat((r1_train_batch, fa3e1_train_batch), dim=1)
                source = torch.cat((source, fa15e1_train_batch), dim=1)

        ##########################################################
        if num_channels ==7 :
            if opt.using_r1==1 and opt.using_r2==1 and opt.using_t1==1 and opt.using_fa3e1==1 and opt.using_fa3e2==1 and opt.using_fa15e1==1 and opt.using_fa15e2==1:
                source = torch.cat((r1_train_batch, r2_train_batch), dim=1)
                source = torch.cat((source, t1_train_batch), dim=1)
                source = torch.cat((source, fa3e1_train_batch), dim=1)
                source = torch.cat((source, fa3e2_train_batch), dim=1)
                source = torch.cat((source, fa15e1_train_batch), dim=1)
                source = torch.cat((source, fa15e2_train_batch), dim=1)

        ##########################################################
        source = source.cuda()
        ct_train_batch = ct_train_batch.cuda()

        source, ct_train_batch = Variable(source), Variable(ct_train_batch)
        outputG = net(source)
        net.zero_grad()

        if opt.whichLoss==1:
            lossG_G = criterion_L1(torch.squeeze(outputG), torch.squeeze(ct_train_batch))
        else:
            lossG_G = criterion_L2(torch.squeeze(outputG), torch.squeeze(ct_train_batch))

        lossG_G.backward() #compute gradients

        optimizer.step() #update network parameters

        running_loss = running_loss + lossG_G.data.item()

        ##########################################################
        if iter%(opt.showTrainLossEvery)==0: #print every 2000 mini-batches
            print('************************************************')
            print('time now is: ' + time.asctime(time.localtime(time.time())))
            print('average running loss between iter [%d, %d] is: %.5f'%(iter-opt.showTrainLossEvery+1,iter,running_loss/(opt.showTrainLossEvery)))
            print('lossG_G is %.5f.'%(lossG_G.item()))
            print('cost time for iter [%d, %d] is %.2f'%(iter - opt.showTrainLossEvery + 1,iter, time.time()-start))
            print('************************************************')
            running_loss = 0.0
            start = time.time()

        ##########################################################
        if iter%(opt.saveModelEvery)==0: #save the model
            #state = {
            #    'epoch': iter+1,
            #    'model': net.state_dict()
            #}
            #torch.save(state, opt.prefixModelName+'%d.pt'%(iter/opt.saveModelEvery))
            torch.save(net.state_dict(), opt.prefixModelName+'%d.pt'%(iter/opt.saveModelEvery))
            print('save model: '+opt.prefixModelName+'%d.pt'%(iter/opt.saveModelEvery))

        ##########################################################
        if iter%opt.decLREvery == 0 and opt.using_fixed_learning_rate ==0 :
            opt.lr = opt.lr*0.5
            if opt.lr <0.00001:
                opt.lr=0.00001         

            for param_group in optimizer.param_groups:
                print("current lr is ", param_group["lr"])
                if param_group["lr"] > opt.lr:
                    param_group["lr"] = opt.lr

            print('iteration %d  lr  %f '%(iter, opt.lr))

    print('Finished Training')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuID)
    main()

