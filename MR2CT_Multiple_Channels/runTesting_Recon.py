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
import SimpleITK as sitk
import loadImages
from loadImages import ScanFile, predict_aver_res_fast

parser = argparse.ArgumentParser(description="PyTorch MR2CT")

parser.add_argument("--gpuID", type=int, default=2, help="which gpu to use")
parser.add_argument("--batchSize", type=int, default=10, help="batch size")

parser.add_argument('--cuda', action='store_false', help='using GPU or not')

parser.add_argument("--modelPath", default="./model/t_545000.pt", type=str, help="name of the model to be loaded")
parser.add_argument("--image_path", default="../images_input", type=str, help="image path")
parser.add_argument("--output_path", default="../outputs", type=str, help="output path")

parser.add_argument("--using_r1", default=0, type=int, help="using r1 for training 1(yes) 0(no)")
parser.add_argument("--using_r2", default=0, type=int, help="using r2 for training 1(yes) 0(no)")
parser.add_argument("--using_t1", default=0, type=int, help="using t1 for training 1(yes) 0(no)")

parser.add_argument("--using_fa3e1",  default=0, type=int, help="using fa3e1  for training 1(yes) 0(no)")
parser.add_argument("--using_fa3e2",  default=0, type=int, help="using fa3e2  for training 1(yes) 0(no)")
parser.add_argument("--using_fa15e1", default=0, type=int, help="using fa15e1 for training 1(yes) 0(no)")
parser.add_argument("--using_fa15e2", default=0, type=int, help="using fa15e2 for training 1(yes) 0(no)")
parser.add_argument("--using_dixon1", default=0, type=int, help="using dixon1 for training 1(yes) 0(no)")
parser.add_argument("--using_dixon2", default=0, type=int, help="using dixon2 for training 1(yes) 0(no)")

parser.add_argument("--ndf", default=32, type=int, help="number of features, default 32")
parser.add_argument("--step1", default=4, type=int, help="dim1 interval, default 4")
parser.add_argument("--step2", default=4, type=int, help="dim2 interval, default 4")
parser.add_argument("--step3", default=4, type=int, help="dim3 interval, default 4")

parser.add_argument("--tissue_type_balanced_sampling", default=0, type=int, help="tissue type balanced sampling, default 0")

d1=64
d2=64
d3=64

dPatch  =[d1,d2,d3] # size of patches of input data
dTarget =[64,64,64] # size of pathes of label data

global opt
opt = parser.parse_args()

def main():
    opt = parser.parse_args()
    print(opt)

    num_channels = opt.using_r1+opt.using_r2+opt.using_t1+opt.using_fa3e1+opt.using_fa3e2+opt.using_fa15e1+opt.using_fa15e2+opt.using_dixon1+opt.using_dixon2
    print(num_channels)

    step=[opt.step1, opt.step2, opt.step3]
    print(step)

    netG = NetS(1, num_channels, opt.ndf)
    if opt.cuda:    
        netG.cuda()

    checkpoint = torch.load(opt.modelPath)
    netG.load_state_dict(checkpoint) ###['model'])
    netG.eval()

    ###################################################
    scan_files = ScanFile(opt.image_path, postfix='mk.nii.gz')
    filenames_test = scan_files.scan_files()

    for filename in filenames_test:
        
        start_time = time.time();
	
        file_mk = filename;
        file_r1 = filename.replace('mk.nii.gz', 'r1.nii.gz')
        file_r2 = filename.replace('mk.nii.gz', 'r2.nii.gz')
        file_t1 = filename.replace('mk.nii.gz', 't1.nii.gz')
        file_fa3e1 = filename.replace('mk.nii.gz', 'fa3e1.nii.gz')
        file_fa3e2 = filename.replace('mk.nii.gz', 'fa3e2.nii.gz')
        file_fa15e1 = filename.replace('mk.nii.gz', 'fa15e1.nii.gz')
        file_fa15e2 = filename.replace('mk.nii.gz', 'fa15e2.nii.gz')
        file_dixon1 = filename.replace('mk.nii.gz', 'dixon1.nii.gz')
        file_dixon2 = filename.replace('mk.nii.gz', 'dixon2.nii.gz')

        #####################################################################
        if opt.using_r1 == 1:
            if os.path.isfile(file_r1) == False: 
                continue;

        #######################
        if opt.using_r2 == 1:
            if os.path.isfile(file_r2) == False: 
                continue;

        #######################
        if opt.using_t1 == 1:
            if os.path.isfile(file_t1) == False:
                continue;
		
        #######################
        if opt.using_fa3e1 == 1:
            if os.path.isfile(file_fa3e1) == False:
                continue;

        #######################
        if opt.using_fa3e2 == 1:
            if os.path.isfile(file_fa3e2) == False:
                continue;

        #######################
        if opt.using_fa15e1 == 1:
            if os.path.isfile(file_fa15e1) == False:
                continue;

        #######################
        if opt.using_fa15e2 == 1:
            if os.path.isfile(file_fa15e2) == False:
                continue;
		
        #######################
        if opt.using_dixon1 == 1:
            if os.path.isfile(file_dixon1) == False:
                continue;

        #######################
        if opt.using_dixon2 == 1:
            if os.path.isfile(file_dixon2) == False:
                continue;

        #####################################################################
        file_ct = filename.replace('mk.nii.gz', 'et.nii.gz');

        p, f = os.path.split(file_ct)
        file_ct = opt.output_path + '/' + f

        if opt.using_r1 == 1:
            r1_itk = sitk.ReadImage(file_r1)
        else:
            r1_itk = [];

        if opt.using_r2 == 1:
            r2_itk = sitk.ReadImage(file_r2)
        else:
            r2_itk = []

        if opt.using_t1 == 1:
            t1_itk = sitk.ReadImage(file_t1)
        else:
            t1_itk = []

        if opt.using_fa3e1 == 1:
            fa3e1_itk = sitk.ReadImage(file_fa3e1)
        else:
            fa3e1_itk = []

        if opt.using_fa3e2 == 1:
            fa3e2_itk = sitk.ReadImage(file_fa3e2)
        else:
            fa3e2_itk = []

        if opt.using_fa15e1 == 1:
            fa15e1_itk = sitk.ReadImage(file_fa15e1)
        else:
            fa15e1_itk = []

        if opt.using_fa15e2 == 1:
            fa15e2_itk = sitk.ReadImage(file_fa15e2)
        else:
            fa15e2_itk = []

        if opt.using_dixon1 == 1:
            dixon1_itk = sitk.ReadImage(file_dixon1)
        else:
            dixon1_itk = []

        if opt.using_dixon2 == 1:
            dixon2_itk = sitk.ReadImage(file_dixon2)
        else:
            dixon2_itk = []

        mk_itk = sitk.ReadImage(file_mk)

        spacing   = mk_itk.GetSpacing()
        origin    = mk_itk.GetOrigin()
        direction = mk_itk.GetDirection()

        if opt.using_r1==1:
            r1np = sitk.GetArrayFromImage(r1_itk)
        else:
            r1np = []	
	    
        if opt.using_r2==1:
            r2np = sitk.GetArrayFromImage(r2_itk)
        else:
            r2np = []

        if opt.using_t1==1:
            t1np = sitk.GetArrayFromImage(t1_itk)
        else:
            t1np = []

        if opt.using_fa3e1==1:
            fa3e1np = sitk.GetArrayFromImage(fa3e1_itk)
        else:
            fa3e1np = []

        if opt.using_fa3e2==1:
            fa3e2np = sitk.GetArrayFromImage(fa3e2_itk)
        else:
            fa3e2np = []

        if opt.using_fa15e1==1:
            fa15e1np = sitk.GetArrayFromImage(fa15e1_itk)
        else:
            fa15e1np = []

        if opt.using_fa15e2==1:
            fa15e2np = sitk.GetArrayFromImage(fa15e2_itk)
        else:
            fa15e2np = []

        if opt.using_dixon1==1:
            dixon1np = sitk.GetArrayFromImage(dixon1_itk)
        else:
            dixon1np = []

        if opt.using_dixon2==1:
            dixon2np = sitk.GetArrayFromImage(dixon2_itk)
        else:
            dixon2np = []

        mknp = sitk.GetArrayFromImage(mk_itk)

        #####################################################################
        ct_out = predict_aver_res_fast(r1np, r2np, t1np, fa3e1np, fa3e2np, fa15e1np, fa15e2np, dixon1np, dixon2np, mknp, opt, dPatch, dTarget, step, netG, 5)
        ct_out[np.where(mknp==0)]=0

        ct_out.astype(np.float)
        volout = sitk.GetImageFromArray(ct_out)
        volout.SetSpacing(spacing)
        volout.SetOrigin(origin)
        volout.SetDirection(direction)
        sitk.WriteImage(volout, file_ct)

        # ## Chunwei Add, scale back to HU
        file_ct_hu = file_ct.replace('et.nii.gz', 'qt.nii.gz')

        # # ct_mean = 113.1875
        # # ct_std = 424.5913
        ct_mean = 111.1232
        ct_std = 436.3707
        ct_out_hu = ct_out*5*ct_std + ct_mean

        ct_out_hu[np.where(mknp==0)] = -1000
         
        volout = sitk.GetImageFromArray(ct_out_hu)
        volout.SetSpacing(spacing)
        volout.SetOrigin(origin)
        volout.SetDirection(direction)
        sitk.WriteImage(volout, file_ct_hu)
        # ## Chunwei add ends


        print("--------------------%s seconds "%(time.time()-start_time))

if __name__ == '__main__':
#     testGradients()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuID)
    main()
