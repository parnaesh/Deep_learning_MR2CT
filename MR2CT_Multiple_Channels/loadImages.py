    
import SimpleITK as sitk
from multiprocessing import Pool
import os, argparse
import numpy as np
import math
import torch
import time

class ScanFile(object):   
    def __init__(self,directory,prefix=None,postfix=None):  
        self.directory=directory  
        self.prefix=prefix  
        self.postfix=postfix  
          
    def scan_files(self):    
        files_list=[]    
            
        for dirpath,dirnames,filenames in os.walk(self.directory):   
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''  
            for special_file in filenames:    
                if self.postfix:  
                    if  special_file.endswith(self.postfix):    
                        files_list.append(os.path.join(dirpath,special_file))    
                elif self.prefix:    
                    if special_file.startswith(self.prefix):  
                        files_list.append(os.path.join(dirpath,special_file))    
                else:    
                    files_list.append(os.path.join(dirpath,special_file))    
                                  
        return files_list    
      
    def scan_subdir(self):  
        subdir_list=[]  
        for dirpath,dirnames,files in os.walk(self.directory):  
            subdir_list.append(dirpath)  
        return subdir_list      
    
####################################################################################### 
def calculate_patch_bounds(patch_size):
    if patch_size%2==1:
        ss = (patch_size-1)//2      
        es = (patch_size-1)//2+1
    else:
        ss = patch_size//2-1      
        es = patch_size//2+1

    return ss, es

#########################################################################################################
def expand_images_fitsize(ctnp, r1np, r2np, t1np, fa3e1np, fa3e2np, fa15e1np, fa15e2np, dixon1np, dixon2np, mknp, dPatch, dTarget, opt):
  
    [dimz,dimx,dimy]=ctnp.shape

    m1_f_low, m1_f_up = calculate_patch_bounds(dPatch[0])
    m2_f_low, m2_f_up = calculate_patch_bounds(dPatch[1])
    m3_f_low, m3_f_up = calculate_patch_bounds(dPatch[2])

    m1_t_low, m1_t_up = calculate_patch_bounds(dTarget[0])
    m2_t_low, m2_t_up = calculate_patch_bounds(dTarget[1])
    m3_t_low, m3_t_up = calculate_patch_bounds(dTarget[2])

    #print('feature m1 %d %d, m2 %d %d, m3 %d %d'%(m1_f_low, m1_f_up, m2_f_low, m2_f_up, m3_f_low, m3_f_up));
    #print('target  m1 %d %d, m2 %d %d, m3 %d %d'%(m1_t_low, m1_t_up, m2_t_low, m2_t_up, m3_t_low, m3_t_up));

    if opt.using_r1==1:
        r1_pad=np.zeros([dimz+dPatch[0], dimx+dPatch[1], dimy+dPatch[2]], dtype=np.float32)
        r1_pad[int(m1_f_low):int(dimz+m1_f_low), int(m2_f_low):int(dimx+m2_f_low), int(m3_f_low):int(dimy+m3_f_low)]=r1np
    else:
        r1_pad = []

    if opt.using_r2==1:
        r2_pad=np.zeros([dimz+dPatch[0], dimx+dPatch[1], dimy+dPatch[2]], dtype=np.float32)
        r2_pad[int(m1_f_low):int(dimz+m1_f_low), int(m2_f_low):int(dimx+m2_f_low), int(m3_f_low):int(dimy+m3_f_low)]=r2np
    else:
        r2_pad = []

    if opt.using_t1==1:
        t1_pad=np.zeros([dimz+dPatch[0], dimx+dPatch[1], dimy+dPatch[2]], dtype=np.float32)
        t1_pad[int(m1_f_low):int(dimz+m1_f_low), int(m2_f_low):int(dimx+m2_f_low), int(m3_f_low):int(dimy+m3_f_low)]=t1np
    else:
        t1_pad = []

    if opt.using_fa3e1==1:
        fa3e1_pad=np.zeros([dimz+dPatch[0], dimx+dPatch[1], dimy+dPatch[2]], dtype=np.float32)
        fa3e1_pad[int(m1_f_low):int(dimz+m1_f_low), int(m2_f_low):int(dimx+m2_f_low), int(m3_f_low):int(dimy+m3_f_low)]=fa3e1np
    else:
        fa3e1_pad = []

    if opt.using_fa3e2==1:
        fa3e2_pad=np.zeros([dimz+dPatch[0], dimx+dPatch[1], dimy+dPatch[2]], dtype=np.float32)
        fa3e2_pad[int(m1_f_low):int(dimz+m1_f_low), int(m2_f_low):int(dimx+m2_f_low), int(m3_f_low):int(dimy+m3_f_low)]=fa3e2np
    else:
        fa3e2_pad = []

    if opt.using_fa15e1==1:
        fa15e1_pad=np.zeros([dimz+dPatch[0], dimx+dPatch[1], dimy+dPatch[2]], dtype=np.float32)
        fa15e1_pad[int(m1_f_low):int(dimz+m1_f_low), int(m2_f_low):int(dimx+m2_f_low), int(m3_f_low):int(dimy+m3_f_low)]=fa15e1np
    else:
        fa15e1_pad = []

    if opt.using_fa15e2==1:
        fa15e2_pad=np.zeros([dimz+dPatch[0], dimx+dPatch[1], dimy+dPatch[2]], dtype=np.float32)
        fa15e2_pad[int(m1_f_low):int(dimz+m1_f_low), int(m2_f_low):int(dimx+m2_f_low), int(m3_f_low):int(dimy+m3_f_low)]=fa15e2np
    else:
        fa15e2_pad = []

    if opt.using_dixon1==1:
        dixon1_pad=np.zeros([dimz+dPatch[0], dimx+dPatch[1], dimy+dPatch[2]], dtype=np.float32)
        dixon1_pad[int(m1_f_low):int(dimz+m1_f_low), int(m2_f_low):int(dimx+m2_f_low), int(m3_f_low):int(dimy+m3_f_low)]=dixon1np
    else:
        dixon1_pad = []

    if opt.using_dixon2==1:
        dixon2_pad=np.zeros([dimz+dPatch[0], dimx+dPatch[1], dimy+dPatch[2]], dtype=np.float32)
        dixon2_pad[int(m1_f_low):int(dimz+m1_f_low), int(m2_f_low):int(dimx+m2_f_low), int(m3_f_low):int(dimy+m3_f_low)]=dixon2np
    else:
        dixon2_pad = []

    mk_pad=np.zeros([dimz+dPatch[0], dimx+dPatch[1], dimy+dPatch[2]], dtype=np.float32)
    mk_pad[int(m1_f_low):int(dimz+m1_f_low), int(m2_f_low):int(dimx+m2_f_low), int(m3_f_low):int(dimy+m3_f_low)]=mknp

    ct_pad=np.zeros([dimz+dPatch[0], dimx+dPatch[1], dimy+dPatch[2]], dtype=np.float32)
    ct_pad[int(m1_t_low):int(dimz+m1_t_low), int(m2_t_low):int(dimx+m2_t_low), int(m3_t_low):int(dimy+m3_t_low)]=ctnp

    list1_air = []
    list2_air = []
    list3_air = []

    list1_brain = []
    list2_brain = []
    list3_brain = []

    list1_bone = []
    list2_bone = []
    list3_bone = []

    if opt.tissue_type_balanced_sampling == 1:   

        print('balanced sampling...')
	
        for i in range(int(m1_f_low), int(m1_f_low + dimz)):
            for j in range(int(m2_f_low), int(m2_f_low + dimx)):
                for k in range(int(m3_f_low), int(m3_f_low + dimy)):

                    if mk_pad[i,j,k] ==1:
                        list1_air.append(i);
                        list2_air.append(j);
                        list3_air.append(k);

                    if mk_pad[i,j,k] ==3:
                        list1_brain.append(i);
                        list2_brain.append(j);
                        list3_brain.append(k);

                    if mk_pad[i,j,k] ==3:
                        list1_bone.append(i);
                        list2_bone.append(j);
                        list3_bone.append(k);
    else:   
        index=0

        print('random sampling...')

        for i in range(int(m1_f_low), int(m1_f_low + dimz)):
            for j in range(int(m2_f_low), int(m2_f_low + dimx)):
                for k in range(int(m3_f_low), int(m3_f_low + dimy)):

                    if mk_pad[i,j,k] >0:
                        index_index = index%3

                        if index_index==0:
                            list1_air.append(i);
                            list2_air.append(j);
                            list3_air.append(k);

                        if index_index==1:
                            list1_brain.append(i);
                            list2_brain.append(j);
                            list3_brain.append(k);
 
                        if index_index==2:
                            list1_bone.append(i);
                            list2_bone.append(j);
                            list3_bone.append(k);

                        index=index+1

    return ct_pad, r1_pad, r2_pad, t1_pad, fa3e1_pad, fa3e2_pad, fa15e1_pad, fa15e2_pad, dixon1_pad, dixon2_pad, mk_pad, list1_air, list2_air, list3_air, list1_brain, list2_brain, list3_brain, list1_bone, list2_bone, list3_bone


########################################################################################################
def load_images(path_train, dPatch, dTarget, opt):

    scan_train = ScanFile(path_train, postfix = 'ct.nii.gz')  
    filenames_train = scan_train.scan_files()  

    ct_train = []

    r1_train = []    
    r2_train = []
    t1_train = []

    fa3e1_train = []
    fa3e2_train = []

    fa15e1_train = []
    fa15e2_train = []

    dixon1_train = []
    dixon2_train = []

    mk_train = []

    l1_train_air = []    
    l2_train_air = []    
    l3_train_air = []    

    l1_train_brain = []    
    l2_train_brain = []    
    l3_train_brain = []    

    l1_train_bone = []    
    l2_train_bone = []    
    l3_train_bone = []    

    ###############################################################
    index_input = 0
    for filename in filenames_train:         
           
        ct_fn = filename
        ct = sitk.ReadImage(ct_fn)
        ctnp = sitk.GetArrayFromImage(ct)

        ######################################################################################
        if opt.using_r1 == 1:
            r1_fn = filename.replace('ct.nii.gz','r1.nii.gz')  
            if os.path.isfile(r1_fn) == False: 
                continue;

        #######################
        if opt.using_r2 == 1:
            r2_fn = filename.replace('ct.nii.gz','r2.nii.gz')  
            if os.path.isfile(r2_fn) == False: 
                continue;

        #######################
        if opt.using_t1 == 1:
            t1_fn = filename.replace('ct.nii.gz','t1.nii.gz')
            if os.path.isfile(t1_fn) == False:
                continue;
		
        #######################
        if opt.using_fa3e1 == 1:
            fa3e1_fn = filename.replace('ct.nii.gz','fa3e1.nii.gz')
            if os.path.isfile(fa3e1_fn) == False:
                continue;

        #######################
        if opt.using_fa3e2 == 1:
            fa3e2_fn = filename.replace('ct.nii.gz','fa3e2.nii.gz')
            if os.path.isfile(fa3e2_fn) == False:
                continue;

        #######################
        if opt.using_fa15e1 == 1:
            fa15e1_fn = filename.replace('ct.nii.gz','fa15e1.nii.gz')
            if os.path.isfile(fa15e1_fn) == False:
                print('return here %s'%(fa15e1_fn))
                continue;

        #######################
        if opt.using_fa15e2 == 1:
            fa15e2_fn = filename.replace('ct.nii.gz','fa15e2.nii.gz')
            if os.path.isfile(fa15e2_fn) == False:
                continue;
		
        #######################
        if opt.using_dixon1 == 1:
            dixon1_fn = filename.replace('ct.nii.gz','dixon1.nii.gz')
            if os.path.isfile(dixon1_fn) == False:
                continue;

        #######################
        if opt.using_dixon2 == 1:
            dixon2_fn = filename.replace('ct.nii.gz','dixon2.nii.gz')
            if os.path.isfile(dixon2_fn) == False:
                continue;

        ######################################################################################
        print('train:  %d  %s'%(index_input, filename))

        if opt.using_r1 == 1:
            r1_fn = filename.replace('ct.nii.gz','r1.nii.gz')  
            r1 = sitk.ReadImage(r1_fn)
            r1np = sitk.GetArrayFromImage(r1)
        else:
            r1np = []

        #######################
        if opt.using_r2 == 1:
            r2_fn = filename.replace('ct.nii.gz','r2.nii.gz')
            r2 = sitk.ReadImage(r2_fn)
            r2np = sitk.GetArrayFromImage(r2)
        else:
            r2np = []

        #######################
        if opt.using_t1 == 1:
            t1_fn = filename.replace('ct.nii.gz','t1.nii.gz')
            t1 = sitk.ReadImage(t1_fn)
            t1np = sitk.GetArrayFromImage(t1)
        else:
            t1np = []

        #######################
        if opt.using_fa3e1 == 1:
            fa3e1_fn = filename.replace('ct.nii.gz','fa3e1.nii.gz')
            fa3e1 = sitk.ReadImage(fa3e1_fn)
            fa3e1np = sitk.GetArrayFromImage(fa3e1)
        else:
            fa3e1np = []

        #######################
        if opt.using_fa3e2 == 1:
            fa3e2_fn = filename.replace('ct.nii.gz','fa3e2.nii.gz')
            fa3e2 = sitk.ReadImage(fa3e2_fn)
            fa3e2np = sitk.GetArrayFromImage(fa3e2)
        else:
            fa3e2np = []

        #######################
        if opt.using_fa15e1 == 1:
            fa15e1_fn = filename.replace('ct.nii.gz','fa15e1.nii.gz')
            fa15e1 = sitk.ReadImage(fa15e1_fn)
            fa15e1np = sitk.GetArrayFromImage(fa15e1)
        else:
            fa15e1np = []

        #######################
        if opt.using_fa15e2 == 1:
            fa15e2_fn = filename.replace('ct.nii.gz','fa15e2.nii.gz')
            fa15e2 = sitk.ReadImage(fa15e2_fn)
            fa15e2np = sitk.GetArrayFromImage(fa15e2)
        else:
            fa15e2np = []	

        #######################
        if opt.using_dixon1 == 1:
            dixon1_fn = filename.replace('ct.nii.gz','dixon1.nii.gz')
            dixon1 = sitk.ReadImage(dixon1_fn)
            dixon1np = sitk.GetArrayFromImage(dixon1)
        else:
            dixon1np = []

        #######################
        if opt.using_dixon2 == 1:
            dixon2_fn = filename.replace('ct.nii.gz','dixon2.nii.gz')
            dixon2 = sitk.ReadImage(dixon2_fn)
            dixon2np = sitk.GetArrayFromImage(dixon2)
        else:
            dixon2np = []
	       
        #######################
        mk_fn = filename.replace('ct.nii.gz','mk.nii.gz')
        mk = sitk.ReadImage(mk_fn)
        mknp = sitk.GetArrayFromImage(mk)
  
        ct_pad, r1_pad, r2_pad, t1_pad, fa3e1_pad, fa3e2_pad, fa15e1_pad, fa15e2_pad, dixon1_pad, dixon2_pad, mk_pad, l1_air, l2_air, l3_air, l1_brain, l2_brain, l3_brain, l1_bone, l2_bone, l3_bone = expand_images_fitsize(ctnp, r1np, r2np, t1np, fa3e1np, fa3e2np, fa15e1np, fa15e2np, dixon1np, dixon2np, mknp, dPatch, dTarget, opt)

        ct_train.append(ct_pad)

        if opt.using_r1==1:
            r1_train.append(r1_pad)

        if opt.using_r2==1:
            r2_train.append(r2_pad)

        if opt.using_t1==1:
            t1_train.append(t1_pad)

        if opt.using_fa3e1==1:
            fa3e1_train.append(fa3e1_pad)

        if opt.using_fa3e2==1:
            fa3e2_train.append(fa3e2_pad)

        if opt.using_fa15e1==1:
            fa15e1_train.append(fa15e1_pad)

        if opt.using_fa15e2==1:
            fa15e2_train.append(fa15e2_pad)

        if opt.using_dixon1==1:
            dixon1_train.append(dixon1_pad)

        if opt.using_dixon2==1:
            dixon2_train.append(dixon2_pad)

        mk_train.append(mk_pad)
                
        l1_train_air.append(l1_air)
        l2_train_air.append(l2_air)
        l3_train_air.append(l3_air)

        l1_train_brain.append(l1_brain)
        l2_train_brain.append(l2_brain)
        l3_train_brain.append(l3_brain)

        l1_train_bone.append(l1_bone)
        l2_train_bone.append(l2_bone)
        l3_train_bone.append(l3_bone)

        index_input = index_input + 1

    return ct_train, r1_train, r2_train, t1_train, fa3e1_train, fa3e2_train, fa15e1_train, fa15e2_train, dixon1_train, dixon2_train, mk_train, l1_train_air, l2_train_air, l3_train_air, l1_train_brain, l2_train_brain, l3_train_brain, l1_train_bone, l2_train_bone, l3_train_bone  

#######################################################################################

def extract_batch_single_subject(ct, r1, r2, t1, fa3e1, fa3e2, fa15e1, fa15e2, dixon1, dixon2, l1, l2, l3, index, batch_size, dPatch, dTarget):
    
    m1_f_low, m1_f_up = calculate_patch_bounds(dPatch[0])
    m2_f_low, m2_f_up = calculate_patch_bounds(dPatch[1])
    m3_f_low, m3_f_up = calculate_patch_bounds(dPatch[2])

    m1_t_low, m1_t_up = calculate_patch_bounds(dTarget[0])
    m2_t_low, m2_t_up = calculate_patch_bounds(dTarget[1])
    m3_t_low, m3_t_up = calculate_patch_bounds(dTarget[2])

    if len(r1) > 0:
        r1_tmp=np.zeros([batch_size, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    else:
        r1_tmp = []

    if len(r2)>0:
        r2_tmp=np.zeros([batch_size, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    else:
        r2_tmp = []

    if len(t1)>0:
        t1_tmp=np.zeros([batch_size, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    else:
        t1_tmp = []

    if len(fa3e1)>0:
        fa3e1_tmp=np.zeros([batch_size, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    else:
        fa3e1_tmp = []

    if len(fa3e2)>0:
        fa3e2_tmp=np.zeros([batch_size, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    else:
        fa3e2_tmp = []

    if len(fa15e1)>0:
        fa15e1_tmp=np.zeros([batch_size, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    else:
        fa15e1_tmp = []

    if len(fa15e2)>0:
        fa15e2_tmp=np.zeros([batch_size, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    else:
        fa15e2_tmp = []

    if len(dixon1)>0:
        dixon1_tmp=np.zeros([batch_size, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    else:
        dixon1_tmp = []

    if len(dixon2)>0:
        dixon2_tmp=np.zeros([batch_size, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    else:
        dixon2_tmp = []

    ct_tmp=np.zeros([batch_size, 1, dTarget[0], dTarget[1], dTarget[2]], dtype=np.float32)

    #print('nums_extracted  %f '%(nums_extracted))
    #print('feature m1 %d %d, m2 %d %d m3 %d %d'%(m1_f_low, m1_f_up, m2_f_low, m2_f_up, m3_f_low, m3_f_up))
    #print('target  m1 %d %d, m2 %d %d m3 %d %d'%(m1_t_low, m1_t_up, m2_t_low, m2_t_up, m3_t_low, m3_t_up))
    #print(ct[0].shape)

    num = len(l1[index]) 
    ind = np.random.choice(num, size=batch_size, replace=False)

    for i in range(0, batch_size):
        i1=l1[index][ind[i]]
        i2=l2[index][ind[i]]
        i3=l3[index][ind[i]]
	
        ct_tmp[i,0,:,:,:] = ct[index][int(i1-m1_t_low):int(i1+m1_t_up),int(i2-m2_t_low):int(i2+m2_t_up),int(i3-m3_t_low):i3+int(m3_t_up)]

        if len(r1)>0:
            r1_tmp[i,0,:,:,:] = r1[index][int(i1-m1_f_low):int(i1+m1_f_up),int(i2-m2_f_low):int(i2+m2_f_up),int(i3-m3_f_low):i3+int(m3_f_up)]

        if len(r2)>0:
            r2_tmp[i,0,:,:,:] = r2[index][int(i1-m1_f_low):int(i1+m1_f_up),int(i2-m2_f_low):int(i2+m2_f_up),int(i3-m3_f_low):i3+int(m3_f_up)]

        if len(t1)>0:
            t1_tmp[i,0,:,:,:] = t1[index][int(i1-m1_f_low):int(i1+m1_f_up),int(i2-m2_f_low):int(i2+m2_f_up),int(i3-m3_f_low):i3+int(m3_f_up)]

        if len(fa3e1)>0:
            fa3e1_tmp[i,0,:,:,:] = fa3e1[index][int(i1-m1_f_low):int(i1+m1_f_up),int(i2-m2_f_low):int(i2+m2_f_up),int(i3-m3_f_low):i3+int(m3_f_up)]

        if len(fa3e2)>0:
            fa3e2_tmp[i,0,:,:,:] = fa3e2[index][int(i1-m1_f_low):int(i1+m1_f_up),int(i2-m2_f_low):int(i2+m2_f_up),int(i3-m3_f_low):i3+int(m3_f_up)]

        if len(fa15e1)>0:
            fa15e1_tmp[i,0,:,:,:] = fa15e1[index][int(i1-m1_f_low):int(i1+m1_f_up),int(i2-m2_f_low):int(i2+m2_f_up),int(i3-m3_f_low):i3+int(m3_f_up)]
    
        if len(fa15e2)>0:
            fa15e2_tmp[i,0,:,:,:] = fa15e2[index][int(i1-m1_f_low):int(i1+m1_f_up),int(i2-m2_f_low):int(i2+m2_f_up),int(i3-m3_f_low):i3+int(m3_f_up)]

        if len(dixon1)>0:
            dixon1_tmp[i,0,:,:,:] = dixon1[index][int(i1-m1_f_low):int(i1+m1_f_up),int(i2-m2_f_low):int(i2+m2_f_up),int(i3-m3_f_low):i3+int(m3_f_up)]

        if len(dixon2)>0:
            dixon2_tmp[i,0,:,:,:] = dixon2[index][int(i1-m1_f_low):int(i1+m1_f_up),int(i2-m2_f_low):int(i2+m2_f_up),int(i3-m3_f_low):i3+int(m3_f_up)]

    return ct_tmp, r1_tmp, r2_tmp, t1_tmp, fa3e1_tmp, fa3e2_tmp, fa15e1_tmp, fa15e2_tmp, dixon1_tmp, dixon2_tmp 

####################################################################################
def predict_aver_res_fast(r1np, r2np, t1np, fa3e1np, fa3e2np, fa15e1np, fa15e2np, dixon1np, dixon2np, mknp, opt, dPatch, dTarget,step, netG, whichNet):

    num_channels = opt.using_r1+opt.using_r2+opt.using_t1+opt.using_fa3e1+opt.using_fa3e2+opt.using_fa15e1+opt.using_fa15e2+opt.using_dixon1+opt.using_dixon2

    [dimz,dimx,dimy]=mknp.shape

    m1_f_low, m1_f_up = calculate_patch_bounds(dPatch[0])
    m2_f_low, m2_f_up = calculate_patch_bounds(dPatch[1])
    m3_f_low, m3_f_up = calculate_patch_bounds(dPatch[2])

    m1_t_low, m1_t_up = calculate_patch_bounds(dTarget[0])
    m2_t_low, m2_t_up = calculate_patch_bounds(dTarget[1])
    m3_t_low, m3_t_up = calculate_patch_bounds(dTarget[2])

    ###############################################################################
    cx = int(dPatch[0]/2); cy = int(dPatch[1]/2); cz = int(dPatch[2]/2)
    sx = int(dPatch[0]/4); ex = int(dPatch[0]/4)
    sy = int(dPatch[1]/4); ey = int(dPatch[1]/4)
    sz = int(dPatch[2]/4); ez = int(dPatch[2]/4)

    ###############################################################################
    mask_in_use = np.zeros([dPatch[0], dPatch[1], dPatch[2]])
    sx = int(round(dPatch[0]/4)); ex = int(round(dPatch[0]*3/4))+1
    sy = int(round(dPatch[1]/4)); ey = int(round(dPatch[1]*3/4))+1
    sz = int(round(dPatch[2]/4)); ez = int(round(dPatch[2]*3/4))+1

    mask_in_use[sx:ex,sy:ey,sz:ez] = 1
    
    m1_f_low_output, m1_f_up_output = calculate_patch_bounds(dPatch[0]/2)
    m2_f_low_output, m2_f_up_output = calculate_patch_bounds(dPatch[1]/2)
    m3_f_low_output, m3_f_up_output = calculate_patch_bounds(dPatch[2]/2)

    m1_t_low_output, m1_t_up_output = calculate_patch_bounds(dTarget[0]/2)
    m2_t_low_output, m2_t_up_output = calculate_patch_bounds(dTarget[1]/2)
    m3_t_low_output, m3_t_up_output = calculate_patch_bounds(dTarget[2]/2)

    ################################################################################
    nmnp = np.zeros([dimz, dimx, dimy])
   
    nm_pad, r1_pad, r2_pad, t1_pad, fa3e1_pad, fa3e2_pad, fa15e1_pad, fa15e2_pad, dixon1_pad, dixon2_pad, mk_pad, l1_air, l2_air, l3_air, l1_brain, l2_brain, l3_brain, l1_bone, l2_bone, l3_bone = expand_images_fitsize(nmnp, r1np, r2np, t1np, fa3e1np, fa3e2np, fa15e1np, fa15e2np, dixon1np, dixon2np, mknp, dPatch, dTarget, opt)

    dimz_new = dimz+dPatch[0]
    dimx_new = dimx+dPatch[1]
    dimy_new = dimy+dPatch[2]

    ct_pad = np.zeros([dimz_new, dimx_new, dimy_new])

    list_i = []
    list_j = []
    list_k = []

    mid_slice = dimz//2

    for i in range(m1_f_low, m1_f_low+dimz, step[0]):
    #for i in range(mid_slice, mid_slice+1):
        for j in range(m2_f_low, m2_f_low+dimx, step[1]):
            for k in range(m3_f_low, m3_f_low+dimy, step[2]):            

                if mk_pad[i,j,k] < 0.1:
                    continue

                list_i.append(i)
                list_j.append(j)
                list_k.append(k)

    num_total = len(list_i)
    num_each = 10
 
    #r1_tmp=np.zeros([num_each, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float16)
    #r2_tmp=np.zeros([num_each, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float16)

    r1_tmp=np.zeros([num_each, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    r2_tmp=np.zeros([num_each, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    t1_tmp=np.zeros([num_each, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    fa3e1_tmp=np.zeros([num_each, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    fa3e2_tmp=np.zeros([num_each, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    fa15e1_tmp=np.zeros([num_each, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    fa15e2_tmp=np.zeros([num_each, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    dixon1_tmp=np.zeros([num_each, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)
    dixon2_tmp=np.zeros([num_each, 1, dPatch[0],  dPatch[1],  dPatch[2]],  dtype=np.float32)

    num_loops = math.ceil(num_total/num_each)

    for i in range(num_loops):

        indexs = i*num_each
        indexe = (i+1)*num_each

        if indexe > num_total:
            indexe = num_total

        num_current=indexe-indexs

        ii = list_i[indexs:indexe]
        jj = list_j[indexs:indexe]
        kk = list_k[indexs:indexe]

        for j in range(num_current):
            if opt.using_r1==1:
                r1_tmp[j, 0, :, :, :] = r1_pad[int(ii[j]-m1_f_low):int(ii[j]+m1_f_up),int(jj[j]-m2_f_low):int(jj[j]+m2_f_up),int(kk[j]-m3_f_low):int(kk[j]+m3_f_up)]

            if opt.using_r2==1:
                r2_tmp[j, 0, :, :, :] = r2_pad[int(ii[j]-m1_f_low):int(ii[j]+m1_f_up),int(jj[j]-m2_f_low):int(jj[j]+m2_f_up),int(kk[j]-m3_f_low):int(kk[j]+m3_f_up)]

            if opt.using_t1==1:
                t1_tmp[j, 0, :, :, :] = t1_pad[int(ii[j]-m1_f_low):int(ii[j]+m1_f_up),int(jj[j]-m2_f_low):int(jj[j]+m2_f_up),int(kk[j]-m3_f_low):int(kk[j]+m3_f_up)]

            if opt.using_fa3e1==1:
                fa3e1_tmp[j, 0, :, :, :] = fa3e1_pad[int(ii[j]-m1_f_low):int(ii[j]+m1_f_up),int(jj[j]-m2_f_low):int(jj[j]+m2_f_up),int(kk[j]-m3_f_low):int(kk[j]+m3_f_up)]
   
            if opt.using_fa3e2==1:
                fa3e2_tmp[j, 0, :, :, :] = fa3e2_pad[int(ii[j]-m1_f_low):int(ii[j]+m1_f_up),int(jj[j]-m2_f_low):int(jj[j]+m2_f_up),int(kk[j]-m3_f_low):int(kk[j]+m3_f_up)]

            if opt.using_fa15e1==1:
                fa15e1_tmp[j, 0, :, :, :] = fa15e1_pad[int(ii[j]-m1_f_low):int(ii[j]+m1_f_up),int(jj[j]-m2_f_low):int(jj[j]+m2_f_up),int(kk[j]-m3_f_low):int(kk[j]+m3_f_up)]

            if opt.using_fa15e2==1:
                fa15e2_tmp[j, 0, :, :, :] = fa15e2_pad[int(ii[j]-m1_f_low):int(ii[j]+m1_f_up),int(jj[j]-m2_f_low):int(jj[j]+m2_f_up),int(kk[j]-m3_f_low):int(kk[j]+m3_f_up)]

            if opt.using_dixon1==1:
                dixon1_tmp[j, 0, :, :, :] = dixon1_pad[int(ii[j]-m1_f_low):int(ii[j]+m1_f_up),int(jj[j]-m2_f_low):int(jj[j]+m2_f_up),int(kk[j]-m3_f_low):int(kk[j]+m3_f_up)]

            if opt.using_dixon2==1:
                dixon2_tmp[j, 0, :, :, :] = dixon2_pad[int(ii[j]-m1_f_low):int(ii[j]+m1_f_up),int(jj[j]-m2_f_low):int(jj[j]+m2_f_up),int(kk[j]-m3_f_low):int(kk[j]+m3_f_up)]

        if opt.using_r1==1:
            r1_input = r1_tmp.astype(float)
            r1_input = torch.from_numpy(r1_input)
            r1_input = r1_input.float()

        if opt.using_r2==1:
            r2_input = r2_tmp.astype(float)
            r2_input = torch.from_numpy(r2_input)
            r2_input = r2_input.float()

        if opt.using_t1==1:
            t1_input = t1_tmp.astype(float)
            t1_input = torch.from_numpy(t1_input)
            t1_input = t1_input.float()

        if opt.using_fa3e1==1:
            fa3e1_input = fa3e1_tmp.astype(float)
            fa3e1_input = torch.from_numpy(fa3e1_input)
            fa3e1_input = fa3e1_input.float()

        if opt.using_fa3e2==1:
            fa3e2_input = fa3e2_tmp.astype(float)
            fa3e2_input = torch.from_numpy(fa3e2_input)
            fa3e2_input = fa3e2_input.float()

        if opt.using_fa15e1==1:
            fa15e1_input = fa15e1_tmp.astype(float)
            fa15e1_input = torch.from_numpy(fa15e1_input)
            fa15e1_input = fa15e1_input.float()

        if opt.using_fa15e2==1:
            fa15e2_input = fa15e2_tmp.astype(float)
            fa15e2_input = torch.from_numpy(fa15e2_input)
            fa15e2_input = fa15e2_input.float()

        if opt.using_dixon1==1:
            dixon1_input = dixon1_tmp.astype(float)
            dixon1_input = torch.from_numpy(dixon1_input)
            dixon1_input = dixon1_input.float()

        if opt.using_dixon2==1:
            dixon2_input = dixon2_tmp.astype(float)
            dixon2_input = torch.from_numpy(dixon2_input)
            dixon2_input = dixon2_input.float()

        if num_channels ==1 :
            if opt.using_r1==1:
                source = r1_input

            if opt.using_r2==1:
                source = r2_input

            if opt.using_t1==1:
                source = t1_input

        if num_channels ==2 :
            if opt.using_dixon1==1 and opt.using_dixon2==1:
                source = torch.cat((dixon1_input, dixon2_input), dim=1)

            if opt.using_fa3e1==1 and opt.using_fa15e1==1:
                source = torch.cat((fa3e1_input, fa15e1_input), dim=1)

        if num_channels ==3 :
            if opt.using_r2==1 and opt.using_fa15e1==1 and opt.using_fa15e2==1:
                source = torch.cat((r2_input, fa15e1_input), dim=1)
                source = torch.cat((source, fa15e2_input), dim=1)

            if opt.using_r1==1 and opt.using_fa3e1==1 and opt.using_fa15e1==1:
                source = torch.cat((r1_input, fa3e1_input), dim=1)
                source = torch.cat((source, fa15e1_input), dim=1)

        if num_channels ==7 :
            if opt.using_r1==1 and opt.using_r2==1 and opt.using_t1==1 and opt.using_fa3e1==1 and opt.using_fa3e2==1 and opt.using_fa15e1==1 and opt.using_fa15e2==1 :
                source = torch.cat((r1_input, r2_input), dim=1)
                source = torch.cat((source, t1_input), dim=1)
                source = torch.cat((source, fa3e1_input), dim=1)
                source = torch.cat((source, fa3e2_input), dim=1)
                source = torch.cat((source, fa15e1_input), dim=1)
                source = torch.cat((source, fa15e2_input), dim=1)

        residual_source = source

        source = source.cuda()
        residual_source = residual_source.cuda()

        if whichNet==5:
            outputG = netG(source)
        else:
            outputG = netG(source, residual_source)

        #outputG = netG(source)
        outputG = np.squeeze(outputG.detach())

        for j in range(num_current):	    

            ttt=np.squeeze(outputG[j,:,:,:])
            #ttt=np.multiply(ttt,mask_in_use)
            ttt=np.multiply(ttt.cpu().numpy(),mask_in_use)
            ct_pad[int(ii[j]-m1_f_low):int(ii[j]+m1_f_up),int(jj[j]-m2_f_low):int(jj[j]+m2_f_up),int(kk[j]-m3_f_low):int(kk[j]+m3_f_up)] += ttt
            nm_pad[int(ii[j]-m1_f_low):int(ii[j]+m1_f_up),int(jj[j]-m2_f_low):int(jj[j]+m2_f_up),int(kk[j]-m3_f_low):int(kk[j]+m3_f_up)] += mask_in_use
		
    ct_pad = np.divide(ct_pad, (nm_pad+1e-6))
    ct_out = ct_pad[int(m1_f_low):int(dimz+m1_f_low), int(m2_f_low):int(dimx+m2_f_low), int(m3_f_low):int(dimy+m3_f_low)]
    return ct_out

