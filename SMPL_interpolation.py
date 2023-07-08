# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:01:12 2022

@author: Lenovo
"""
from util_smpl import axisangle_quat 
import torch
import numpy as np
#from SMPL import getSMPL


def slerp(starting_q, ending_q, t ):

    #cosa = np.dot(starting_q,ending_q)   
    cosa=torch.sum(starting_q*ending_q,axis=1,keepdims=True) #24,1
    
    
    # If the dot product is negative, the quaternions have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    reverse_by_false=torch.ones(cosa.shape)
    torch.where(cosa<0.0,reverse_by_false,-reverse_by_false)
    
    cosa=cosa*reverse_by_false
    ending_q=ending_q*reverse_by_false
    
    """
    if ( cosa < 0.0):      #24,
        ending= -ending
        cosa = -cosa
    """
    
    k0=torch.zeros(cosa.shape)
    k1=torch.zeros(cosa.shape)
    
    sina = torch.sqrt( 1.0 - cosa*cosa )
    a = torch.atan2( sina, cosa )
    
    k0 = torch.sin((1.0 - t)*a)  / sina
    k1 = torch.sin(t*a) / sina

    print(sina)
    input()
    print(k0)
    print(k1)
    input()

        
    result_q = starting_q*k0 + ending_q*k1
    
    """
    # If the inputs are too close for comfort, linearly interpolate
    if (cosa > 0.9995): 
        k0 = 1.0 - t
        k1 = t
    else: 
        sina = np.sqrt( 1.0 - cosa*cosa )
        a = np.arctan2( sina, cosa )
        k0 = np.sin((1.0 - t)*a)  / sina
        k1 = np.sin(t*a) / sina
        
    result_q = starting_q*k0 + ending_q*k1
    """
    return result_q

def interpolate_pose(pose0,pose1,ts):  
    print(pose0.shape)
    q0=axisangle_quat(torch.from_numpy(pose0)) #24,3->24,4
    q1=axisangle_quat(torch.from_numpy(pose1)) 
    results=[]
    print("d,coia,zie,")
    print(q0)
    print(q1.size())
    print("done")
    
    for t in ts:
        res=slerp(q0,q1,t)
        results.append(res)
    
    print("interpolated",torch.stack(results).shape)


    return torch.stack(results) #n,24,4
    

if __name__ == '__main__':
    """
    #unit test
    p=torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]]).view(-1,4)
    q=torch.tensor([[0.707,0,0,0.707],[0.707,0,0,0.707]]).view(-1,4)
    
    r=slerp(p,q,0.333)
    print(r)
    rr=slerp(p,q,0.667)
    print(rr)
    """
    
    #global test
    #device = torch.device('cuda', 1)
    device=torch.device("cpu")
    smpl  = getSMPL()
    #smpl = SMPL(os.path.join(os.path.dirname(__file__),'model\\neutral_smpl_with_cocoplus_reg.txt'), obj_saveable = True).to(device)
    pose0= np.load("tpose.npy").reshape(-1,3)
    pose1= np.array([
            1.22162998e+00,   1.17162502e+00,   1.16706634e+00,
            -1.20581151e-03,   8.60930011e-02,   4.45963144e-02,
            -1.52801601e-02,  -1.16911056e-02,  -6.02894090e-03,
            1.62427306e-01,   4.26302850e-02,  -1.55304456e-02,
            2.58729942e-02,  -2.15941742e-01,  -6.59851432e-02,
            7.79098943e-02,   1.96353287e-01,   6.44420758e-02,
            -5.43042570e-02,  -3.45508829e-02,   1.13200583e-02,
            -5.60734887e-04,   3.21716577e-01,  -2.18840033e-01,
            -7.61821344e-02,  -3.64610642e-01,   2.97633410e-01,
            9.65453908e-02,  -5.54007106e-03,   2.83410680e-02,
            -9.57194716e-02,   9.02515948e-02,   3.31488043e-01,
            -1.18847653e-01,   2.96623230e-01,  -4.76809204e-01,
            -1.53382001e-02,   1.72342166e-01,  -1.44332021e-01,
            -8.10869411e-02,   4.68325168e-02,   1.42248288e-01,
            -4.60898802e-02,  -4.05981280e-02,   5.28727695e-02,
            3.20133418e-02,  -5.23784310e-02,   2.41559884e-03,
            -3.08033824e-01,   2.31431410e-01,   1.62540793e-01,
            6.28208935e-01,  -1.94355965e-01,   7.23800480e-01,
            -6.49612308e-01,  -4.07179184e-02,  -1.46422181e-02,
            4.51475441e-01,   1.59122205e+00,   2.70355493e-01,
            2.04248756e-01,  -6.33800551e-02,  -5.50178960e-02,
            -1.00920045e+00,   2.39532292e-01,   3.62904727e-01,
            -3.38783532e-01,   9.40650925e-02,  -8.44506770e-02,
            3.55101633e-03,  -2.68924050e-02,   4.93676625e-02],dtype = np.float).reshape(-1,3)
    
    pose0[0]=pose1[0]
    
    ts=[0.25,0.5,0.75]
    pose=interpolate_pose(pose0,pose1,ts)
        
    
    beta = np.array([-0.25349993,  0.25009069,  0.21440795,  0.78280628,  0.08625954,
            0.28128183,  0.06626327, -0.26495767,  0.09009246,  0.06537955 ])
     
    beta=np.tile(beta,(3,1))
    print(beta.shape)
    
    vbeta = torch.tensor(beta).float().to(device)
    #vpose = torch.tensor(np.array([pose])).float().to(device)
    vpose = pose.float().to(device)
    
    print("hello")
    verts, j, r = smpl(vbeta, vpose, get_skin = True)
    for i,t in enumerate(ts): 
        print("save!")
        smpl.save_obj(verts[i].cpu().numpy(), './mesh_{}.obj'.format(t))
    
    
