import torch
import json
import sys
import numpy as np
#from .util_smpl import batch_global_rigid_transformation, batch_rodrigues, reflect_pose
from util_smpl import batch_global_rigid_transformation, batch_rodrigues, reflect_pose, quat2mat 
from SMPL_interpolation import interpolate_pose
import torch.nn as nn
import os
import trimesh
from util_smpl import axisangle_quat 

class SMPL(nn.Module):
    def __init__(self, model_path, joint_type = 'cocoplus', obj_saveable = False):
        super(SMPL, self).__init__()

        if joint_type not in ['cocoplus', 'lsp']:
            msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp"'.format(joint_type)
            sys.exit(msg)

        self.model_path = model_path
        self.joint_type = joint_type
        with open(model_path, 'r') as reader:
            model = json.load(reader) 
        
        if obj_saveable:
            self.faces = model['f']
        else:
            self.faces = None

        np_v_template = np.array(model['v_template'], dtype = np.float32)
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype = np.float32)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_J_regressor = np.array(model['J_regressor'], dtype = np.float32)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_posedirs = np.array(model['posedirs'], dtype = np.float32)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_joint_regressor = np.array(model['cocoplus_regressor'], dtype = np.float32)
        if joint_type == 'lsp':  #joint regressior 
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor[:, :14]).float())
        else:
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())

        np_weights = np.array(model['weights'], dtype = np.float32)

        vertex_count = np_weights.shape[0] 
        vertex_component = np_weights.shape[1]

        # batch_size = 10
        # np_weights = np.tile(np_weights, (batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))

        self.register_buffer('e3', torch.eye(3).float())
        
        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        if not self.faces:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write( 'v {:f} {:f} {:f}\n'.format( v[0], v[1], v[2]) )

            for f in self.faces: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f {:d} {:d} {:d}\n'.format(f[0] + 1, f[1] + 1, f[2] + 1) )

    #theta_quad should be  24,4
    def forward(self, beta, theta_quad, trans, get_skin = False, theta_in_rodrigues=True):
        device = beta.device
        self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0] #n
        #print(self.shapedirs.size())
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template  # n*6890*3
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor) #n*6890  and  6890*24 J_regressor from shaped template to 24joints
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        #print(v_shaped[:, :, 0].size())
        #print(Jx.size())
        J = torch.stack([Jx, Jy, Jz], dim = 2) #n*24*3
        #print(J.size())
        #print(theta_quad.shape) #theta for now is axis angle (1,72)
        if theta_in_rodrigues:
            #Rs = batch_rodrigues(theta_quad.view(-1, 3)).view(-1, 24, 3, 3) #axis angle-> quaternion->rotate matrix n*24*3*3
            
            #print(theta_quad)
            #input()

            Rs = quat2mat(theta_quad.view(-1,4)).view(-1,24,3,3)
        else: #theta is already rotation matrices
            Rs = theta_quad.view(-1,24,3,3)

        #print("Rs: ",Rs)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207) # n* 207 --23*9
        
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped  # n*6890*3

        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)
        #print(self.parents)
        #print( self.J_transformed-J)
        #print(num_batch)
        W=self.weight.expand(num_batch,*self.weight.shape[1:])  #n*6890*24 should be sparse
        #print("W:",W.shape)
        
        
        #16 from 24 joints in a weighted way 
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)  # T=n*6890*4*4  公式（2）

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2) #n*6890*(3+1)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1)) #insert one dim n*6890*4*1 to v_posed_homo, then it's homo calcul 4*4 and 4*1  

        verts = v_homo[:, :, :3, 0]  #from homo to world n*6890*3

        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)

        if get_skin:
            return verts+trans.view(-1,1,3), joints+trans.view(-1,1,3), Rs
        else:
            return joints

    def deform_clothes_smpl_usingseveralpoints(self, theta, J, v_smpl, v_cloth, neighbors = 3):
        assert len(theta) == 1, 'currently we only support batchsize=1'
        num_batch=1

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl

        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)

            # Calculate closest SMPL N vertices for each vertex of the closest cloth mesh
            correspondance = []
            for i in range(neighbors):
                new_correspondance = torch.argmin(dists, 2)
                dists[0, np.arange(dists.shape[1]), new_correspondance] += 100
                correspondance.append(new_correspondance)
            correspondance = torch.stack(correspondance, -1)

        v_posed_cloth = pose_params[0, correspondance[0]] + v_cloth
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        applying_T = T[0, correspondance].mean(2)
        # Normalizing average of these T:
        norm = torch.sqrt((applying_T[0, :, :3, :3]**2).sum(-1)).unsqueeze(-1)
        applying_T[0, :, :3, :3] /= norm

        v_homo_cloth = torch.matmul(applying_T, torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        return verts_smpl, verts_cloth

    def deform_clothed_smpl_usingseveralpoints(self, theta, J, v_smpl, v_cloth, neighbors = 3):
        assert len(theta) == 1, 'currently we only support batchsize=1'
        num_batch=1

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl

        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)

            # EQUIVALENT TO 'correspondance = torch.argsort(dists, 2)[:, :, :neighbors]' BUT MUCH FASTER:
            correspondance = []
            for i in range(neighbors):
                new_correspondance = torch.argmin(dists, 2)
                dists[0, np.arange(dists.shape[1]), new_correspondance] += 100
                correspondance.append(new_correspondance)
            correspondance = torch.stack(correspondance, -1)

        v_posed_cloth = pose_params[0, correspondance].mean(2) + v_cloth
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        applying_T = T[0, correspondance].mean(2)
        # Normalizing average of this T:
        norm = torch.sqrt((applying_T[0, :, :3, :3]**2).sum(-1)).unsqueeze(-1)
        applying_T[0, :, :3, :3] /= norm

        v_homo_cloth = torch.matmul(applying_T, torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        return verts_smpl, verts_cloth

    def deform_clothed_smpl_usingseveralpoints2(self, theta, J, v_smpl, v_cloth, neighbors = 3):
        assert len(theta) == 1, 'currently we only support batchsize=1'
        num_batch=1

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl

        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)

            # EQUIVALENT TO 'correspondance = torch.argsort(dists, 2)[:, :, :neighbors]' BUT MUCH FASTER:
            correspondance = []
            corresponding_dists = []
            for i in range(neighbors):
                new_correspondance = torch.argmin(dists, 2)
                direct_dists = dists[0, np.arange(dists.shape[1]), new_correspondance[0]]
                dists[0, np.arange(dists.shape[1]), new_correspondance[0]] += 100
                correspondance.append(new_correspondance)
                corresponding_dists.append(direct_dists)
            correspondance = torch.stack(correspondance, -1)
            corresponding_dists = torch.stack(corresponding_dists, -1)

            interpolation = 'linear'
            #interpolation = 'powerfour'
            #interpolation = 'squared'
            if interpolation == 'squared': # NOTE: Consider transforming this from m to mm to avoid numeric issues
                #sum_dists = torch.sqrt(torch.sqrt((corresponding_dists**4).sum(-1).unsqueeze(-1)))
                sum_dists = torch.sqrt((corresponding_dists**2).sum(-1).unsqueeze(-1))
                weights_dists = (sum_dists - corresponding_dists)
                weights_dists = weights_dists/weights_dists.sum(-1).unsqueeze(-1)
                #weights_dists = weights_dists/torch.sqrt((weights_dists**2).sum(-1).unsqueeze(-1))
            elif interpolation == 'powerfour':
                sum_dists = torch.sqrt(torch.sqrt((corresponding_dists**4).sum(-1).unsqueeze(-1)))
                weights_dists = (sum_dists - corresponding_dists)
                weights_dists = weights_dists/weights_dists.sum(-1).unsqueeze(-1)
            elif interpolation == 'linear':
                sum_dists = torch.abs(corresponding_dists).sum(-1).unsqueeze(-1)
                weights_dists = (sum_dists - corresponding_dists)
                weights_dists = weights_dists/weights_dists.sum(-1).unsqueeze(-1)

        v_posed_cloth = (pose_params[0, correspondance]*weights_dists.unsqueeze(0).unsqueeze(-1)).sum(2) + v_cloth
        #v_posed_cloth = pose_params[0, correspondance].mean(2) + v_cloth
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        applying_T = T[0, correspondance[0, :, 0]]
        #applying_T = (T[0, correspondance]*weights_dists.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)).sum(2)
        #applying_T = T[0, correspondance].mean(2)
        # Normalizing average of this T:
        #norm = torch.sqrt((applying_T[0, :, :3, :3]**2).sum(-1)).unsqueeze(-1)
        #applying_T[0, :, :3, :3] /= norm

        v_homo_cloth = torch.matmul(applying_T, torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        return verts_smpl, verts_cloth


    def deform_clothed_smpl(self, theta, J, v_smpl, v_cloth):
        assert len(theta) == 1, 'currently we only support batchsize=1'
        num_batch=1

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl

        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)
            correspondance = torch.argmin(dists, 2)

        v_posed_cloth = pose_params[0, correspondance[0]] + v_cloth
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        v_homo_cloth = torch.matmul(T[0, correspondance[0]], torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        return verts_smpl, verts_cloth
   
    def deform_clothed_smpl_w_normals(self, theta, J, v_smpl, v_cloth, v_normals):
        assert len(theta) == 1, 'currently we only support batchsize=1'
        num_batch=1

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl

        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)
            correspondance = torch.argmin(dists, 2)

        v_posed_cloth = pose_params[0, correspondance[0]] + v_cloth
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        v_homo_cloth = torch.matmul(T[0, correspondance[0]], torch.unsqueeze(v_posed_homo_cloth, -1))

        v_normals_posed = torch.cat([v_normals, torch.ones(num_batch, v_normals.shape[1], 1, device=self.cur_device)], dim=2)
        v_normals_posed = torch.matmul(T[0, correspondance[0]], torch.unsqueeze(v_normals_posed, -1))

        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]
        verts_normals = v_normals_posed[:, :, :3, 0]

        return verts_smpl, verts_cloth, verts_normals
   
    def deform_clothed_smpl_consistent(self, theta, J, v_smpl, v_cloth, normals_cloth, thresh=0.2):
        assert len(theta) == 1, 'currently we only support batchsize=1'
        num_batch=1

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl

        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)
            normals_smpl = trimesh.Trimesh(v_smpl.cpu().data.numpy()[0], self.faces, process=False).vertex_normals
            angle_bw_points = np.dot(normals_cloth, normals_smpl.T)
            #angles = np.arccos(angle_bw_points)

            # Basically removing those that are not appropriate:
            dists = dists[0]
            dists[angle_bw_points < thresh] += 1000
            #dists[np.abs(angle_bw_points) < 0.7] += 1000

            correspondance = torch.argmin(dists, 1).unsqueeze(0)

        v_posed_cloth = pose_params[0, correspondance[0]] + v_cloth
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        v_homo_cloth = torch.matmul(T[0, correspondance[0]], torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        return verts_smpl, verts_cloth

    def normalization_cloth_beta(self, v, beta, v_smpl=None):
        if type(v_smpl) == type(None):
            # JUST GETTING SMPL MODEL ON T-Pose:
            v_smpl = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        else:
            v_smpl = v_smpl.unsqueeze(0)

        with torch.no_grad():
            dists = ((v_smpl - v.unsqueeze(1))**2).sum(-1)
            correspondance = torch.argmin(dists, 1)
            # NOTE: Matmul is very highdimensional. Reduce only getting those that are necessary:
            # NOTE: For dense meshes is actually faster to run the matmul first and get correspnding rows later
            beta_normalization = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1])[0][correspondance]
            #v_normalized = v - beta_normalization
        return beta_normalization.cpu().data.numpy()

    def expand_cloth_beta(self, v, beta, new_beta, v_smpl=None):  
        if type(v_smpl) == type(None):
            # JUST GETTING SMPL MODEL ON T-Pose:
            v_smpl = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        else:
            v_smpl = v_smpl.unsqueeze(0)

        with torch.no_grad():
            dists = ((v_smpl - v.unsqueeze(1))**2).sum(-1)
            correspondance = torch.argmin(dists, 1)
            # NOTE: Matmul is very highdimensional. Reduce only getting those that are necessary:
            # NOTE: For dense meshes it's actually faster to run the matmul first and get correspnding rows later
            beta_normalization = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1])[0][correspondance]
            beta_addition = torch.matmul(new_beta, self.shapedirs).view(-1, self.size[0], self.size[1])[0][correspondance]
            v_normalized = v - beta_normalization + beta_addition
        return v_normalized


    def unpose_and_deform_cloth(self, v_cloth_posed, theta_from, theta_to, beta, Jsmpl, vsmpl, theta_in_rodrigues=True):
        ### UNPOSE:
        device = theta_from.device
        self.cur_device = torch.device(device.type, device.index)
        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if theta_in_rodrigues:
            Rs = batch_rodrigues(theta_from.view(-1, 3)).view(-1, 24, 3, 3)
        else: #theta is already rotations
            Rs = theta_from.view(-1,24,3,3)

        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)  #pose displacement 
        
        
        pose_displ = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed = pose_displ + v_shaped  #SMPL model on theta and beta
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        v_smpl = v_homo[:, :, :3, 0] #after skinnning
        with torch.no_grad():  #establish correspondance 
            dists = ((v_smpl - v_cloth_posed.unsqueeze(1))**2).sum(-1)
            correspondance = torch.argmin(dists, 1)

        invT = torch.inverse(T[0, correspondance])
        v = torch.cat([v_cloth_posed, torch.ones(len(v_cloth_posed), 1, device=self.cur_device)], 1)
        v = torch.matmul(invT, v.unsqueeze(-1))[:, :3, 0]  
        unposed_v = v - pose_displ[0, correspondance] #re move pose deformation
        
        ### REPOSE:
        Rs = batch_rodrigues(theta_to.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_cloth = pose_params[:, correspondance] + unposed_v.unsqueeze(0)
        self.J_transformed, A = batch_global_rigid_transformation(Rs, Jsmpl, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_cloth = torch.matmul(T[0, correspondance], torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_cloth = v_homo_cloth[:, :, :3, 0]
        return verts_cloth[0]

    def unpose_and_deform_cloth_w_normals(self, v_cloth_posed, v_normals, theta_from, theta_to, beta, Jsmpl, vsmpl):
        device = theta_from.device
        self.cur_device = torch.device(device.type, device.index)
        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        Rs = batch_rodrigues(theta_from.view(-1, 3)).view(-1, 24, 3, 3)

        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        pose_displ = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed = pose_displ + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        v_smpl = v_homo[:, :, :3, 0]
        with torch.no_grad():
            dists = ((v_smpl - v_cloth_posed.unsqueeze(1))**2).sum(-1)
            correspondance = torch.argmin(dists, 1)

        invT = torch.inverse(T[0, correspondance])
        v = torch.cat([v_cloth_posed, torch.ones(len(v_cloth_posed), 1, device=self.cur_device)], 1)
        v = torch.matmul(invT, v.unsqueeze(-1))[:, :3, 0]
        unposed_v = v - pose_displ[0, correspondance]
        # Not applying T to normals since model was trained on normals from T-Pose
        
        ### REPOSE:
        Rs = batch_rodrigues(theta_to.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_cloth = pose_params[:, correspondance] + unposed_v.unsqueeze(0)
        self.J_transformed, A = batch_global_rigid_transformation(Rs, Jsmpl, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_cloth = torch.matmul(T[0, correspondance], torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        v_normals_posed = torch.cat([v_normals.unsqueeze(0), torch.ones(num_batch, v_normals.shape[0], 1, device=self.cur_device)], dim=2)
        v_normals_posed = torch.matmul(T[0, correspondance], torch.unsqueeze(v_normals_posed, -1))
        v_normals_posed = v_normals_posed[:, :, :3, 0]

        return verts_cloth[0], v_normals_posed[0]


    def unpose_and_deform_cloth_w_normals2(self, v_cloth_posed, v_normals, theta_from, theta_to, beta, Jsmpl, vsmpl, v_normals_smooth, theta_in_rodrigues=True):
        device = theta_from.device
        self.cur_device = torch.device(device.type, device.index)
        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if theta_in_rodrigues:
            Rs = batch_rodrigues(theta_from.view(-1, 3)).view(-1, 24, 3, 3)
        else: #theta is already rotations
            Rs = theta_from.view(-1,24,3,3)

        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        pose_displ = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed = pose_displ + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        v_smpl = v_homo[:, :, :3, 0]
        angle_thresh = 0.1
        with torch.no_grad():
            dists = ((v_smpl - v_cloth_posed.unsqueeze(1))**2).sum(-1)
            normals_smpl = trimesh.Trimesh(v_smpl.cpu().data.numpy()[0], self.faces, process=False).vertex_normals
            angle_bw_points = np.dot(v_normals_smooth, normals_smpl.T)
            dists[angle_bw_points < angle_thresh] += 1000
            correspondance = torch.argmin(dists, 1)

        invT = torch.inverse(T[0, correspondance])
        v = torch.cat([v_cloth_posed, torch.ones(len(v_cloth_posed), 1, device=self.cur_device)], 1)
        v = torch.matmul(invT, v.unsqueeze(-1))[:, :3, 0]
        unposed_v = v - pose_displ[0, correspondance]
        # Not applying T to normals since model was trained on normals from T-Pose
        
        ### REPOSE:
        Rs = batch_rodrigues(theta_to.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_cloth = pose_params[:, correspondance] + unposed_v.unsqueeze(0)
        self.J_transformed, A = batch_global_rigid_transformation(Rs, Jsmpl, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo_cloth = torch.matmul(T[0, correspondance], torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_cloth = v_homo_cloth[:, :, :3, 0]

        v_normals_posed = torch.cat([v_normals.unsqueeze(0), torch.ones(num_batch, v_normals.shape[0], 1, device=self.cur_device)], dim=2)
        v_normals_posed = torch.matmul(T[0, correspondance], torch.unsqueeze(v_normals_posed, -1))
        v_normals_posed = v_normals_posed[:, :, :3, 0]

        return verts_cloth[0], v_normals_posed[0]


    def unnormalize_cloth_pose(self, v_cloth_posed, theta, beta, theta_in_rodrigues=True):
        device = theta.device
        self.cur_device = torch.device(device.type, device.index)
        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if theta_in_rodrigues:
            Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        else: #theta is already rotations
            Rs = theta.view(-1,24,3,3)

        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        pose_displ = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed = pose_displ + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
        
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        v_smpl = v_homo[:, :, :3, 0]
        with torch.no_grad():
            dists = ((v_smpl - v_cloth_posed.unsqueeze(1))**2).sum(-1)
            correspondance = torch.argmin(dists, 1)

        invT = torch.inverse(T[0, correspondance])
        v = torch.cat([v_cloth_posed, torch.ones(len(v_cloth_posed), 1, device=self.cur_device)], 1)
        v = torch.matmul(invT, v.unsqueeze(-1))[:, :3, 0]
        unposed_v = v - pose_displ[0, correspondance]

        return unposed_v

#    def unnormalize_cloth_beta(self, v, beta, v_smpl=None):
#        if type(v_smpl) == type(None):
#            # JUST GETTING SMPL MODEL ON T-Pose:
#            v_smpl = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
#        else:
#            v_smpl = v_smpl.unsqueeze(0)
#
#        with torch.no_grad():
#            dists = ((v_smpl - v.unsqueeze(1))**2).sum(-1)
#            correspondance = torch.argmin(dists, 1)
#            # NOTE: Matmul is very highdimensional. Reduce only getting those that are necessary:
#            # NOTE: For dense meshes is actually faster to run the matmul first and get correspnding rows later
#            beta_normalization = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1])[0][correspondance]
#            v_normalized = v + beta_normalization
#        return v_normalized

    def skeleton(self,beta,require_body=False):
        num_batch = beta.shape[0]
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if require_body:
            return J, v_shaped
        else:
            return J

def getSMPL():
    return SMPL(os.path.normpath(os.path.join(os.path.dirname(__file__),'model/neutral_smpl_with_cocoplus_reg.txt')), obj_saveable = True)
def getTmpFile():
    return os.path.join(os.path.dirname(__file__),'hello_smpl.obj')

if __name__ == '__main__':
    #device = torch.device('cuda', 1)
    device=torch.device("cpu")
    # smpl = SMPL('/home/jby/pytorch_ext/smpl_pytorch/model/neutral_smpl_with_cocoplus_reg.txt',obj_saveable=True)
    smpl = SMPL(os.path.join(os.path.dirname(__file__),'model/neutral_smpl_with_cocoplus_reg.txt'), obj_saveable = True).to(device)
    pose0= np.load("tpose.npy").reshape(-1,3)

    #f=np.load("poses_000.npz")

    #pose1 = f["thetas"][0]

    pose1= np.array([
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.],dtype = np.float64).reshape(-1,3)


    pose0=pose1.reshape(24,3)

    
    ts=[0.25,0.5,0.75]
    #trans=torch.tensor([0.,0.,0.])
    #pose=interpolate_pose(pose0,pose1,ts)
        
    pose=axisangle_quat(torch.from_numpy(pose0)).unsqueeze(0)

    beta = np.load("betas.npy")[0][:10]
    #print(d)
    #input()

     
    #beta=np.tile(beta,(3,1))
    #print(beta.shape)
    
    beta =  np.expand_dims(beta, axis=0)

    vbeta = torch.tensor(beta).float().to(device)
    #vpose = torch.tensor(np.array([pose])).float().to(device)
    vpose = pose.float().to(device)
    #print("hello")

    #print(vpose.shape)
    #print(vbeta.shape)

    trans=torch.tensor([0.,0.,0.])
    verts, j, r = smpl(vbeta, vpose, trans, get_skin = True)

    #print(verts.device)
    #print(verts.shape)

    smpl.save_obj(verts[0].cpu().numpy(), './teste123{}.obj'.format(0))
    
    """
    for i,t in enumerate(ts): 
        print("save!")
        smpl.save_obj(verts[i].cpu().numpy(), './mesh_{}.obj'.format(t))
    """
"""
    rpose = reflect_pose(pose)
    vpose = torch.tensor(np.array([rpose])).float().to(device)
    print("nice")
    verts, j, r = smpl(vbeta, vpose, get_skin = True)
    print("hello")
    smpl.save_obj(verts[0].cpu().numpy(), './rmesh.obj')
    """
