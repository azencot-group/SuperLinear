# Description: This file contains the implementation of the following methods:
# 1. GradManipulation - the base class for all the methods
# 2. DynamicWeightAvg - dynamic/static weight averaging (DWA)
# 3. CovWeighting - covariance weighting (CW) https://arxiv.org/abs/2009.01717
# 4. UncertaintyWeighting - uncertainty weighting (UW) https://arxiv.org/abs/1705.07115
# 5. PCGrad - projected gradient descent (PCGrad) https://arxiv.org/abs/2001.06782
# 6. GradNorm - gradient normalization (GradNorm) https://arxiv.org/abs/1711.02257


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import time




class GradManipulation:
    def __init__(self, optimizer, args, tasks = None, model = None):
        self.args = args
        self.optimizer = optimizer
        self.create_graph = False
        self.k_tasks = len(tasks)


        
       # if self.k_tasks is None:
       #     self.k_tasks = len(self.args.data_list.split("_"))
        print("K tasks", self.k_tasks)

        self.grad_info = {"tasks": tasks}
        print(tasks)
        

        # used for grad inference
        self.conflict_table = np.zeros((self.k_tasks,self.k_tasks))
        self.grad_info['conflict_table'] = []

        self.task_weight = torch.ones(self.k_tasks)
        self.grad_info['task_weight'] = []


        # get relevant param names


        self.layer_names = []

        # Iterate through the optimizer's parameter groups and print parameter names
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                found = False
                for name, tensor in model.state_dict().items():
                    if torch.equal(param.data, tensor.data):
                        self.layer_names.append(name)
                        found = True
                        break
                if not found:
                    print('No match found')
        print("layer_names")
        print(self.layer_names)

                        
              #  print(param_name)
              #  if param_name:
              #      self.layer_names[param] = param_name





        self.num_layers = len(self.layer_names)




        self.layerwise_conflict_table =  { layer_name: np.zeros((self.k_tasks,self.k_tasks)) for layer_name in self.layer_names}
        self.grad_info['layerwise_conflict_table'] = []

        self.ignore_grad_ind = []

        # ignore the gradient of the last layer for each head of multi-head
     #   if args.multi_head:
     #       self.ignore_grad_ind = np.arange(self.k_tasks*2,0,-1)*(-1)
        self.grad_activation(activate = False)

        self.mag_table = []
        self.grad_info['mag'] = []

        self.mag_sim_table = []
        self.grad_info['mag_sim'] = []
        self.mag_sim_table_manip = []

    def step(self):
        self.optimizer.step()

    def backward(self):
        pass

    def stage_update(self):
        _ = self.get_conflict_table()
        if self.args.layerwise:
            _ = self.get_layerwise_conflict_table()
        _ = self.get_mag_table()
        if self.args.mag_sim:
            _ = self.get_mag_sim_table()


    def get_conflict_table(self, reset = True):
        c_t = torch.from_numpy(self.conflict_table)
        if reset:
            self.conflict_table = np.zeros((self.k_tasks,self.k_tasks))
        self.grad_info['conflict_table'].append(c_t)
        return c_t

    def get_layerwise_conflict_table(self, reset = True):
        c_t = copy.deepcopy(self.layerwise_conflict_table)
        for layer_name in c_t:
            c_t[layer_name] = torch.from_numpy(c_t[layer_name])
        if reset: 
            self.layerwise_conflict_table =  { layer_name: np.zeros((self.k_tasks,self.k_tasks)) for layer_name in self.layer_names}
        self.grad_info['layerwise_conflict_table'].append(c_t)
        # remove elements with the term "bias" in the last 4 characters
    #    for layer_name in list(c_t):
    #        if layer_name[-4:] == "bias":
    #            c_t.pop(layer_name)
        return c_t
    
    def get_mag_table(self, reset = True):
        
      #  print(len(self.mag_table))
      #  print(self.mag_table)

        m_t = torch.stack(self.mag_table)
        m_t_mean = torch.mean(m_t, dim=0)
        m_t_std = torch.std(m_t, dim=0)
        if reset:

            self.mag_table = []
        self.grad_info['mag'].append(m_t_mean)
        return m_t_mean, m_t_std
        

    def get_mag_sim_table(self, reset = True):
        m_t = torch.stack(self.mag_sim_table)
        m_t_mean = torch.mean(m_t, dim=0)
        m_t_std = torch.std(m_t, dim=0)
        if len(self.mag_sim_table_manip)>0:
            m_t_manip = torch.stack(self.mag_sim_table_manip)
            m_t_manip_mean = torch.mean(m_t_manip, dim=0)
            m_t_manip_std = torch.std(m_t_manip, dim=0)

        else:
            m_t_manip_mean = None
            m_t_manip_std = None
        if reset:
            self.mag_sim_table = []
            self.mag_sim_table_manip = []
        self.grad_info['mag_sim'].append(m_t_mean)
        return m_t_mean, m_t_std, m_t_manip_mean, m_t_manip_std


    def grad_activation(self, activate = True):
        if len(self.ignore_grad_ind)==0:
            return
        for group in self.optimizer.param_groups:
            param_group = group['params']
            remove_ind = np.arange(0,len(param_group))[self.ignore_grad_ind]
            for i in range(len(param_group)):
                p = param_group[i]
                if i in remove_ind:
                    p.requires_grad = activate


    def vectorize_grad(self):
        grads = {}
        layer_cnt = 0

        for group in self.optimizer.param_groups:
            for p in group['params']:
                layer_name = list(self.layer_names)[layer_cnt]

               # if (p.grad is None) or (self.grad_manip_layers!="" and p not in self.grad_manip_layers):
                if p.grad is not None:  
                    grads[layer_name] = p.grad.clone().flatten()
                else:
                    grads[layer_name] = None
                layer_cnt += 1


                

        grad_vec  = torch.cat([g for g in grads.values() if g is not None])
        if self.args.layerwise:
            return grad_vec, grads
        return grad_vec, None

    def get_grads(self, objectives):
        grad_vecs = []
        grad_layers = []
        for obj in objectives:
            if obj is not None:
                self.optimizer.zero_grad(set_to_none=True)
                obj.backward(retain_graph=True, create_graph=self.create_graph)
                #self.optimizer.zero_grad()
                #obj.backward()
                grad_vec, grad_layer = self.vectorize_grad()
            else:
                grad_vec = None
                grad_layer = {}
                for layer_name in self.layer_names:
                    grad_layer[layer_name] = None
            
            grad_vecs.append(grad_vec)
            grad_layers.append(grad_layer)
        return grad_vecs, grad_layers


    def set_grads(self, grad_vec, objectives):
        # if there's layers that we ignored their gradients (e.g., the last layer of each head in multi-head)
        if len(self.ignore_grad_ind)>0:
            self.grad_activation(activate = True)
            main_loss = 0
            main_objective = torch.mean(objectives)
            self.optimizer.zero_grad(set_to_none=True)
            main_objective.backward(retain_graph=True)

        # set the gradients
        for group in self.optimizer.param_groups:
            param_group = group['params']
            remove_ind = np.arange(0,len(param_group))[self.ignore_grad_ind]
            for i in range(len(param_group)): 
                if i in remove_ind or param_group[i].grad is None:
                    continue
                p = param_group[i]
                length = np.prod(p.grad.shape)
                length = int(length)  #bug fix for value 1.0 which is a float when a tensor is a signel value              
                p.grad = grad_vec[0:length].reshape(p.grad.shape)
                grad_vec = grad_vec[length:]

    def set_lambda_scale(self, lambda_scale):
        self.task_weight = lambda_scale

    def collect_mag(self, grad_vecs):
        mag = torch.zeros(len(grad_vecs))
        for i in range(len(grad_vecs)):
            if grad_vecs[i] is not None:
                mag[i] = torch.norm(grad_vecs[i],p=2)
        return mag

    def collect_mag_similarities(self, grad_vecs):
        num_tasks = len(grad_vecs)
        mag_similarity = torch.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(num_tasks):
                if grad_vecs[i] is None or grad_vecs[j] is None:
                    continue

                mag_ij = 2*(torch.norm(grad_vecs[i],p=2)*torch.norm(grad_vecs[j],p=2))/(torch.norm(grad_vecs[i],p=2)**2+torch.norm(grad_vecs[j],p=2)**2)
                mag_similarity[i, j] = mag_ij
        return mag_similarity

    def collect_conflicts(self, grad_vecs):
        num_tasks = len(grad_vecs)
        conflict_table = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(num_tasks):

                if grad_vecs[i] is None or grad_vecs[j] is None:
                    continue
                grad_i = grad_vecs[i]
                grad_j = grad_vecs[j]
                conflict = torch.dot(grad_i, grad_j) < 0
                conflict_table[i, j] += 1 if conflict else 0
        return conflict_table

    def collect_layerwise_conflicts(self, grad_layers):
    #    print(grad_layers)
        num_tasks = len(grad_layers)
        num_tasks = self.k_tasks
        #num_layers = len(self.optimizer.param_groups[0]['params'])

       # layer_names = [p[0] for p in self.optimizer.param_groups[0]['params']]
        
        #layerwise_conflict_table = np.zeros((num_layers,num_tasks, num_tasks))
        layerwise_conflict_table = { layer_name: np.zeros((num_tasks, num_tasks)) for layer_name in self.layer_names}

      #  print("grad_layers")
      #  print(len(grad_layers))


        for i in range(self.k_tasks):
            for j in range(self.k_tasks):
                for layer_name in self.layer_names:
                    # check if "bias" is in the last 4 characters of the layer name
                    if layer_name[-4:] == "bias":
                        continue

                    grad_i = grad_layers[i][layer_name]
                    grad_j = grad_layers[j][layer_name]


                    # check if the layer has a bias
                    if layer_name[-6:]+"bias"  in self.layer_names:
                        grad_i_b = grad_layers[i][layer_name[:-6]+"bias"]
                        grad_j_b = grad_layers[j][layer_name[:-6]+"bias"]

                        grad_i = torch.cat([grad_i, grad_i_b])
                        grad_j = torch.cat([grad_j, grad_j_b])

                    if grad_i is None or grad_j is None or grad_i.shape == torch.Size([]):
                        continue
                    
     




                    conflict = torch.dot(grad_i, grad_j) < 0
                    layerwise_conflict_table[layer_name][i, j] += 1 if conflict else 0
    
        return layerwise_conflict_table



    def grad_inference(self, objectives):
       # print("grad inference")
       # print(objectives)
        grad_vecs, grad_layers = self.get_grads(objectives)

        conficts = self.collect_conflicts(grad_vecs)
        self.conflict_table += conficts

        mag = self.collect_mag(grad_vecs)
        self.mag_table.append(mag)


        if self.args.mag_sim:
            mag_sim = self.collect_mag_similarities(grad_vecs)
            self.mag_sim_table.append(mag_sim) 

        if self.args.layerwise:
            layerwise_conflicts =  self.collect_layerwise_conflicts(grad_layers)
            for layer_name in layerwise_conflicts:
                self.layerwise_conflict_table[layer_name] += layerwise_conflicts[layer_name]


        """
        pcgrad_vecs = copy.deepcopy(grad_vecs)
        pcgrad_vecs = torch.vstack(pcgrad_vecs)

        grads_inds = np.arange(len(grad_vecs))
        #random.shuffle(grads_inds)
        # vectorized version
        


        # compute the magnitude similarity before the update
        mag_similarity = torch.zeros( (len(grad_vecs), len(grad_vecs)) )
        mag = torch.zeros(len(grad_vecs))
        for i in range(len(pcgrad_vecs)):
            mag[i] = torch.norm(pcgrad_vecs[i],p=2)

            for j in range(len(pcgrad_vecs)):
                mag_ij = 2*(torch.norm(pcgrad_vecs[i],p=2)*torch.norm(pcgrad_vecs[j],p=2))/(torch.norm(pcgrad_vecs[i],p=2)**2+torch.norm(pcgrad_vecs[j],p=2)**2)
                mag_similarity[i,j] = mag_ij
        self.mag_table.append(mag)
        self.mag_sim_table.append(mag_similarity)

        # compute the conflict table with cosine similarity
        i = 0 
        for g_i in pcgrad_vecs:
            for j in grads_inds:
                g_j = pcgrad_vecs[j]
                g_i_g_j = torch.dot(g_i, g_j)
                # if there's a conflict between the gradients
                if g_i_g_j < 0:
                    self.conflict_table[i,j] += 1
            i = i+1

        """





class DynamicWeightAvg:
    def __init__(self, args):
        self.plot_task_loss = args.plot_task_loss
        self.mo_method = args.mo_method
        self.t_factor = args.t_factor
        self.norm_type = args.norm_type
        self.temp_v = args.temp_v
        self.temp_h = args.temp_h
        self.norm_mask = args.norm_mask
        self.norm_mask_p = args.norm_mask_p

        # for idw
        self.outlier_penalty = args.outlier_penalty
        self.temp_penalty = args.temp_penalty
        self.loss_v_list = []
        self.loss_h_list = []
        if self.plot_task_loss:
            self.task_coeff_list = []

        if args.mask_loss=='MAE':
            self.mask_mae = True
        else:
            self.mask_mae = False
    def standard_score(self,x, mean, std):
        return (x - mean) / std
    
    def outlier_score(self,z):
        p_weight = 1/(self.temp_penalty*torch.exp(z-1))
        return torch.where(z>1,p_weight,torch.tensor(1.))


    def get_weights(self, objectives_v = None, objectives_h = None):
        if 'dwa' in self.mo_method:
            return self.dwa(objectives_v, objectives_h)
        elif 'idw' in self.mo_method:
            return self.idw(objectives_v, objectives_h)
        
    def idw(self, objectives_v = None, objectives_h = None):
        if objectives_v is not None:
            objectives_v = objectives_v.flatten().detach()
            self.loss_v_list.append(objectives_v)
            self.loss_v_list = self.loss_v_list[-self.t_factor:]
            loss_mean_v = torch.stack(self.loss_v_list).mean(dim=0)
            if self.mask_mae:
                # if element in loss_mean_v is zero set to 1 to avoid division by zero.
                # this does not affect the result since the corresponding loss will be zero in any case
                d_loss_mean_v = torch.where(loss_mean_v==0,torch.tensor(1.),loss_mean_v)
                #self.loss_w_v = 1/d_loss_mean_v.max()
                self.loss_w_v = (1/d_loss_mean_v)**self.norm_mask_p

            else:
                w_v = loss_mean_v.max()/loss_mean_v
                self.loss_w_v = w_v/torch.sum(w_v)

            if self.outlier_penalty:
                loss_std_v = torch.stack(self.loss_v_list).std(dim=0)
                z_v = self.standard_score(objectives_v, loss_mean_v, loss_std_v)
                self.loss_w_v = self.loss_w_v * self.outlier_score(z_v)


        if objectives_h is not None:
            objectives_h = objectives_h.flatten().detach()
            self.loss_h_list.append(objectives_h)
            self.loss_h_list = self.loss_h_list[-self.t_factor:]
            loss_mean_h = torch.stack(self.loss_h_list).mean(dim=0)
            if self.mask_mae:
                # if element in loss_mean_h is zero set to 1 to avoid division by zero.
                # this does not affect the result since the corresponding loss will be zero in any case
                d_loss_mean_h = torch.where(loss_mean_h==0,torch.tensor(1.),loss_mean_h)
               # self.loss_w_h = 1/d_loss_mean_h.max()
                self.loss_w_h = (1/d_loss_mean_h)**self.norm_mask_p
                if self.norm_mask:
                    self.loss_w_h = self.loss_w_h/torch.sum(self.loss_w_h)
            else:
                w_h = loss_mean_h.max()/loss_mean_h
                self.loss_w_h = w_h/torch.sum(w_h)
            if self.outlier_penalty:
                loss_std_h = torch.stack(self.loss_h_list).std(dim=0)
                z_h = self.standard_score(objectives_h, loss_mean_h, loss_std_h)
                self.loss_w_h = self.loss_w_h * self.outlier_score(z_h)



        if objectives_v is not None and objectives_h is not None:
            self.loss_w = self.loss_w_h.view(-1,1)  @ self.loss_w_v.view(1,-1) 
        elif objectives_v is not None:
            self.loss_w = self.loss_w_v
        elif objectives_h is not None:
            self.loss_w = self.loss_w_h

        if self.plot_task_loss:
            self.task_coeff_list.append(self.loss_w.detach().cpu().numpy())
        
        return self.loss_w  

    def dwa(self, objectives_v = None, objectives_h = None):
        if objectives_v is not None:
            objectives_v = objectives_v.flatten().detach()
            if len(self.loss_v_list)>0:
                w_v = objectives_v/torch.stack(self.loss_v_list).mean(dim=0)
                if self.norm_type=='exp':
                    w_v = torch.exp(w_v/self.temp_v)
            else:
                w_v = torch.ones(len(objectives_v))
            self.loss_w_v =  w_v/torch.sum(w_v)

            
            self.loss_v_list.append(objectives_v)
            self.loss_v_list = self.loss_v_list[-self.t_factor:]


        if objectives_h is not None:
            objectives_h = objectives_h.flatten().detach()
            if len(self.loss_h_list)>0:
                w_h = objectives_h/torch.stack(self.loss_h_list).mean(dim=0)
                w_h = torch.exp(w_h/self.temp_h)
                if self.norm_type=='exp':
                    w_h = torch.exp(w_h/self.temp_h)
            else:
                w_h = torch.ones(len(objectives_h))
            self.loss_w_h =  w_h/torch.sum(w_h)

            self.loss_h_list.append(objectives_h)
            self.loss_h_list = self.loss_h_list[-self.t_factor:]


        if objectives_v is not None and objectives_h is not None:
            self.loss_w = self.loss_w_h.view(-1,1)  @ self.loss_w_v.view(1,-1) 
        elif objectives_v is not None:
            self.loss_w = self.loss_w_v
        elif objectives_h is not None:
            self.loss_w = self.loss_w_h

        if self.plot_task_loss:
            self.task_coeff_list.append(self.loss_w.detach().cpu().numpy())
        
        return self.loss_w  

        

class CovWeighting(GradManipulation):
    def __init__(self, optimizer, args, k_tasks = None):
        super().__init__(optimizer, args, k_tasks)
        self.plot_task_loss = args.plot_task_loss
        self.t_factor = args.t_factor
        self.remove_std = args.remove_std
        self.cur_factor = 1
        self.loss_ratio = torch.zeros(self.k_tasks)
        self.loss_mean = torch.ones(self.k_tasks)
        self.loss_std = torch.ones(self.k_tasks)
        self.loss_ratio_mean = torch.ones(self.k_tasks)
        self.loss_ratio_coeff = torch.ones(self.k_tasks)*1/self.k_tasks
        self.loss_ratio_std = torch.ones(self.k_tasks)
        self.loss_ratio_var = torch.ones(self.k_tasks)
        if self.plot_task_loss:
            self.task_coeff_list = []

    def backward(self, objectives):
        objectives = objectives.flatten()
 
        self.loss_ratio = (objectives/self.loss_mean.to(objectives.device)).detach()
        self.loss_mean = self.loss_mean.to(objectives.device)*(1-1/self.cur_factor) + objectives*(1/self.cur_factor)

        self.loss_ratio_mean_new = self.loss_ratio_mean.to(objectives.device)*(1-1/self.cur_factor) + self.loss_ratio*(1/self.cur_factor)

        if self.cur_factor > 1:
            self.loss_ratio_var = self.loss_ratio_var.to(objectives.device)*(1-1/self.cur_factor) + (self.loss_ratio-self.loss_ratio_mean.to(objectives.device))*(self.loss_ratio-self.loss_ratio_mean_new)*(1/self.cur_factor)
        self.loss_ratio_std = torch.sqrt(self.loss_ratio_var).to(objectives.device)
        self.loss_ratio_mean = self.loss_ratio_mean_new.to(objectives.device)
        if self.remove_std:
            self.loss_ratio_coeff = self.loss_ratio_std/self.loss_ratio_mean
        else:
            self.loss_ratio_coeff = 1/self.loss_ratio_mean
        normalizer = torch.sum(self.loss_ratio_coeff)
        self.loss_ratio_coeff = self.loss_ratio_coeff/normalizer
        if self.plot_task_loss:
            self.task_coeff_list.append(self.loss_ratio_coeff.detach().cpu().numpy())
        loss = torch.sum(objectives*self.loss_ratio_coeff.detach())
        self.optimizer.zero_grad( )
        loss.backward()
        self.cur_factor = np.minimum([self.cur_factor+1], [self.t_factor])[0]


class UncertaintyWeighting(GradManipulation):
    def __init__(self, optimizer, args, k_tasks = None):
        super().__init__(optimizer, args, k_tasks)
        self.plot_task_loss = args.plot_task_loss
        if self.plot_task_loss:
            self.task_coeff_list = []


class PCGrad(GradManipulation):
    def __init__(self, optimizer, args, k_tasks = None):
        super().__init__(optimizer, args, k_tasks)
        self.mo_method  = args.mo_method
        self.vmap = args.vmap
        self.chunk_size = None if args.chunk_size==-1 else args.chunk_size # for vmaps

    def backward(self, objectives):
        grad_vecs = self.get_grads(objectives)
        pcgrad_vecs = copy.deepcopy(grad_vecs)
        pcgrad_vecs = torch.vstack(pcgrad_vecs)

        grads_inds = np.arange(len(grad_vecs))
        random.shuffle(grads_inds)
        # vectorized version

        

        # compute the magnitude similarity before the update
        if self.args.grad_inference:
            mag_similarity = torch.zeros( (len(grad_vecs), len(grad_vecs)) )
            for i in range(len(pcgrad_vecs)):
                for j in range(len(pcgrad_vecs)):
                    mag_ij = 2*(torch.norm(pcgrad_vecs[i],p=2)*torch.norm(pcgrad_vecs[j],p=2))/(torch.norm(pcgrad_vecs[i],p=2)**2+torch.norm(pcgrad_vecs[j],p=2)**2)
                    #mag_ij = torch.norm(pcgrad_vecs[i],p=2)/torch.norm(pcgrad_vecs[j],p=2) 
                    mag_similarity[i,j] = mag_ij
            self.mag_sim_table.append(mag_similarity)
        
        
        if self.vmap:
            def update(g_i):
                for j in grads_inds:
                    g_j = grad_vecs[j]
                    g_i_g_j = torch.dot(g_i, g_j)
                    # if there's a conflict between the gradients 
                    g_i -= torch.where(g_i_g_j < 0, ((g_i_g_j)/(g_j.norm()**2))*g_j , 0)
                return g_i
            pcgrad_vecs = torch.vmap(update, chunk_size=self.chunk_size)(pcgrad_vecs)

        # inference version, similar but slower
        else:
            i = 0 
            for g_i in pcgrad_vecs:
                for j in grads_inds:
                   # g_j = grad_vecs[j]
                    g_j = pcgrad_vecs[j]
                    g_i_g_j = torch.dot(g_i, g_j)
                    # if there's a conflict between the gradients
                    if g_i_g_j < 0:
                        if "nonstationary" not  in self.mo_method:
                            self.conflict_table[i,j] += 1
                        proj = (g_i_g_j) / (g_j.norm()**2)
                        g_i -= proj * g_j
                i = i+1
        
        
        # compute the magnitude similarity after the update
        mag_similarity = torch.zeros( (len(grad_vecs), len(grad_vecs)) )
        for i in range(len(pcgrad_vecs)):
            for j in range(len(pcgrad_vecs)):
                mag_ij = 2*(torch.norm(pcgrad_vecs[i],p=2)*torch.norm(pcgrad_vecs[j],p=2))/(torch.norm(pcgrad_vecs[i],p=2)**2+torch.norm(pcgrad_vecs[j],p=2)**2)
               # mag_ij = torch.norm(pcgrad_vecs[i],p=2)/torch.norm(pcgrad_vecs[j],p=2) 
                mag_similarity[i,j] = mag_ij
        self.mag_sim_table_manip.append(mag_similarity)
        final_grad_vec = torch.mean(pcgrad_vecs, dim=0)
       # final_grad_vec = torch.matmul(self.task_weight.to(pcgrad_vecs.device),pcgrad_vecs)
        self.set_grads(final_grad_vec, objectives)




class GradNorm(GradManipulation):
    def __init__(self, optimizer, args, k_tasks = None):  
        super().__init__(optimizer, args, k_tasks)
        self.k_tasks = k_tasks  
        self.alpha = args.gradnorm_alpha
        self.create_graph = True
        self.weights = torch.nn.Parameter(torch.ones(self.k_tasks))
        self.T = self.weights.sum().detach() # sum of weights
        self.lr2 = args.gradnorm_lr
        self.optimizer2 = torch.optim.Adam([self.weights], lr=self.lr2)
        self.l0 = None
        self.plot_task_loss =  args.plot_task_loss
        if self.plot_task_loss:
            self.task_coeff_list = []

    def backward(self, objectives):
        if self.l0 is None:
            self.l0 = objectives.clone().detach()

        self.weights = self.weights.to(objectives.device)
        weighted_loss = self.weights @ objectives
        self.optimizer.zero_grad()
        weighted_loss.backward(retain_graph=True)

        objectives = objectives * self.weights
        grad_vecs = self.get_grads(objectives)
      #  gradnorm_vecs = copy.deepcopy(grad_vecs)
        gradnorm_vecs = grad_vecs
        gradnorm_vecs = torch.vstack(gradnorm_vecs)

        grads_inds = np.arange(len(grad_vecs))


        gw = torch.norm(gradnorm_vecs, dim=1)
        loss_ratio = objectives.detach() / self.l0

        rt = loss_ratio / loss_ratio.mean()
        gw_avg = gw.mean().detach()
        constant = (gw_avg * rt ** self.alpha).detach()
        # check if gw has gradient
        gradnorm_loss = torch.abs(gw - constant).sum()
        self.optimizer2.zero_grad()
        gradnorm_loss.backward()

        self.optimizer2.step()
        self.step()
        if self.plot_task_loss:
            self.task_coeff_list.append(self.weights.clone().detach().cpu().numpy())

        self.weights = (self.weights / self.weights.sum() * self.T).detach()
        self.weights = torch.nn.Parameter(self.weights)
        self.optimizer2 = torch.optim.Adam([self.weights], lr=self.lr2)



