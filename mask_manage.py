import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

import random
import copy
'xiugai'

import torch.nn.functional as F

class PruningNetworkManager:
    def __init__(self, model,output_dir):

        self.output_dir = output_dir
        self.pruning_layers = self.get_pruning_layers(model)
        num_layers = len(self.pruning_layers)
        for pl in self.pruning_layers:
            pl.total_layers = num_layers
        self.layers = self.printgra(model)
        self.all_num = 0.0
        self.grow_num = 0.0
        self.prune_num = 0.0
        self.masks = []
        self.mask_grows = []
       
        self.grads=[]
        self.count=1

        self.lambda_r = 0.2

        self.min_keep_ratio = 0.2

        self.regrow_grad_alpha = 0.8
        self.regrow_spike_alpha = 0.2


        # ===== 新增：用于“自适应重建”的统计量 =====
        self.num_layers = num_layers

        # 每层累计 mask 变化比例（动态剪枝扰动）
        self.layer_change_accum = [0.0 for _ in range(num_layers)]
        self.layer_change_count = [0 for _ in range(num_layers)]

        # 每层累计平均 spike rate
        self.layer_spike_accum = [0.0 for _ in range(num_layers)]
        self.layer_spike_count = [0 for _ in range(num_layers)]
        


    def get_pruning_layers(self, model):
        #model_list=list(model.modules())
        pruning_layers = []
        #print(model)
        for module in model.modules():
        
        #for i in range(len(model_list)):
            #print(i)
            #print(model_list[i])
            if type(module).__name__ == 'PruningLayer':
            #if isinstance(model_list[i], PruningLayer):
                #print(model_list[i])
                #if type(module) is type(PruningLayer):
                #if isinstance(module, PruningLayer):
                print('aaa')
                #print(module)
                pruning_layers.append(module)
        #print(pruning_layers)
        return pruning_layers

    def evaling(self):
        for pruning_layer in self.pruning_layers:
            pruning_layer.seteval()

    def training(self):
        for pruning_layer in self.pruning_layers:
            #print('11111111')
            pruning_layer.settrain()
            #print('22222222222')
    def reset_zeros(self):
        
        for pruning_layer in self.pruning_layers:
            pruning_layer.reset_zero()

    def update_masks(self,model,a,b):

        old=copy.deepcopy(self.masks)
        acts = []
        #print(self.pruning_layers)
        for pruning_layer in self.pruning_layers:
            #print('aaa')
            #print(pruning_layer)
            activation = pruning_layer.get_actt()
            #print(activation)
            acts.append(activation)
        #print(acts)

        num_layers = len(acts)
            
       
        #print(len(acts))
        
        sorted_indices = torch.argsort(torch.cat(acts), descending=True)
        print(sorted_indices.shape)
        num_elements = int(len(sorted_indices) * a)
        num_elements = min(max(num_elements, 0), len(sorted_indices) - 1)
        #print(num_elements)
        threshold_indice = sorted_indices[num_elements]
        print(num_elements)
        print(threshold_indice)
        threshold=torch.cat(acts)[threshold_indice]
        print(len(acts))
        print(threshold)
        i = 0
        #print(len(acts))
        for i in range(len(acts)):
            '''
            if len(self.masks) != num_layers:
                mask=(acts[i] > threshold).float().detach()
                self.masks.append(mask)
            else:
                self.masks[i]=(acts[i] > threshold).float().detach()
            '''
            layer_act = acts[i]
            mask = (layer_act > threshold).float().detach()

            # ===== 每层最小保留比例保护 =====
            min_keep = max(1, int(layer_act.numel() * self.min_keep_ratio))
            cur_keep = int(mask.sum().item())

            if cur_keep < min_keep:
                topk_idx = torch.topk(layer_act, k=min_keep, largest=True).indices
                mask = torch.zeros_like(layer_act)
                mask[topk_idx] = 1.0

            if len(self.masks) != num_layers:
                self.masks.append(mask)
            else:
                self.masks[i] = mask
            i += 1
        i = 0
        prune_num =0

        for i in range(len(self.masks)):
            prune_num += torch.sum(self.masks[i] == 0).item()
            i+=1
        print(prune_num)




        i=0
        channel_gradients_grows=[]


        #regrowth
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                gradients =module.weight.grad

                # gradients_grow=torch.where(self.masks[i] == 0,torch.abs(module.weight.grad),
                #                                              torch.zeros_like(module.weight.grad))
                grad = module.weight.grad


                ################################
                '''
                if grad is None:
                    gradients_grow = torch.zeros_like(module.weight, device=module.weight.device)
                else:
                    # 关键：mask 放到与 grad 相同的 device
                    mask = self.masks[i].to(device=grad.device)
                    #################################
                    gradients_grow = torch.where(mask == 0, grad.abs(), torch.zeros_like(grad))
                '''

                ###############################################
                '''
                if grad is None:
                    gradients_grow = torch.zeros_like(module.weight, device=module.weight.device)
                else:
                    mask = self.masks[i].to(device=grad.device)

                    # 只针对已剪枝通道做 regrowth score
                    pruned_idx = (mask == 0)

                    # 先取纯梯度主项
                    grad_abs = grad.abs()

                    # 取该层 spike rate
                    sr = self.pruning_layers[i].get_spike_rate()

                    if sr is None:
                        # 如果还没积累到 spike rate，就退化为原论文 regrowth
                        gradients_grow = torch.where(pruned_idx, grad_abs, torch.zeros_like(grad_abs))
                    else:
                        sr = sr.to(grad.device).float()

                        # 只对已剪枝通道做层内归一化
                        pruned_sr = sr[pruned_idx]

                        if pruned_sr.numel() == 0:
                            gradients_grow = torch.zeros_like(grad_abs)
                        else:
                            sr_min, sr_max = pruned_sr.min(), pruned_sr.max()
                            if (sr_max - sr_min) > 1e-12:
                                pruned_sr_norm = (pruned_sr - sr_min) / (sr_max - sr_min)
                            else:
                                pruned_sr_norm = torch.zeros_like(pruned_sr)

                            # 构造完整 regrowth score，默认全 0
                            gradients_grow = torch.zeros_like(grad_abs)

                            # 梯度主导，spike rate 仅做加分项
                            gradients_grow[pruned_idx] = grad_abs[pruned_idx] * (1.0 + self.lambda_r * pruned_sr_norm)


                    '''

                #####################################
                if grad is None:
                    gradients_grow = torch.zeros_like(module.weight, device=module.weight.device)
                else:
                    mask = self.masks[i].to(device=grad.device)
                    pruned_idx = (mask == 0)
                    grad_abs = grad.abs()

                    # 默认全 0
                    gradients_grow = torch.zeros_like(grad_abs)

                    # 没有已剪枝通道，直接跳过
                    if pruned_idx.sum().item() == 0:
                        gradients_grow = torch.zeros_like(grad_abs)
                    else:
                        sr = self.pruning_layers[i].get_spike_rate()

                        # 如果没有 spike 统计，就退化为纯梯度 regrowth
                        if sr is None:
                            gradients_grow[pruned_idx] = grad_abs[pruned_idx]
                        else:
                            sr = sr.to(grad.device).float()

                            pruned_grad = grad_abs[pruned_idx]
                            pruned_sr = sr[pruned_idx]

                            # -------- 1) 对 pruned grad 做层内归一化 --------
                            g_min, g_max = pruned_grad.min(), pruned_grad.max()
                            if (g_max - g_min) > 1e-12:
                                pruned_grad_norm = (pruned_grad - g_min) / (g_max - g_min)
                            else:
                                pruned_grad_norm = torch.zeros_like(pruned_grad)

                            # -------- 2) 对 pruned spike rate 做层内归一化 --------
                            s_min, s_max = pruned_sr.min(), pruned_sr.max()
                            if (s_max - s_min) > 1e-12:
                                pruned_sr_norm = (pruned_sr - s_min) / (s_max - s_min)
                            else:
                                pruned_sr_norm = torch.zeros_like(pruned_sr)

                            # -------- 3) 梯度主导 + spike 辅助 --------
                            score = (
                                    self.regrow_grad_alpha * pruned_grad_norm +
                                    self.regrow_spike_alpha * pruned_sr_norm
                            )

                            gradients_grow[pruned_idx] = score


                    #########################################
                    '''
                    sr = self.pruning_layers[i].get_spike_rate()
                    if sr is None:
                        sr = torch.ones_like(grad).detach().cpu()

                    sr = sr.to(grad.device).float()
                    sr = sr / (sr.mean() + 1e-6)
                    sr = sr.clamp(0.0, 2.0)
                    gradients_grow = torch.where(mask == 0, grad.abs() * sr, torch.zeros_like(grad))
                    '''

                channel_gradients_grows.append(gradients_grow)
                i += 1
        sorted_indices_grow = torch.argsort(torch.cat(channel_gradients_grows), descending=True)
        print(f"sorted_indices_grow.shape: {sorted_indices_grow.shape}")
        num_elements_grow = int(len(sorted_indices_grow) * b)
        print(f"num_elements_grow: {num_elements_grow}")
        print(f"num_elements_grow:{num_elements_grow}")
        threshold_indice_grow = sorted_indices_grow[num_elements_grow]
        #print(threshold_indice)
        threshold_grow=torch.cat(channel_gradients_grows)[threshold_indice_grow]
        print(f"threshold_grow:{threshold_grow}")
        for i in range(len(self.masks)):
            self.masks[i][channel_gradients_grows[i] > threshold_grow] = 1
        i = 0
        prune_num =0

        for i in range(len(self.masks)):
            prune_num += torch.sum(self.masks[i] == 0).item()
            i+=1
        print(f"prune_num:{prune_num}")
        
        
        num=0
        new=copy.deepcopy(self.masks)
        if len(old) == num_layers:
            for o,n in zip(old,new):
                numl=torch.logical_xor(o,n)
                numl=torch.sum(numl==1).item()
                num=num+numl
        print(num)
        
        df = pd.DataFrame([num/2])
        print(df)

        save_path = os.path.join(self.output_dir, 'changes.csv')
        df.to_csv(save_path, mode='a', index=False)
        # df.to_csv('changes.csv', mode='a', index=False)

        # ===== 新增：记录每层 mask 变化比例（动态剪枝扰动） =====
        if len(old) == num_layers:
            for layer_idx, (o, n) in enumerate(zip(old, new)):
                changed = torch.logical_xor(o, n).sum().item()
                ratio = changed / float(n.numel())
                self.layer_change_accum[layer_idx] += ratio
                self.layer_change_count[layer_idx] += 1
        else:
            # 第一次没有 old mask，就记 0
            for layer_idx, n in enumerate(new):
                self.layer_change_accum[layer_idx] += 0.0
                self.layer_change_count[layer_idx] += 1

        # ===== 新增：记录每层平均 spike rate =====
        for layer_idx, pruning_layer in enumerate(self.pruning_layers):
            sr = pruning_layer.get_spike_rate()
            if sr is None:
                sr_mean = 0.0
            else:
                sr_mean = float(sr.mean().item())

            self.layer_spike_accum[layer_idx] += sr_mean
            self.layer_spike_count[layer_idx] += 1
        
    
  
        
      
                    
                    


    def do_masks(self,model):

        i = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                prune_indices = (self.masks[i] == 0).nonzero().view(-1)
                #print(module.weight.data.shape)
                mask_l = torch.ones_like(module.weight.data)
                mask_l[prune_indices,:, :, :] = 0
                module.weight.data.mul_(mask_l)
                i += 1
        i = 0
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                prune_indices = (self.masks[i] == 0).nonzero().view(-1)
                mask_l = torch.ones_like(module.weight.data)
                mask_l[prune_indices] = 0
                module.weight.data.mul_(mask_l)
                module.bias.data.mul_(mask_l)
                i += 1

    def compute_prune(self):
        i = 0
        self.prune_num =0

        for i in range(len(self.masks)):
            self.prune_num += torch.sum(self.masks[i] == 0).item()
            i+=1
        j=0
        self.reserve_num=0
        for j in range(len(self.masks)):
            self.grow_num += torch.sum(self.masks[j] == 1).item()
            j += 1
        s=0
        self.all_num=0
        for s in range(len(self.masks)):
            self.all_num +=self.masks[s].numel()
            s += 1



    


    def save_csv(self):
        data = {'Zero_num': [self.prune_num],'reserve_num': [self.reserve_num]}
        df = pd.DataFrame(data)

        save_path = os.path.join(self.output_dir, "num.csv")
        df.to_csv(save_path, mode="a", index=False)
        # df.to_csv('num.csv', mode='a', index=False)
        # print(mask[0])
       
        i = 0

        for i in range(len(self.masks)):
            self.prune_num_l = torch.sum(self.masks[i] == 0).item()         
            l = {'Zero Percentage_l': [self.prune_num_l]}
            print(l)
            df = pd.DataFrame(l)
            # df.to_csv('layerprune.csv', mode='a', index=False)

            save_path = os.path.join(self.output_dir, 'layerprune.csv')
            df.to_csv(save_path, mode='a', index=False)

            i+=1
        
        


    def save_csv_max(self):
        data = {'Zero_num': [self.prune_num],'grow Percentage': [self.grow_num/self.all_num]}
        df = pd.DataFrame(data)

        save_path = os.path.join(self.output_dir, 'num_max.csv')
        df.to_csv(save_path, mode='a', index=False)
        # df.to_csv('num_max.csv', mode='a', index=False)
        # print(mask[0])
        
        
        
        i = 0

        for i in range(len(self.masks)):
            self.prune_num_l = torch.sum(self.masks[i] == 0).item()         
            l = {'Zero Percentage_l': [self.prune_num_l]}
            print(l)
            df = pd.DataFrame(l)

            save_path = os.path.join(self.output_dir, 'layerprune_max.csv')
            df.to_csv(save_path, mode='a', index=False)
            # df.to_csv('layerprune_max.csv', mode='a', index=False)
            i+=1
        mask=[]
        for i in range(len(self.masks)):
            mask.append(self.masks[i].cpu().numpy())
        df=pd.DataFrame(mask)
        save_path = os.path.join(self.output_dir, 'mask.csv')
        df.to_csv(save_path, mode='a', index=False)
        # df.to_csv("mask.csv",mode="a",index=False)
        
        


    def printgra(self, model):
        layers = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
                layers.append(module)
        return layers


    def prints(self):
        masks = []
        for pruning_layer in self.pruning_layers:
            mask = pruning_layer.get_mask()
            # print(mask.shape)
            masks.append(mask)
        i = 0
        '''for layer in self.layers:
            #print(layer)
            weight_gradient = layer.weight
            print(weight_gradient.shape)
            pruned_indices = (masks[i] == 0).nonzero().squeeze(dim=1)
            print(masks[0].shape)
            print(pruned_indices.shape)
            if pruned_indices.nelement() == 0:
                #print('111')
                continue
            else:
                pruned_gradients = weight_gradient[pruned_indices]
            i=i+1
            print(f"Gradients of pruned Conv2d layer weights: {pruned_gradients}")
        '''
        # print(layer)
        print(len(masks))
        print(len(self.layers))
        weight_gradient = self.layers[1].weight
        print(weight_gradient.shape)
        pruned_indices = (masks[0] == 0).nonzero()
        # pruned_indices = np.where(masks[0] == 0)
        print(masks[0].shape)
        print(pruned_indices.shape)
        if pruned_indices.nelement() == 0:
            print('111')

        else:
            pruned_gradients = weight_gradient[pruned_indices]
            print(pruned_gradients.shape)

            print(f"Gradients of pruned Conv2d layer weights: {pruned_gradients[0]}")

    def get_pruned_out_idx_list(self, device=None):
        """
        返回一个 list，与 do_masks() 里 Conv2d 的遍历顺序一致：
        pruned_out_idx_list[i] 是第 i 个 Conv2d 要剪掉的 out-channel 索引（1D LongTensor）
        """
        pruned = []
        for i, mask in enumerate(self.masks):
            idx = (mask == 0).nonzero(as_tuple=False).view(-1)
            if device is not None:
                idx = idx.to(device)
            pruned.append(idx)
        return pruned

    def save_final_mask(self):
        mask = []
        for i in range(len(self.masks)):
            mask.append(self.masks[i].cpu().numpy())
        df = pd.DataFrame(mask)
        save_path = os.path.join(self.output_dir, "maskfinal.csv")
        df.to_csv(save_path, index=False)

    def save_cfg(self):
        cfg_channels = [int(mask.sum().item()) for mask in self.masks]
        cfg_path = os.path.join(self.output_dir, "cfg.txt")
        with open(cfg_path, "w") as f:
            f.write(str(cfg_channels))



    def get_avg_change_ratios(self):
        vals = []
        for s, c in zip(self.layer_change_accum, self.layer_change_count):
            if c == 0:
                vals.append(0.0)
            else:
                vals.append(s / c)
        return vals

    def get_avg_spike_rates(self):
        vals = []
        for s, c in zip(self.layer_spike_accum, self.layer_spike_count):
            if c == 0:
                vals.append(0.0)
            else:
                vals.append(s / c)
        return vals

    def get_depth_priors(self):
        # 越深层值越大
        if self.num_layers <= 1:
            return [1.0]
        vals = []
        for i in range(self.num_layers):
            x = float(i + 1) / float(self.num_layers)
            vals.append(x * x)   # 用平方，稍微强调深层
        return vals



class PruningLayer(nn.Module):
    def __init__(self, layer_id: int,   total_layers = 13):
        super(PruningLayer, self).__init__()
        self.activation_means = None
        self.mask = None
       
        
        self.v_accumulated = None
        self.prune_count = 0 
        self.restore_count = 0
        self.p1 = 0
        self.p2 = 0
        self.a = 0
        self.aa = 0
        
        self.count1 = 1
        self.trainingstate = True
        self.spikes = 0
        self.membrance=0

        # uniqueness 参数
        self.lambda_u = 1.0
        self.compute_uniqueness_every = 8
        self._step = 0

        self.layer_id = int(layer_id)
        self.total_layers = total_layers

        self.snr_eps = 1e-6
        self.activity_mode = "snr"  #复现原论文时需要改成mean

        #uniqueness+gate
        self.uniqueness_last = None
        self.gate_last = None

        #梯度+ spikes
        self.spike_rate_accum = None

        #控制uniquness
        self.alpha_u = 0.4

        self.spike_count = 0

        # EMA 参数
        self.importance_ema_m = 0.95
        self.spike_ema_m = 0.95

        self.spike_count = 0

        self.uni_topk = 4

    def seteval(self):
        self.trainingstate = False

    def settrain(self):
        #print('bbbbbbb')
        self.trainingstate = True

    '''
    def forward(self, x,v):
        #print('aaaaaa')
        if self.trainingstate:
            #print('aaaaaaaaaa')

            if self.aa==0 or self.v_accumulated == None:
                #print('111111111111111111111111111')
                self.spikes = x.detach()
                self.membrance=v.abs().detach()
                
                v_temp = (self.spikes + self.membrance).detach()
                self.v_accumulated=torch.mean(v_temp, dim=(0, 1, 3, 4))

            else:
                self.spikes = x.detach()
                self.membrance = v.abs().detach()
                #print(self.spikes[0][0][0])
                #print(self.membrance[0][0][0])
                v_temp = (self.spikes + self.membrance).detach()
                self.v_accumulated = self.v_accumulated * self.count1+torch.mean(v_temp, dim=(0, 1, 3, 4))
                #self.activation_means = self.decay_rate * self.activation_means + (1 - self.decay_rate) * torch.mean(x,dim=(0,1,3,4)).detach()
                self.count1 += 1
                self.v_accumulated/= self.count1
                # self.activation_means = (self.activation_means * self.count1 + torch.mean(x,dim=(0, 1,3,4))).detach()
                # self.count1 += 1
                #print(self.count1)
                # self.activation_means /= self.count1

        return x  
    '''

    @torch.no_grad()
    def forward(self, x: torch.Tensor, v: torch.Tensor):
        """
        x: spike sequence, shape [T, N, C, H, W]
        v: membrane sequence, shape [T, N, C, H, W]
        """
        if (not self.trainingstate) or (x is None) or (v is None):
            return x

        self._step += 1

        # ===== 1) 构建通道特征 F_k =====
        # 原版 activity 用 spikes + |v|
        spikes = x.detach()
        memb = v.detach().abs()
        z = spikes + memb  # [T, N, C, H, W]

        # activity: per-channel mean over (T,N,H,W)
        # shape: [C]
        activity = z.mean(dim=(0, 1, 3, 4)).float()

        # ===== spike rate accumulation for regrowth =====
        # 当前 batch 的每通道 spike rate
        spike_rate = spikes.mean(dim=(0, 1, 3, 4)).float()  # [C]


        # ===== spike rate accumulation for regrowth =====

        '''
        if self.spike_rate_accum is None or self.aa == 1:
            self.spike_rate_accum = spike_rate.detach().cpu()
            self.spike_count = 1
        else:
            self.spike_rate_accum = self.spike_rate_accum * self.spike_count + spike_rate.detach().cpu()
            self.spike_count += 1
            self.spike_rate_accum = self.spike_rate_accum / self.spike_count
        '''




        #############################################################################
        '''
        if self.activity_mode == "mean":
            importance =  activity
        else:
            mu = z.mean(dim=(0, 1, 3, 4)).float()  # [C]
            var = z.var(dim=(0, 1, 3, 4), unbiased=False).float()  # [C]
            std = (var + self.snr_eps).sqrt()  # [C]
            snr = (mu / std).clamp(min=0.0)  # [C]

            importance = snr
        '''

        ###########################################################
        '''
        # ===== 2) uniqueness: 计算通道间余弦相似度 =====
        # 注意：计算 cos matrix 是 O(C^2)，可以通过 compute_uniqueness_every 降频
        if self.compute_uniqueness_every > 1 and (self._step % self.compute_uniqueness_every != 0):
            # 降频时：只用 activity（或用上一次 uniqueness 的效果隐含在 v_accumulated 中）
            importance = activity
        else:
            # F: [C, M], 其中 M = T*N*H*W
            # 先把 channel 维移到最前，再 flatten 其它维
            Fmat = z.permute(2, 0, 1, 3, 4).contiguous().flatten(1).float()  # [C, M]

            # 为了数值稳定，做 L2 normalize，余弦相似度 = dot product
            Fnorm = F.normalize(Fmat, p=2, dim=1, eps=1e-12)  # [C, M]

            # cos_sim: [C, C]
            cos_sim = Fnorm @ Fnorm.t()

            # 去掉对角线（自己和自己相似度=1）
            Cc = cos_sim.shape[0]
            if Cc <= 1:
                uniqueness = torch.ones_like(activity)
            else:
                sum_all = cos_sim.sum(dim=1)          # 每行求和
                mean_other = (sum_all - 1.0) / (Cc - 1)  # 减去对角线的 1，再平均
                uniqueness = 1.0 - mean_other          # [C]
                # 可选：限制范围，避免极端数值
                uniqueness = uniqueness.clamp(min=0.0, max=2.0)

            # ===== 3) 合成最终重要性分数 =====
            # 推荐形式：activity * (1 + lambda_u * uniqueness)
            # 解释：uniqueness=0 时退化为 activity；uniqueness 越大越加分
            importance = activity * (1.0 + self.lambda_u *(1.0 - self.layer_id / max(1, (self.total_layers - 1)))* uniqueness)
        '''


        ########################################################################

        # ===== 2) uniqueness(z) + spike gating (NO activity) =====

        # ---- gate: 用 spikes 的放电率衡量通道是否“在工作” ----
        # shape: [C]
        spike_rate = spikes.mean(dim=(0, 1, 3, 4)).float()

        spike_rate_cpu = spike_rate.detach().float().cpu()

        # self.spike_rate_accum = spike_rate.detach().cpu()
        if self.spike_rate_accum is None or self.aa == 1:
            self.spike_rate_accum = spike_rate.detach().cpu()
        else:
            '''
            self.spike_rate_accum = self.spike_rate_accum * self.count1 + spike_rate.detach().cpu()
            self.spike_rate_accum = self.spike_rate_accum / (self.count1 + 1)
            '''
            m = self.spike_ema_m
            self.spike_rate_accum = m * self.spike_rate_accum + (1.0 - m) * spike_rate_cpu

        mean_sr = spike_rate.mean()
        # soft gate：相对均值归一化 + clamp，避免尺度漂移和极端值
        gate = (spike_rate / (spike_rate.mean() + 1e-6)).clamp(0.0, 2.0)

        thr = 0.1 * mean_sr
        gate = gate * (spike_rate > thr).float()

        # ---- uniqueness: 用 z = spikes + |v| 的余弦相似度 ----
        need_compute = True
        if self.compute_uniqueness_every > 1:
            # 第一次必须算；之后按频率算
            if (self.uniqueness_last is not None) and (self._step % self.compute_uniqueness_every != 0):
                need_compute = False

        if not need_compute:
            uniqueness = self.uniqueness_last
            gate_use = gate
        else:
            # F: [C, M], M = T*N*H*W
            Fmat = z.permute(2, 0, 1, 3, 4).contiguous().flatten(1).float()  # [C, M]
            Fnorm = F.normalize(Fmat, p=2, dim=1, eps=1e-12)  # [C, M]
            cos_sim = Fnorm @ Fnorm.t()  # [C, C]

            Cc = cos_sim.shape[0]
            if Cc <= 1:
                uniqueness = torch.ones_like(spike_rate)
            else:
                sum_all = cos_sim.sum(dim=1)
                mean_other = (sum_all - 1.0) / (Cc - 1)
                # (1 - sim_z)
                uniqueness = (1.0 - mean_other).clamp(min=0.0, max=2.0)

            # 缓存给降频使用
            self.uniqueness_last = uniqueness.detach()

            gate_use = gate

        # ---- final importance: 只用 uniqueness * gate ----
        importance = (uniqueness * gate_use)

        ################################################

        # 归一化 activity
        # act_min, act_max = activity.min(), activity.max()
        # if (act_max - act_min) > 1e-12:
        #     activity_norm = (activity - act_min) / (act_max - act_min)
        # else:
        #     activity_norm = torch.zeros_like(activity)
        # # ===== 2) uniqueness: 计算通道间余弦相似度 =====
        # # 注意：计算 cos matrix 是 O(C^2)，可以通过 compute_uniqueness_every 降频
        # if self.compute_uniqueness_every > 1 and (self._step % self.compute_uniqueness_every != 0):
        #     # 降频时：只用 activity
        #     importance = activity_norm
        # else:
        #
        #     '''
        #     # F: [C, M], 其中 M = T*N*H*W
        #     # 先把 channel 维移到最前，再 flatten 其它维
        #     Fmat = z.permute(2, 0, 1, 3, 4).contiguous().flatten(1).float()  # [C, M]
        #
        #     # 为了数值稳定，做 L2 normalize，余弦相似度 = dot product
        #     Fnorm = F.normalize(Fmat, p=2, dim=1, eps=1e-12)  # [C, M]
        #
        #     # cos_sim: [C, C]
        #     cos_sim = Fnorm @ Fnorm.t()
        #
        #     # 去掉对角线（自己和自己相似度=1）
        #     Cc = cos_sim.shape[0]
        #     if Cc <= 1:
        #         uniqueness = torch.ones_like(activity)
        #     else:
        #         sum_all = cos_sim.sum(dim=1)  # 每行求和
        #         mean_other = (sum_all - 1.0) / (Cc - 1)  # 减去对角线的 1，再平均
        #         uniqueness = 1.0 - mean_other  # [C]
        #         uniqueness = uniqueness.clamp(min=0.0, max=2.0)
        #     '''
        #
        #     # F: [C, M], 其中 M = T*N*H*W
        #     Fmat = z.permute(2, 0, 1, 3, 4).contiguous().flatten(1).float()  # [C, M]
        #
        #     # 为了数值稳定，做 L2 normalize，余弦相似度 = dot product
        #     Fnorm = F.normalize(Fmat, p=2, dim=1, eps=1e-12)  # [C, M]
        #
        #     # cos_sim: [C, C]
        #     cos_sim = Fnorm @ Fnorm.t()
        #
        #     Cc = cos_sim.shape[0]
        #     if Cc <= 1:
        #         uniqueness = torch.ones_like(activity)
        #     else:
        #         # 去掉对角线，避免自己和自己相似度=1进入 top-k
        #         cos_sim = cos_sim.clone()
        #         cos_sim.fill_diagonal_(-1.0)
        #
        #         k = min(self.uni_topk, Cc - 1)
        #         topk_vals, _ = torch.topk(cos_sim, k=k, dim=1, largest=True)
        #
        #         mean_other = topk_vals.mean(dim=1)
        #         uniqueness = 1.0 - mean_other
        #         uniqueness = uniqueness.clamp(min=0.0, max=1.0)
        #     # ===== 3) 改成退火式混合评分 =====
        #     # importance = alpha * uniqueness + (1 - alpha) * activity
        #     # 建议 alpha 从小到大试：0.2 / 0.4 / 0.6 / 0.8
        #     alpha = self.alpha_u
        #
        #
        #
        #     # 归一化 uniqueness
        #     uni_min, uni_max = uniqueness.min(), uniqueness.max()
        #     if (uni_max - uni_min) > 1e-12:
        #         uniqueness_norm = (uniqueness - uni_min) / (uni_max - uni_min)
        #     else:
        #         uniqueness_norm = torch.zeros_like(uniqueness)
        #
        #     # 最终 importance
        #     importance = alpha * uniqueness_norm + (1.0 - alpha) * activity_norm
        #
        # # ===== 调试：看 importance 和 activity 有多像 =====
        # if self._step % 100 == 0:
        #     corr = torch.corrcoef(torch.stack([
        #         activity_norm.detach(),
        #         importance.detach()
        #     ]))[0, 1]
        #
        #     print(f"[Layer {self.layer_id}] alpha={self.alpha_u:.2f}, corr(activity, importance)={corr.item():.4f}")
        #




        ###############################
        # ===== 4) running average 累计 =====
        '''
        if self.v_accumulated is None or self.aa == 1:
            self.v_accumulated = importance.detach().cpu()
            self.count1 = 1
            self.aa = 0
        else:
            # 原版的逐步平均写法
            self.v_accumulated = self.v_accumulated * self.count1 + importance.detach().cpu()
            self.count1 += 1
            self.v_accumulated = self.v_accumulated / self.count1
        '''

        if self.v_accumulated is None or self.aa == 1:
            self.v_accumulated = importance.detach().cpu()
            self.aa = 0
        else:
            m = self.importance_ema_m
            self.v_accumulated = m * self.v_accumulated + (1.0 - m) * importance.detach().cpu()


        return x

    def get_spike_rate(self):
        return self.spike_rate_accum

    def reset_zero(self):
        self.v_accumulated = None
        self.count1 = 1
        self.aa=1
        self._step = 0
        self.uniqueness_last = None
        self.gate_last = None

        self.spike_rate_accum = None

        self.spike_count = 0
   
    def get_prunenum(self):
        num = torch.sum(self.mask == 0).item()
        return num

    def get_allnum(self):
        num = self.mask.numel()
        return num

    def get_actt(self):
        #print("Calling get_actt")

        return self.v_accumulated

  

    def get_mask(self):
        return self.mask

