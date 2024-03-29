from torch import nn
import sys
import torch

class IIC_Loss(nn.Module):
    def __init__(self):
        super(IIC_Loss, self).__init__()

    def __call__(self, x_out, x_tf_out):
        return self.IID_loss(x_out, x_tf_out)
        
    # 定义损失函数
    def IID_loss(self, x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
        # has had softmax applied
        _, k = x_out.size() # 第一个是batch 第二个k是类别数
        p_i_j = self.compute_joint(x_out, x_tf_out)
        assert (p_i_j.size() == (k, k))
    
        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()  # but should be same, symmetric
    
        # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
        p_i_j[(p_i_j < EPS).data] = EPS
        p_j[(p_j < EPS).data] = EPS
        p_i[(p_i < EPS).data] = EPS
    
        loss = - p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))
    
        loss = loss.sum()
    
        loss_no_lamb = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
    
        loss_no_lamb = loss_no_lamb.sum()
    
        return loss, loss_no_lamb
    
    # 计算联合分布矩阵
    def compute_joint(self, x_out, x_tf_out):
        # produces variable that requires grad (since args require grad)
        bn, k = x_out.size()
        assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)
    
        # 矩阵相乘 需要两个size完全相同 对应位置相乘
    
        p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
        p_i_j = p_i_j.sum(dim=0)  # k, k
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise
    
        return p_i_j

# TEST
# x = torch.tensor([[0.1, 0.9], 
#                   [0.1, 0.9]])
# y = torch.tensor([[0.1, 0.9],
#                   [0.1, 0.9]])
# 
# iic_loss = IIC_Loss()
# # com_joint = iic_loss.compute_joint(x, y)
# # print(com_joint)
# 
# loss, loss_no_lamb = iic_loss.IID_loss(x, y)
# print(loss, loss_no_lamb)
