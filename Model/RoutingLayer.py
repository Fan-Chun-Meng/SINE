import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fn


class RoutingLayer(nn.Module):
    def __init__(self, dim, num_caps):
        super(RoutingLayer, self).__init__()
        assert dim % num_caps == 0 # 这句的意思：如果xxx，程序正常往下运行
        self.d, self.k = dim, num_caps # 输入维度对应pca线性层的输出维度：4*50，输出纬度是聚类特征数4
        self._cache_zero_d = torch.zeros(1, self.d) # (1,200)
        self._cache_zero_k = torch.zeros(1, self.k) # (1,4)

    def forward(self, input_, neighbors, max_iter):
        x = input_.view(input_.shape[0] * input_.shape[1], input_.shape[2])
        dev = x.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        n, m = x.size(0), neighbors.size(0) // x.size(0) # n=64*5=320；m=1280/64/5=4; ？原来256 这里的m没看懂
        d, k, delta_d = self.d, self.k, self.d // self.k # d=200;k=4;delta_d=50就是nhidden

        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d) # 维度不变还是(320,200)
        # (64*5,200)，这行输入x本来就是(64*5,200)，感觉是先拆成了(64*5,4,50)，每行做normalize后又拼成(64*5,200)输出
        # F.normalize dim=2是对第三个纬度，也就是每一行操作

        z = torch.cat([x, self._cache_zero_d], dim=0) # z:(321,200)

        index_tensor = torch.arange(len(neighbors)).to(dev)
        result_tensor = neighbors.to(torch.int64) + (index_tensor.to(torch.int64) // 20) * 5
        z = z[result_tensor]
        z = z.view(n, m, k, delta_d)

        u = None
        for clus_iter in range(max_iter):
            # 初始化概率分布p
            if u is None:

                p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
            else:

                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p = fn.softmax(p, dim=2)


            u = torch.sum(z * p.view(n, m, k, 1), dim=1)
            u += x.view(n, k, delta_d)

            if clus_iter < max_iter - 1:
                u = fn.normalize(u, dim=2)

        return u.view(input_.shape[0], input_.shape[1], d)