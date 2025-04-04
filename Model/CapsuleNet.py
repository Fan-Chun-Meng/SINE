import time
from collections import Counter

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as fn
from memory_profiler import profile
from torch_geometric.nn import TransformerConv, global_mean_pool, global_add_pool, GINConv
from torch_geometric.data import Data, Batch
from Model.RoutingLayer import RoutingLayer
from Model.SparseInputLinear import SparseInputLinear
from Utils.utils import scatter_add, dense_to_sparse


class CapsuleNet(nn.Module):
    def __init__(self, nfeat, nclass, hyperpm, ncaps, nhidden, graph_type="knn"):
        super(CapsuleNet, self).__init__()
        ncaps, rep_dim = ncaps, nhidden * ncaps
        self.pca = SparseInputLinear(nfeat, rep_dim)
        conv_ls = []
        for i in range(hyperpm.nlayer):
            conv = RoutingLayer(rep_dim, ncaps)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        self.mlp = nn.Linear(rep_dim, nclass)
        self.dropout = hyperpm.dropout
        self.routit = hyperpm.routit
        self.ncaps = ncaps
        self.rep_dim = rep_dim
        self.nhidden = nhidden
        self.latent_nnb_k = hyperpm.latent_nnb_k
        self.graph_type = graph_type
        graphs = []
        for i in range(ncaps):
            # graph = TransformerConv(nhidden, nhidden, edge_dim=1)
            graph = TransformerConv(nhidden, nhidden)
            #graph = GINConv(nn.Linear(nhidden,nhidden))
            self.add_module('graph_conv1_%d' % i, graph)
            graphs.append(graph)
        self.graphs = graphs
        self.conv = nn.Conv1d(in_channels=ncaps, out_channels=1, kernel_size=1)

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, input_, nb, edge_time_ori):
        # input:(64,5,100) nb(adj):(64,5,4) edge_time: dict:20
        # nb：[
        #     [1, 2, 3, 4],
        #     [0, 2, 3, 4],
        #     [0, 1, 3, 4],
        #     [0, 1, 2, 4],
        #     [0, 1, 2, 3]
        # ]

        cur_adj_list = []
        edge_time_list = []
        x_list = []
        nb = nb.view(-1) #

        input_ = self.pca(input_)
        input_ = fn.leaky_relu(input_)

        for idx_l, conv in enumerate(self.conv_ls):

            input_ = conv(input_, nb, self.routit)
            x = input_.view(input_.shape[0], input_.shape[1], self.ncaps, self.nhidden) # x:(64,5,4,50)

            if idx_l == len(self.conv_ls) - 1:

                hidden_disen_x = x.detach().clone() # (64,5,4,50)

                # result = []
                x_list = []
                edge_time_list = []
                cur_adj_list = []
                for idx_f in range(self.ncaps):

                    cur_X = hidden_disen_x[:, :, idx_f, :]
                    if self.graph_type == "cknn":
                        cur_adj = self.cknn_graph(X=cur_X, k=self.latent_nnb_k)
                    if self.graph_type == "knn":
                        cur_adj = self.knn_graph(input_=cur_X, k=self.latent_nnb_k)


                    non_zero_indices_ = torch.nonzero(cur_adj == 1.0)
                    edge_list_ = non_zero_indices_[non_zero_indices_[:, 1] != non_zero_indices_[:, 2]]
                    first_element_counts = Counter(edge_list_[:, 0].tolist())
                    counts_list = list(first_element_counts.items())
                    # for i in range(len(counts_list) - 1):
                    #     if counts_list[i][0] + 1 != counts_list[i+1][0]:
                    #         missing_value = counts_list[i][0] + 1
                    #         counts_list.insert(i+1, (missing_value, 0))
                    second_elements = [t[1] for t in counts_list]
                    edge_time = edge_time_ori[edge_list_[:,0],edge_list_[:,1],edge_list_[:,2]].unsqueeze(1)
                    start_nodes_, end_nodes_ = edge_list_[:, 1], edge_list_[:, 2]
                    edge_matrix = torch.stack([start_nodes_, end_nodes_])
                    # if len(second_elements) < 640:
                    #     print()
                    for i in range(len(second_elements)):
                        cur_adj_list.append(edge_matrix[:, sum(second_elements[:i]):sum(second_elements[:i+1])])
                        edge_time_list.append(edge_time[sum(second_elements[:i]):sum(second_elements[:i + 1]),:])
                    # a = x[:, :, idx_f, :]
                    # b = x[:, :, idx_f, :].squeeze(1)
                    x_list_ = [x[:, :, idx_f, :]]
                    x_list.extend(x_list_)

        cur_adj_list = [cur_adj_list[i:i+len(cur_adj_list) // self.ncaps] for i in range(0, len(cur_adj_list), len(cur_adj_list) // self.ncaps)]
        edge_time_list = [edge_time_list[i:i + len(edge_time_list) // self.ncaps] for i in
                        range(0, len(edge_time_list), len(edge_time_list) // self.ncaps)]

        grouped_list = []

        for num_ncaps in range(self.ncaps):
            # _ = [Data(x=(x_list[num_ncaps][i,:,:]), edge_index=cur_adj_list[num_ncaps][i],
            #          edge_attr=edge_time_list[num_ncaps][i]) for i in range(len(x_list[num_ncaps]))]
            _ = [Data(x=(x_list[num_ncaps][i, :, :]), edge_index=cur_adj_list[num_ncaps][i]) for i in range(len(x_list[num_ncaps]))]
            grouped_list.append(_)


        outs = []

        for i in range(self.ncaps):
            batch_data = Batch.from_data_list(grouped_list[i])
            # out = self.graphs[i](batch_data.x, batch_data.edge_index,
            #                   edge_attr=batch_data.edge_attr)
            out = self.graphs[i](x = batch_data.x, edge_index = batch_data.edge_index)
            outs.append(global_mean_pool(out, batch=batch_data.batch))


        out = torch.stack(outs, dim=1)
        #out = out.reshape(int(out.size(0) / 10), 10, out.size(2))
        # out = self.conv(out)
        out = out.reshape(out.size(0), 1, out.size(1)*out.size(2))
        return out, []

    def normalize_adj(self, mx):
        """Row-normalize matrix: symmetric normalized Laplacian"""
        rowsum = mx.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

    def knn_graph(self, input_, k):

        edge = []
        for i in range(len(input_)): # input_(64,5,50)
            X = input_[i].squeeze(0) #移除维度为 1 的维度=>（5，50）
            assert k < X.shape[0]
            # D = self.pairwise_euclidean_distances(X, X) # 计算节点特征之间的欧氏距离矩阵。(5,5)
            D = torch.cdist(X, X, p=2.0) # 使用 PyTorch提供的内置函数torch.cdist 替代 pairwise_euclidean_distances

            D.fill_diagonal_(0.0) # 将对角线元素填充为 0，因为每个节点到自身的距离应为 0。
            D_low_k, _ = torch.topk(D, k=k, largest=False, dim=-1) # 对每一行，找出最小的 k 个距离。
            D_ml_k, _ = torch.max(D_low_k, dim=-1) # 对每一行，找出最小的 k 个距离中的最大值。
            adj = (D - D_ml_k.unsqueeze(dim=-1) <= 0).float().fill_diagonal_(0.0) # 基于阈值构建邻接矩阵，如果距离小于等于阈值，则两个节点之间有边。
            adj = (adj + adj.T) / 2.0 # 确保邻接矩阵是对称的，因为 k-NN 图应该是无向图。
            adj.fill_diagonal_(1.0) # 将对角线元素填充为 1，确保每个节点与自身相邻。
            edge.append(adj) # 将构建好的邻接矩阵添加到列表 edge 中。
        return torch.stack(edge, dim=0)

    def cknn_graph(self, X, k, delta=1):
        assert k < X.shape[0]
        D = self.pairwise_euclidean_distances(X, X)
        D.fill_diagonal_(0.0)
        D_low_k, _ = torch.topk(D, k=k, largest=False, dim=-1)
        D_low_k = D_low_k[:, -1]
        adj = (D.square() < delta * delta * torch.matmul(D_low_k.view(-1, 1),
                                                         D_low_k.view(1, -1))).float().fill_diagonal_(0.0)
        adj = (adj + adj.T) / 2.0
        adj.fill_diagonal_(1.0)
        return adj

    def pairwise_euclidean_distances(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf).sqrt()

    def gcn_agg(self, adj, X):
        adj_sp = self.adj_process(adj)
        output = torch.sparse.mm(adj_sp, X)
        return output

    def adj_process(self, adj):
        adj_shape = adj.size()
        adj_indices, adj_values = dense_to_sparse(adj)
        adj_values = self.row_normalize(adj_indices, adj_values, adj_shape)
        return torch.sparse.FloatTensor(adj_indices, adj_values, adj_shape)

    def row_col_normalize(self, adj_indices, adj_values, adj_shape):
        row, col = adj_indices
        deg = scatter_add(adj_values, row, dim=0, dim_size=adj_shape[0])
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        adj_values = deg_inv_sqrt[row] * adj_values * deg_inv_sqrt[col]
        return adj_values

    def row_normalize(self, adj_indices, adj_values, adj_shape):
        row, _ = adj_indices
        deg = scatter_add(adj_values, row, dim=0, dim_size=adj_shape[0])
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        adj_values = deg_inv_sqrt[row] * adj_values
        return adj_values

    def col_normalize(self, adj_indices, adj_values, adj_shape):
        _, col = adj_indices
        deg = scatter_add(adj_values, col, dim=0, dim_size=adj_shape[0])
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        adj_values = deg_inv_sqrt[col] * adj_values
        return adj_values
