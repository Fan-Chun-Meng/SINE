import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

class GraphConvolutionalModule(nn.Module):
    def __init__(self, graph_input_size, graph_hidden_size, graph_output_size):
        super(GraphConvolutionalModule, self).__init__()
        self.graph_conv1_1 = TransformerConv(graph_input_size, graph_hidden_size, edge_dim=1)
        self.graph_conv2_1 = TransformerConv(graph_hidden_size, graph_output_size, edge_dim=1)


        # 第二个图卷积层
        self.graph_conv1_2 = TransformerConv(graph_input_size, graph_hidden_size, edge_dim=1)
        self.graph_conv2_2 = TransformerConv(graph_hidden_size, graph_output_size, edge_dim=1)

        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, data, adj, edge_time, edge_similar):
        #data = data.squeeze(1)
        time_Data_list = []
        for i in range(data.shape[0]):
            time_Data_list.append(Data(x=(data[i,:,:]).squeeze(0), edge_index=adj[i,:,:].squeeze(0), edge_attr=edge_time[i,:,:].squeeze(0)))
        time_Batch = Batch.from_data_list(time_Data_list)
        similar_Data_list = []
        for i in range(data.shape[0]):
            similar_Data_list.append(Data(x=data[i, :, :].squeeze(0), edge_index=adj[i,:,:].squeeze(0), edge_attr=edge_similar[i,:,:].squeeze(0)))
        similar_Batch = Batch.from_data_list(similar_Data_list)

        # 第一个图卷积神经网络
        x1 = F.relu(self.graph_conv1_1(time_Batch.x, time_Batch.edge_index, edge_attr = time_Batch.edge_attr))
        x1 = (self.graph_conv2_1(x1, time_Batch.edge_index, edge_attr = time_Batch.edge_attr))

        x2 = F.relu(self.graph_conv1_2(similar_Batch.x, similar_Batch.edge_index, edge_attr = similar_Batch.edge_attr))
        x2 = (self.graph_conv2_2(x2, similar_Batch.edge_index, edge_attr = similar_Batch.edge_attr))
        x1 = global_mean_pool(x1, batch=time_Batch.batch)
        x2 = global_mean_pool(x2, batch=similar_Batch.batch)
        x = torch.stack([x1, x2], dim=1)
        x = self.conv(x)
        return x