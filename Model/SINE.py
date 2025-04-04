import sys
import torch
import torch.nn as nn
from Model.CapsuleNet import CapsuleNet
from Model.ResNet import ResCNN
from Utils.utils import hyperpm


def forward_pass(i, output_segments, adj, edge_time, graph_module):
    """
    Perform graph convolution operation
    :param i: Time step index
    :param output_segments: Input data
    :param adj: Adjacency matrix
    :param edge_time: Edge time information
    :param graph_module: Graph convolution module
    :return: Result of graph convolution
    """
    return graph_module(output_segments[i], adj, edge_time)


class SINE(nn.Module):
    def __init__(self, args):
        """
        Initialize the SINE network
        :param args: Network configuration parameters
        """
        super(SINE, self).__init__()

        # Network configuration parameters
        self.channel_input = args.channel_input
        self.channel_output = args.channel_output
        self.data_input_size = args.data_input_size
        self.conv_split_size = args.conv_split_size
        self.graph_input_size = args.graph_input_size
        self.graph_output_size = args.graph_output_size
        self.graph_hidden_size = args.graph_hidden_size
        self.num_heads = args.num_heads
        self.num_station = args.num_stations
        self.nhidden = args.nhidden
        self.time_steps = args.time_steps
        self.time = args.times

        # Convolutional layers
        self.conv_module = ResCNN(num_classes=100, in_channels=1)
        self.conv_module_loc = ResCNN(num_classes=2, in_channels=2, out_channels=1)
        self.conv_module_dep = ResCNN(num_classes=1, in_channels=2, out_channels=1)
        self.conv_module_mag = ResCNN(num_classes=1, in_channels=2, out_channels=1)

        # Graph convolution modules
        self.graph_modules_ = nn.ModuleList(
            [CapsuleNet(args.nfeat, args.nclass, hyperpm, args.ncaps, args.nhidden, graph_type=args.graph_type).to(
                args.device) for _ in range(args.data_input_size // args.conv_split_size)]
        )
        self.graph_modules = CapsuleNet(args.nfeat, args.nclass, hyperpm, args.ncaps, args.nhidden,
                                        graph_type=args.graph_type).to(args.device)

        # Feature projection layer
        self.feature_projection = nn.Linear(500, 1200)

        # Multi-head self-attention layers
        self.MLP_loc = nn.Linear(args.d_model, 2)
        self.MLP_dep = nn.Linear(args.d_model, 1)
        self.MLP_mag = nn.Linear(args.d_model, 1)

        # LSTM layers
        self.lstm_loc = nn.LSTM(input_size=args.nhidden * 4, hidden_size=int(args.d_model / 2), num_layers=2,
                                batch_first=True, bidirectional=True)
        self.lstm_dep = nn.LSTM(input_size=args.nhidden * 4, hidden_size=int(args.d_model / 2), num_layers=2,
                                batch_first=True, bidirectional=True)
        self.lstm_mag = nn.LSTM(input_size=args.nhidden * 4, hidden_size=int(args.d_model / 2), num_layers=2,
                                batch_first=True, bidirectional=True)

        # Encoder layers
        self.loc_encoder = nn.Linear(args.num_stations * 2, args.d_model)
        self.dep_encoder = nn.Linear(args.num_stations * 1, args.d_model)
        self.mag_encoder = nn.Linear(args.num_stations * 1, args.d_model)

        # Learnable parameters
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.5))

        # Layer normalization
        self.LN_loc = nn.LayerNorm(args.d_model)
        self.LN_dep = nn.LayerNorm(args.d_model)
        self.LN_mag = nn.LayerNorm(args.d_model)

    def forward(self, x, adj, edge_time, station_list, pick_prob):
        """
        Forward pass of the network
        :param x: Input data
        :param adj: Adjacency matrix
        :param edge_time: Edge time
        :param station_list: Station information
        :param pick_prob: Picking probability
        :return: Outputs: Latitude, Longitude, Depth, Magnitude
        """
        # Split the input data into time steps
        split_data = torch.chunk(x, self.time_steps, dim=2)
        pick_prob = torch.stack(torch.chunk(pick_prob, self.time_steps, dim=2), dim=0)
        pick_prob = pick_prob.reshape(pick_prob.shape[0], len(x), self.num_station, pick_prob.shape[3])
        stacked_data = torch.stack(split_data, dim=0)

        adj_repeated = adj.repeat(int(self.data_input_size / self.conv_split_size), 1, 1)
        adj_repeated = adj_repeated.reshape((int(self.data_input_size / self.conv_split_size), len(x),
                                             self.num_station, adj_repeated.shape[2]))
        edge_time_repeated = edge_time.repeat(int(self.data_input_size / self.conv_split_size), 1, 1)
        edge_time_repeated = edge_time_repeated.reshape(
            (int(self.data_input_size / self.conv_split_size), len(x), self.num_station, edge_time_repeated.shape[2]))

        output_segments_list = []
        for time_num in range(self.time):
            if stacked_data.shape[0] ==1:
                wave_data = stacked_data[time_num, :, :, :].squeeze(0)
                p_time_data = pick_prob[time_num, :, :, :].squeeze(0)
                adj_data = adj_repeated[time_num, :, :, :].squeeze(0)
                edge_time_data = edge_time_repeated[time_num, :, :, :].squeeze(0)
            else:
                wave_data = stacked_data[time_num, :, :, :]
                p_time_data = pick_prob[time_num, :, :, :]
                adj_data = adj_repeated[time_num, :, :, :]
                edge_time_data = edge_time_repeated[time_num, :, :, :]

            reshaped_data = wave_data.reshape(
                (int(len(x) * self.num_station), 1, self.conv_split_size))
            output_segments = reshaped_data * p_time_data.reshape(
                (int(len(x) * self.num_station), 1, self.conv_split_size))
            output_segments = self.conv_module(output_segments)
            output_segments = output_segments.reshape(
                (len(x), self.num_station, self.conv_split_size))
            #output_segments = self.feature_projection(output_segments.reshape(len(output_segments), -1))
            output_segments, h_pred_feat = self.graph_modules(output_segments, adj_data, edge_time_data)
            output_segments_list.append(output_segments)

        output_segments = torch.stack(output_segments_list, axis=1).squeeze(2)

        # Process through LSTM layers
        output_loc, _ = self.lstm_loc(output_segments)
        output_dep, _ = self.lstm_dep(output_segments)
        output_mag, _ = self.lstm_mag(output_segments)

        output_loc = torch.mean(output_loc, dim=1)
        output_dep = torch.mean(output_dep, dim=1)
        output_mag = torch.mean(output_mag, dim=1)

        # Encoder processing
        loc_encoder = self.loc_encoder(station_list[:, :, :2].reshape(len(station_list), -1))
        dep_encoder = self.dep_encoder(station_list[:, :, 2:3].reshape(len(station_list), -1))
        mag_encoder = self.mag_encoder(station_list[:, :, 3:].reshape(len(station_list), -1))

        # Combine LSTM output with encoder output
        output_loc = torch.stack([output_loc, loc_encoder], dim=1)
        output_mag = torch.stack([output_mag, mag_encoder], dim=1)
        output_dep = torch.stack([output_dep, dep_encoder], dim=1)

        # Output layers
        loc_output = self.conv_module_loc(output_loc, False, True)
        lat_output = loc_output[0].squeeze(2)
        lon_output = loc_output[1].squeeze(2)
        dep_output = self.conv_module_dep(output_dep, False).squeeze(2)
        mag_output = self.conv_module_mag(output_mag, False).squeeze(2)

        return lat_output, lon_output, dep_output, mag_output
