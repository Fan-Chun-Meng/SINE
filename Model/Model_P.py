import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.BiLSTMBlock import BiLSTMBlock
from Model.ResCNN import ResCNNBlock
from Model.Transformer_Block import Transformer, SeqSelfAttention


class Model_P(nn.Module):

    def __init__(self, args):
        super(Model_P, self).__init__()
        self.in_channels = args.channel_input
        self.in_samples = args.data_input_size

        self.drop_rate = 0.1
        self.norm = False

        self.filters = [8,16,16,32,32,64,64,]  # Number of filters for the convolutions
        self.kernel_sizes = [11, 9, 7, 7, 5, 5, 3]  # Kernel sizes for the convolutions
        self.res_cnn_kernels = [3, 3, 3, 3, 2, 3, 2]

        # Encoder stack
        self.encoder = nn.Sequential(
            # spatial feature CNN+ResCNN
            CNN_Block(
                input_channels=self.in_channels,
                filters=self.filters,
                kernel_sizes=self.kernel_sizes,
                in_samples=self.in_samples,
                res_kernel_sizes=self.res_cnn_kernels,
                drop_rate=self.drop_rate,
            ),
            # Temporal feature
            Temporal_Block(
                input_size=self.filters[-1],
                drop_rate=self.drop_rate,
                original_compatible=False,
            )
        )

        self.Decoder_detection = Decoder(
            input_channels=16,
            filters=self.filters[::-1],
            kernel_sizes=self.kernel_sizes[::-1],
            out_samples=self.in_samples,
            original_compatible=False,
            )

        self.Pick_detection = Decoder(
            input_channels=16,
            filters=self.filters[::-1],
            kernel_sizes=self.kernel_sizes[::-1],
            out_samples=self.in_samples,
            original_compatible=False,
        )

        self.dropout = nn.Dropout(self.drop_rate)
        self.pick_lstms = nn.LSTM(16, 16, bidirectional=False)
        self.pick_attentions = SeqSelfAttention(input_size=16, attention_width=3, eps=1e-7)


    def forward(self, x):
        assert x.ndim == 3
        assert x.shape[1:] == (self.in_channels, self.in_samples)

        # Shared encoder part
        x = self.encoder(x)

        # Detection part
        detection = torch.squeeze(self.Decoder_detection(x), dim=1)
        outputs = [detection]

        # Pick part
        px = self.pick_lstms(x.permute(2, 0, 1))[0]
        px = self.dropout(px)
        px, _ = self.pick_attentions(px.permute(1, 2, 0))
        pred = torch.squeeze(self.Pick_detection(px), dim=1)  # Remove channel dimension

        outputs.append(pred)

        return tuple(outputs)



class CNN_Block(nn.Module):

    def __init__(self, input_channels, filters, kernel_sizes, in_samples, res_kernel_sizes, drop_rate):
        super().__init__()

        convs = []
        pools = []
        members = []
        self.paddings = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )

            # To be consistent with the behaviour in tensorflow,
            # padding needs to be added for odd numbers of input_samples
            padding = in_samples % 2

            # Padding for MaxPool1d needs to be handled manually to conform with tf padding
            self.paddings.append(padding)
            pools.append(nn.MaxPool1d(2, padding=0))
            in_samples = (in_samples + padding) // 2

        for ker in res_kernel_sizes:
            members.append(ResCNNBlock(filters[-1], ker, drop_rate))

        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)
        self.res_conv = nn.ModuleList(members)



    def forward(self, x):
        for conv, pool, padding in zip(self.convs, self.pools, self.paddings):
            x = torch.relu(conv(x))
            if padding != 0:
                # Only pad right, use -1e10 as negative infinity
                x = F.pad(x, (0, padding), "constant", -1e10)
            x = pool(x)
        for member in self.res_conv:
            x = member(x)
        return x

class Temporal_Block(nn.Module):
    def __init__(
        self, input_size, drop_rate, hidden_size=16, original_compatible=False):
        super().__init__()

        self.Lstm = nn.Sequential(
            BiLSTMBlock(
            input_size=input_size,
            hidden_size=hidden_size,
            drop_rate=drop_rate,
            original_compatible=original_compatible
            ),
            BiLSTMBlock(
                input_size=hidden_size,
                hidden_size=hidden_size,
                drop_rate=drop_rate,
                original_compatible=original_compatible
            )
        )

        self.trans = nn.Sequential(
            Transformer(
                input_size=16, drop_rate=drop_rate, eps=1e-7
            ),
            Transformer(
                input_size=16, drop_rate=drop_rate, eps=1e-7
            ),
        )

    def forward(self, x):
        x = self.Lstm(x)
        x = self.trans(x)
        return x



class Decoder(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        kernel_sizes,
        out_samples,
        original_compatible=False,
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.original_compatible = original_compatible

        self.n = len(kernel_sizes)

        self.crops = []
        current_samples = out_samples
        for i, _ in enumerate(filters):
            padding = current_samples % 2
            current_samples = (current_samples + padding) // 2
            if padding == 1:
                self.crops.append(len(filters) - 1 - i)

        convs = []
        for in_channels, out_channels, kernel_size in zip([input_channels] + filters[:-1], filters, kernel_sizes):
            convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )
        convs.append(nn.Conv1d(in_channels=filters[-1], out_channels=1, kernel_size=11, padding=5))

        self.convs = nn.ModuleList(convs)



    def forward(self, x):
        for i, conv in enumerate(self.convs):

            if i != len(self.convs) - 1:
                x = self.upsample(x)

            if self.original_compatible:
                if i == 3:
                    x = x[:, :, 1:-1]
            else:
                if i in self.crops:
                    x = x[:, :, :-1]
            if i == len(self.convs) - 1:
                x = F.sigmoid(conv(x))
            else:
                x = F.relu(conv(x))

        return x