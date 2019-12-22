import torch.nn as nn

class LinNet(nn.Module):
    """
    A simple MLP with configurable activation functions.
    """

    def __init__(self,
                 n_hid=[200],
                 n_in=784,
                 n_out=10,
                 activation=nn.ReLU,
                 output=lambda: nn.LogSoftmax(-1)):
        super().__init__()

        if type(n_hid) != list:
            n_hid = [n_hid]
        n_layers = [n_in] + n_hid + [n_out]

        self.layers = []
        for i_layer, (n1, n2) in enumerate(zip(n_layers, n_layers[1:])):
            mods = [nn.Linear(n1, n2, bias=False)]
            act_fn = activation if i_layer < len(n_layers) - 2 else output
            if act_fn is not None:
                mods.append(act_fn())
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ConvNet(nn.Module):
    """
    A simple CNN+MLP stacked network, with configurable activation functions.
    """

    def __init__(self,
                 n_ch_conv = [32, 64],
                 kernel_size_conv = [5, 5],
                 conv_args = {'stride':1, 'padding':0, 'bias':False},
                 activation_conv=nn.ReLU,
                 pool_conv=lambda: nn.MaxPool2d(kernel_size=2, stride=2),
                 dropout=False,
                 n_in_fc = 1024,
                 n_hid_fc=[128],
                 n_out=10,
                 activation_fc=nn.ReLU,
                 bias_fc=False,
                 output=lambda: nn.LogSoftmax(-1)):
        super().__init__()

        if type(n_ch_conv) != list:
            n_ch_conv = [n_ch_conv]
        n_ch_conv = [1] + n_ch_conv

        if type(activation_conv) != list:
            activation_conv = [activation_conv]*len(n_ch_conv)

        self.layers_conv = []
        for n_ch_in, n_ch_out, k_size, act_fn in zip(n_ch_conv, n_ch_conv[1:], kernel_size_conv, activation_conv):
            mods = [nn.Conv2d(n_ch_in, n_ch_out, k_size, **conv_args)]
            if act_fn is not None:
                mods.append(act_fn())
            if pool_conv is not None:
                mods.append(pool_conv())
            layer = nn.Sequential(*mods)
            self.layers_conv.append(layer)

        self.layers_conv = nn.ModuleList(self.layers_conv)

        if type(n_hid_fc) != list:
            n_hid_fc = [n_hid_fc]
        n_layers = [n_in_fc] + n_hid_fc + [n_out]

        if callable(dropout):
            self.dropout = dropout()
        else:
            if dropout:
                self.dropout = nn.Dropout()
            else:
                self.dropout = None

        self.layers_fc = []
        for i_layer, (n_in, n_out) in enumerate(zip(n_layers, n_layers[1:])):
            mods = [nn.Linear(n_in, n_out, bias=bias_fc)]
            act_fn = activation_fc if i_layer < len(n_layers) - 2 else output
            if act_fn is not None:
                mods.append(act_fn())
            layer = nn.Sequential(*mods)
            self.layers_fc.append(layer)

        self.layers_fc = nn.ModuleList(self.layers_fc)

        self.printed_size = False

    def forward(self, x):

        for layer in self.layers_conv:
            x = layer(x)
        x = x.reshape(x.size(0), -1)

        if self.dropout is not None:
            x = self.dropout(x)

        if not self.printed_size:
            print("Size of input to first linear layer is", x.shape)
            self.printed_size = True

        for layer in self.layers_fc:
            x = layer(x)

        return x