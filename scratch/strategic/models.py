import torch
from torch import nn


class BaseConvolutionalModel(nn.Module):

    def __init__(self, height, width, output_classes):
        super(BaseConvolutionalModel, self).__init__()
        self.conv = nn.Sequential(
            self.make_conv_block(1, 8),
            # -2
            self.make_conv_block(8, 16),
            # -4
            self.make_conv_block(16, 32),
            # -6
            self.make_conv_block(32, 64),
            # -8
            nn.Flatten(),
            nn.Linear(in_features=64*(height-8)*(width-8), out_features=256),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=256, out_features=output_classes)
        )

    def make_conv_block(self, input_channels, output_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):

        return self.conv(x)


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_shape, hidden_layer_dimensions, num_classes):
        super(FullyConnectedNetwork, self).__init__()

        flatten_size = 1
        for dim in input_shape:
            flatten_size *= dim

        layers = [nn.Flatten()]
        in_features = flatten_size
        for out_features in hidden_layer_dimensions:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features

        layers.append(nn.Linear(in_features, num_classes))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)
