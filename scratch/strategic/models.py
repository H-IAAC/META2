from torch import nn

class BaseConvolutionalModel(nn.Module):

    def __init__(self, height, width, output_classes):
        super(BaseConvolutionalModel, self).__init__()
        self.conv = nn.Sequential(
            self.make_conv_block(1, 8),
            #-2
            self.make_conv_block(8, 16),
            #-4
            self.make_conv_block(16, 32),
            #-6
            self.make_conv_block(32, 64),
            #-8
            nn.Flatten(),
            nn.Linear(in_features = 64*(height-8)*(width-8), out_features = 256),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features = 256, out_features = output_classes)
        )

    def make_conv_block(self, input_channels, output_channels, kernel_size=3):
        return nn.Sequential(
          nn.Conv2d(input_channels, output_channels, kernel_size),
          nn.BatchNorm2d(output_channels),
          nn.LeakyReLU(0.2)
      )

    def forward(self, x):

        return self.conv(x)