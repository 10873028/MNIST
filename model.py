from torch import nn


class MNISTModel(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(MNISTModel, self).__init__()
        self.mConv0 = nn.Conv2d(
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=(4, 4),
            stride=1,
            padding=(3, 3),
        )
        self.mReLU0 = nn.LeakyReLU(0.2)
        self.mConv1 = nn.Conv2d(
            in_channels=outChannels,
            out_channels=outChannels,
            kernel_size=(2, 2),
            stride=1,
            padding=(1, 1),
        )
        self.mReLU1 = nn.LeakyReLU(0.2)
        self.mMaxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.mFlat = nn.Flatten()
        self.mBN = nn.BatchNorm1d(num_features=8192)
        self.mFC = nn.Linear(in_features=8192, out_features=10)
        self.mSoftmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.mConv0(x)
        x = self.mReLU0(x)
        x = self.mConv1(x)
        x = self.mReLU1(x)
        x = self.mMaxPool(x)
        x = self.mFlat(x)
        x = self.mBN(x)
        x = self.mFC(x)
        x = self.mSoftmax(x)
        return x
