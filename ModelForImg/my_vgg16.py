import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, Shape):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=Shape[1], out_channels=64, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.bn6 = nn.BatchNorm1d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.bn7 = nn.BatchNorm1d(256)
        self.relu7 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.bn8 = nn.BatchNorm1d(512)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.bn9 = nn.BatchNorm1d(512)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.bn10 = nn.BatchNorm1d(512)
        self.relu10 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.bn11 = nn.BatchNorm1d(512)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, padding=0)
        # self.bn12 = nn.BatchNorm1d(512)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, padding=0)
        # self.bn13 = nn.BatchNorm1d(512)
        self.relu13 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.relu14 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.relu15 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(in_features=128, out_features=10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        # x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        # x = self.bn6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        # x = self.bn7(x)
        x = self.relu7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        # x = self.bn8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        # x = self.bn9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        # x = self.bn10(x)
        x = self.relu10(x)
        x = self.pool4(x)
        x = self.conv11(x)
        # x = self.bn11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        # x = self.bn12(x)
        x = self.relu12(x)
        x = self.conv13(x)
        # x = self.bn13(x)
        x = self.relu13(x)
        x = self.pool5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu14(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu15(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x