from collections import namedtuple
import torch.nn as nn
from torchvision import models


class vgg19(nn.Module):

    def __init__(self, pre_trained=True, require_grad=False):
        super(vgg19, self).__init__()
        self.vgg_feature = models.vgg19(pretrained=pre_trained).features
        self.seq_list = [nn.Sequential(ele) for ele in self.vgg_feature]
        self.vgg_layer = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                          'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                          'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                          'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                          'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']

        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, x):

        conv1_1 = self.seq_list[0](x)
        relu1_1 = self.seq_list[1](conv1_1)
        conv1_2 = self.seq_list[2](relu1_1)
        relu1_2 = self.seq_list[3](conv1_2)
        pool1 = self.seq_list[4](relu1_2)

        conv2_1 = self.seq_list[5](pool1)
        relu2_1 = self.seq_list[6](conv2_1)
        conv2_2 = self.seq_list[7](relu2_1)
        relu2_2 = self.seq_list[8](conv2_2)
        pool2 = self.seq_list[9](relu2_2)

        conv3_1 = self.seq_list[10](pool2)
        relu3_1 = self.seq_list[11](conv3_1)
        conv3_2 = self.seq_list[12](relu3_1)
        relu3_2 = self.seq_list[13](conv3_2)
        conv3_3 = self.seq_list[14](relu3_2)
        relu3_3 = self.seq_list[15](conv3_3)
        conv3_4 = self.seq_list[16](relu3_3)
        relu3_4 = self.seq_list[17](conv3_4)
        pool3 = self.seq_list[18](relu3_4)

        conv4_1 = self.seq_list[19](pool3)
        relu4_1 = self.seq_list[20](conv4_1)
        conv4_2 = self.seq_list[21](relu4_1)
        relu4_2 = self.seq_list[22](conv4_2)
        conv4_3 = self.seq_list[23](relu4_2)
        relu4_3 = self.seq_list[24](conv4_3)
        conv4_4 = self.seq_list[25](relu4_3)
        relu4_4 = self.seq_list[26](conv4_4)
        pool4 = self.seq_list[27](relu4_4)

        conv5_1 = self.seq_list[28](pool4)
        relu5_1 = self.seq_list[29](conv5_1)
        conv5_2 = self.seq_list[30](relu5_1)
        relu5_2 = self.seq_list[31](conv5_2)
        conv5_3 = self.seq_list[32](relu5_2)
        relu5_3 = self.seq_list[33](conv5_3)
        conv5_4 = self.seq_list[34](relu5_3)
        relu5_4 = self.seq_list[35](conv5_4)
        pool5 = self.seq_list[36](relu5_4)

        vgg_output = namedtuple("vgg_output", self.vgg_layer)

        vgg_list = [conv1_1, relu1_1, conv1_2, relu1_2, pool1,
                    conv2_1, relu2_1, conv2_2, relu2_2, pool2,
                    conv3_1, relu3_1, conv3_2, relu3_2, conv3_3, relu3_3, conv3_4, relu3_4, pool3,
                    conv4_1, relu4_1, conv4_2, relu4_2, conv4_3, relu4_3, conv4_4, relu4_4, pool4,
                    conv5_1, relu5_1, conv5_2, relu5_2, conv5_3, relu5_3, conv5_4, relu5_4, pool5]

        out = vgg_output(*vgg_list)

        return out
