import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


class SiameseNetworkResnet(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        The ResNet-18 model is the feature extractor.
    """
    def __init__(self):
        super(SiameseNetworkResnet, self).__init__()
        # self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

        # over-write the first conv layer to be able to read grayscales images
        # (3,x,x) where 3 is RGB channels -> (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()
        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output

class SiameseNetworkNext(nn.Module):
    """
        The ResNext-101 model is the feature extractor.
        TODO: Currently not working : loss behaviour.
    """
    def __init__(self):
        super(SiameseNetworkNext, self).__init__()
        self.resnet = torchvision.models.resnext101_64x4d(weights=None)

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output

class SiameseNetworkEffnet(nn.Module):
    """
        The Efficient-net large model is the feature extractor.
    """
    def __init__(self):
        super(SiameseNetworkEffnet, self).__init__()
        self.effnet = torchvision.models.efficientnet_v2_l(weights=None)

        self.effnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.fc_in_features = self.effnet.classifier[1].in_features

        self.effnet = torch.nn.Sequential(*(list(self.effnet.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.effnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.effnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output

class SiameseNetworkMobnet(nn.Module):
    """
        The MobileNet-large model is the feature extractor.
        TODO: currently loss is decreasing during training, but accuracy on test set stays at 50% for long.
    """
    def __init__(self):
        super(SiameseNetworkMobnet, self).__init__()

        # self.mobnet = torchvision.models.mobilenet_v3_large(weights=None)
        self.mobnet = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)

        self.mobnet.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.fc_in_features = self.mobnet.classifier[0].in_features
        # self.fc_out_features = self.mobnet.classifier[3].out_features

        self.mobnet = torch.nn.Sequential(*(list(self.mobnet.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.mobnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.mobnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output

class SiameseNetworkVGGnet(nn.Module):
    """
        The MobileNet-large model is the feature extractor.
        TODO: currently loss is decreasing during training, but accuracy on test set stays at 50% for long.
    """
    def __init__(self):
        super(SiameseNetworkVGGnet, self).__init__()

        self.vggnet = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)

        self.vggnet.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc_in_features = self.vggnet.classifier[0].in_features
        # self.fc_out_features = self.vggnet.classifier[3].out_features

        self.vggnet = torch.nn.Sequential(*(list(self.vggnet.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.vggnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.vggnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output

class SiameseNetworkVit(nn.Module):
    """
        The VisionTransformer-large model is the feature extractor.
        Slightly different modifications.
        TODO: FIX learning rate has to be much lower. Hyperparam search. USE pretrained
    """
    def __init__(self):
        super(SiameseNetworkVit, self).__init__()

        # self.vit = torchvision.models.vit_l_32(torchvision.models.ViT_L_32_Weights.DEFAULT)
        self.vit = torchvision.models.vit_l_32(weights=None)

        self.vit.conv_proj = nn.Conv2d(1, 1024, kernel_size=(32, 32), stride=(32, 32))
        self.fc_in_features = self.vit.heads.head.in_features
        self.fc_out_features = self.vit.heads.head.out_features

        # self.vit = torch.nn.Sequential(*(list(self.vit.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_out_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.vit.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.vit(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output


### Test models

class SiameseNetworkResnetTest(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        The ResNet-18 model is the feature extractor.
    """
    def __init__(self):
        super(SiameseNetworkResnetTest, self).__init__()
        # self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)

        # over-write the first conv layer to be able to read grayscales images
        # (3,x,x) where 3 is RGB channels -> (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        self.sigmoid = nn.Sigmoid()
        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # def forward_once(self, x):
    #     output = self.resnet(x)
    #     output = output.view(output.size()[0], -1)
    #     return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.resnet(input1)
        output1 = output1.view(output1.size()[0], -1)
        output2 = self.resnet(input2)
        output2 = output2.view(output1.size()[0], -1)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output

class SiameseNetworkNextTest(nn.Module):
    """
        The ResNext-101 model is the feature extractor.
    """
    def __init__(self):
        super(SiameseNetworkNextTest, self).__init__()
        self.resnext = torchvision.models.resnext101_64x4d(weights=torchvision.models.ResNeXt101_64X4D_Weights.DEFAULT)

        self.resnext.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnext.fc.in_features

        self.resnext = torch.nn.Sequential(*(list(self.resnext.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.resnext.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.resnext(input1)
        output1 = output1.view(output1.size()[0], -1)
        output2 = self.resnext(input2)
        output2 = output2.view(output1.size()[0], -1)


        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)

        return output
    
class SiameseNetworkEffnetTest(nn.Module):
    """
        The Efficient-net large model is the feature extractor.
    """
    def __init__(self):
        super(SiameseNetworkEffnetTest, self).__init__()
        self.effnet = torchvision.models.efficientnet_v2_l(weights=torchvision.models.EfficientNet_V2_L_Weights.DEFAULT)

        self.effnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.fc_in_features = self.effnet.classifier[1].in_features

        self.effnet = torch.nn.Sequential(*(list(self.effnet.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.effnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input1, input2):
        output1 = self.effnet(input1)
        output1 = output1.view(output1.size()[0], -1)
        output2 = self.effnet(input2)
        output2 = output2.view(output1.size()[0], -1)


        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output
    
class SiameseNetworkMobnetTest(nn.Module):
    """
        The MobileNet-large model is the feature extractor.
        TODO: currently loss is decreasing during training, but accuracy on test set stays at 50% for long.
    """
    def __init__(self):
        super(SiameseNetworkMobnetTest, self).__init__()

        # self.mobnet = torchvision.models.mobilenet_v3_large(weights=None)
        self.mobnet = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)

        self.mobnet.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.fc_in_features = self.mobnet.classifier[0].in_features
        # self.fc_out_features = self.mobnet.classifier[3].out_features

        self.mobnet = torch.nn.Sequential(*(list(self.mobnet.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.mobnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input1, input2):
        output1 = self.mobnet(input1)
        output1 = output1.view(output1.size()[0], -1)
        output2 = self.mobnet(input2)
        output2 = output2.view(output1.size()[0], -1)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output
    
class SiameseNetworkVGGnetTest(nn.Module):
    """
        The MobileNet-large model is the feature extractor.
        TODO: currently loss is decreasing during training, but accuracy on test set stays at 50% for long.
    """
    def __init__(self):
        super(SiameseNetworkVGGnetTest, self).__init__()

        self.vggnet = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)

        self.vggnet.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc_in_features = self.vggnet.classifier[0].in_features
        # self.fc_out_features = self.vggnet.classifier[3].out_features

        self.vggnet = torch.nn.Sequential(*(list(self.vggnet.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.vggnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input1, input2):
        output1 = self.vggnet(input1)
        output1 = output1.view(output1.size()[0], -1)
        output2 = self.vggnet(input2)
        output2 = output2.view(output1.size()[0], -1)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output
    
class SiameseNetworkVitTest(nn.Module):
    """
        The VisionTransformer-large model is the feature extractor.
        Slightly different modifications.
        TODO: FIX learning rate has to be much lower. Hyperparam search. USE pretrained
    """
    def __init__(self):
        super(SiameseNetworkVitTest, self).__init__()

        # self.vit = torchvision.models.vit_l_32(torchvision.models.ViT_L_32_Weights.DEFAULT)
        self.vit = torchvision.models.vit_l_32(weights=torchvision.models.ViT_L_32_Weights.DEFAULT)

        # self.vit.conv_proj = nn.Conv2d(1, 1024, kernel_size=(32, 32), stride=(32, 32))
        self.fc_in_features = self.vit.heads.head.in_features
        self.fc_out_features = self.vit.heads.head.out_features

        print(self.fc_out_features)

        # self.vit = torch.nn.Sequential(*(list(self.vit.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_out_features * 2, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.vit.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input1, input2):
        output1 = self.vit(input1)
        output1 = output1.view(output1.size()[0], -1)
        output2 = self.vit(input2)
        output2 = output2.view(output1.size()[0], -1)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output