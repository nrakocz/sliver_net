import logging
import os

import torch
import torchvision.models as tmodels
import torch.nn.functional as F


def load_backbone(model_name):
    
    kermany_pretrained_weights = "/opt/data/commonfilesharePHI/jnchiang/projects/Ophth/iRORAcRORA/models/kermany_pretrained.pth"
    
    if "imagenet" in str(model_name).lower():
        logging.info("Loading ImageNet Model")
        model = tmodels.resnet18(pretrained=True)
    elif "kermany" in str(model_name).lower():
        logging.info("Loading model from Kermany")
        model = tmodels.resnet18(num_classes=4, pretrained=False)
        model_weights = kermany_pretrained_weights
        model.load_state_dict(torch.load(model_weights, map_location=torch.device("cuda"))['model'])
    elif "sliver" in str(model_name).lower():
        logging.info("Loading model from Kermany for SLIVER-NET")
        model = tmodels.resnet18(num_classes=4, pretrained=False)
        model_weights = kermany_pretrained_weights
        model.load_state_dict(torch.load(model_weights, map_location=torch.device("cuda"))['model'])
    else:
        logging.info("Loading model from scratch")
        model = tmodels.resnet18(pretrained=False)
    
    # This will be 512 units... HARD CODED b/c resnet
    # after hacking off the FC and pooling layer, Resnet18 downsamples 5x
    # output size: B x C x H // 32 x W // 32
    return torch.nn.Sequential(*list(model.children())[:-2])

def nonadaptiveconcatpool2d(x, k):
    # concatenating average and max pool, with kernel and stride the same
    ap = F.avg_pool2d(x, kernel_size=(k, k), stride=(k, k))
    mp = F.max_pool2d(x, kernel_size=(k, k), stride=(k, k))
    return torch.cat([mp, ap], 1) 
   
class FeatureCNN2(torch.nn.Module):
    # def __init__(self,input_size,n_out=1, input_channels=1024,ncov=0,kernel_size=3,pool_size=4,add_layers=False):
    def __init__(self,n_out=1, input_channels=1024,ncov=0,kernel_size=3,pool_size=4,add_layers=False):
        super(FeatureCNN2, self).__init__()
        CONV1_FILTERS = 16
        CONV2_FILTERS = 32
        CONV3_FILTERS = 64
        CONV4_FILTERS = 128
        FC1_OUT = 24
        
        self.add_layers = add_layers
        self.ncov = ncov
        self.conv1 = torch.nn.Conv1d(input_channels, CONV1_FILTERS, kernel_size=kernel_size, stride=1, padding=0)
        # output size: N-(kernel_size-1)
        self.conv2 = torch.nn.Conv1d(CONV1_FILTERS, CONV2_FILTERS, kernel_size=kernel_size, stride=1, padding=0)
        self._conv_filters = CONV2_FILTERS
        # output size: (N-2(kernel_size-1))//pool_stride
        self._nlayers = 2
        # out_size = input_size-self._nlayers*(kernel_size-1)
        
        if(add_layers):
            self.poolMid = torch.nn.MaxPool1d(kernel_size=pool_size, padding= pool_size//2 ,stride=pool_size)
            # out_size = (out_size//pool_size)+1
            self.conv3 = torch.nn.Conv1d(CONV2_FILTERS, CONV3_FILTERS, kernel_size=kernel_size, stride=1, padding=0)
            self.conv4 = torch.nn.Conv1d(CONV3_FILTERS, CONV4_FILTERS, kernel_size=kernel_size, stride=1, padding=0)
            # out_size = out_size - 2*(kernel_size-1)
            self._nlayers = 4
            self._conv_filters = CONV4_FILTERS
        
        # output size: N-2(kernel_size-1)
        # self.pool = torch.nn.MaxPool1d(kernel_size=out_size, padding=0)
        # dynamic global pooling instead of computing the size
        self.pool = torch.nn.AdaptiveMaxPool1d(1, return_indices=True)
        
        self._output_size = 1
        self.fc1 = torch.nn.Linear(self._conv_filters + ncov, FC1_OUT)
        self.fc2 = torch.nn.Linear(FC1_OUT, n_out)
    
    def forward(self,x,cov=None):
        # x: B x C x N
        x = self.conv1(x)
        # x: B x C x N-(kernel_size-1) // 1
        x = F.relu(x)
        x = self.conv2(x)
        # x: B x C x N - 2(kernel_size-1) // 1
        x = F.relu(x)
        if(self.add_layers):
            x = self.poolMid(x)
            # x: B x C x (N - (2(kernel_size-1) // 1) // pool_size) + 1
            x = self.conv3(x)
            # x: B x C x (N - (2(kernel_size-1) // 1) // pool_size) + 1 - (kernel_size-1) // 1
            x = F.relu(x)
            # x: B x C x (N - (2(kernel_size-1) // 1) // pool_size) + 1 - 2(kernel_size-1) // 1
            x = self.conv4(x)
            x = F.relu(x)
            
        x, idx = self.pool(x)
        # x: B x C x 1
        x = x.view(-1, self._conv_filters)
        # x: B x C
#         x = x.flatten()
        if(self.ncov): x=torch.cat((x,cov),dim=1)
        # x: B x C+ncov
        x = self.fc1(x)
        # x: B x n_FC
        x = F.relu(x)
        x = self.fc2(x)
        # x: B x n_out
        return x  
    
    def max_slice(self, x):
        # x: B x C x N
        x = self.conv1(x)
        # x: B x C x N-(kernel_size-1) // 1
        x = F.relu(x)
        x = self.conv2(x)
        # x: B x C x N - 2(kernel_size-1) // 1
        x = F.relu(x)
        if(self.add_layers):
            x = self.poolMid(x)
            # x: B x C x (N - (2(kernel_size-1) // 1) // pool_size) + 1
            x = self.conv3(x)
            # x: B x C x (N - (2(kernel_size-1) // 1) // pool_size) + 1 - (kernel_size-1) // 1
            x = F.relu(x)
            # x: B x C x (N - (2(kernel_size-1) // 1) // pool_size) + 1 - 2(kernel_size-1) // 1
            x = self.conv4(x)
            x = F.relu(x)
            
        _, idx = self.pool(x)   
        return idx
    
    
class SliverNet2(torch.nn.Module):
    def __init__(self, backbone=None, n_out=2, ncov=0,add_layers=False):
        super().__init__()
        if backbone is None:
            self.backbone=load_backbone(model_name="scratch", n_feats=n_feats)
        else:
            self.backbone = backbone
        # get_backbone(model_name, n_feats)  # change to load_backbone        
        # self.model = torch.nn.Sequential(self.model, NonAdaptiveConcatPool2d(8))
        self.cov_model = FeatureCNN2(ncov=ncov,n_out=n_out,add_layers=add_layers)
    
    def forward(self,x,cov=None):
        # B x C x (n_slices x orig_W) x orig_W
        x = self.backbone(x) # get the feature maps
        # B x C x (n_slices x W) x W
        kernel_size = x.shape[-1]  # W
        x = nonadaptiveconcatpool2d(x, kernel_size) # pool the feature maps with kernel and stride W
        # B x C x n_slices x 1
        x = x.squeeze(dim=-1) 
        # B x C x n_slices
        return self.cov_model(x,cov)  # 1d cnn, etc  
    
    def feature_maps(self, x):
        # generate heatmaps on each image in the batch
        # make sure model is in eval mode
        # to visualize image i out of b
        # b, c, h,w = x.shape
        # ax.imshow(img[i][1],cmap='gray',alpha=1)
        # ax.imshow(hm[i], alpha=0.3, extent=(0,w,h,0),
        #               interpolation='bilinear', cmap='magma')
        
        # x: B x C x (n_slices x orig_W) x orig_W
        hm = self.backbone(x)
        # hm: B x C x (n_slices x W) x W
        return torch.mean(hm, 1)
    
    def max_slices(self, x, kernel_size=3):
        # in progress.
        # eventually we will find the slice(s) that give the highest signal
        # given kernel size k and number of layers 2
        # output i is effected by slices: i: i+2(k-1)
        # so, to be conservative we can mark slices [i-2, i+2(k-1)]
        # for k=3 this will provide 7 slice [i-2,i+4]
        x = self.backbone(x)
        kernel_size = x.shape[-1]  # W
        x = nonadaptiveconcatpool2d(x, kernel_size) # pool the feature maps with kernel and stride W
        # B x C x n_slices x 1
        x = x.squeeze(dim=-1) 
        idx = self.cov_model.max_slice(x)
        idx = idx.view(-1, self.cov_model._conv_filters)

        return idx
        
