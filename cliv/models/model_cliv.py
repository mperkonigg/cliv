import torch
from torch import nn
import numpy as np
from segmentation_models_pytorch import Unet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# this code is taken from: https://github.com/arneschmidt/pionono_segmentation and remodeled quite a bit

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

class LatentVariable(nn.Module):
    def __init__(self, num_annotators: int, latent_dims: int=2, init_sigma=8.0):
        """This module defines the random latent variable z with distribution q(z|r) with r being the rater.

        Args:
            num_annotators (int): number of annotator distributions within the latent variable
            latent_dims (int, optional): dimensions of latent distributions. Defaults to 2.
        """
        super(LatentVariable, self).__init__()
        self.latent_dims = latent_dims
        self.no_annotators = num_annotators

        z_values = np.random.standard_normal(size=[num_annotators, latent_dims])*init_sigma
        self.z_vecs = nn.ParameterList([torch.nn.Parameter(torch.tensor(zval)) for zval in z_values])
        self.name = 'LatentVariable'

    def forward(self, annotator: torch.Tensor):
        z = torch.zeros([len(annotator), self.latent_dims]).to(device)
        annotator = annotator.long()
        for i in range(len(annotator)):
            a = annotator[i]
            z_i = self.z_vecs[a]
            z[i] = z_i
        return z

class ClivHead(nn.Module):
    """
    The Segmentation head combines the sample taken from the latent space,
    and feature map by concatenating them along their channel axis.
    """
    def __init__(self, num_filters_last_layer: int, latent_dim: int, num_output_channels: int, num_classes:int, no_convs_fcomb:int,
                 head_kernelsize:int, head_dilation, use_tile:bool=True, spatial_axes: list[int]=[2, 3]):
        super(ClivHead, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = spatial_axes
        self.num_filters_last_layer = num_filters_last_layer
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.head_kernelsize = head_kernelsize
        self.name = 'ClivHead'

        if len(self.spatial_axes)==3:
            conv_f = nn.Conv3d
        else:
            conv_f = nn.Conv2d

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(conv_f(self.num_filters_last_layer+self.latent_dim, self.num_filters_last_layer,
                                    kernel_size=self.head_kernelsize, dilation=head_dilation, padding='same'))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(conv_f(self.num_filters_last_layer, self.num_filters_last_layer,
                                        kernel_size=self.head_kernelsize, dilation=head_dilation, padding='same'))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)
            self.last_layer = conv_f(self.num_filters_last_layer, self.num_classes, kernel_size=self.head_kernelsize,
                                        dilation=head_dilation, padding='same')
            self.activation = torch.nn.Softmax(dim=1)

            self.layers.apply(self.initialize_weights)
            self.last_layer.apply(self.initialize_weights)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z, use_softmax=True):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            for sa in self.spatial_axes:
                z = torch.unsqueeze(z, sa)
                z = self.tile(z, sa, feature_map.shape[sa])

            # Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            x = self.layers(feature_map)
            y = self.last_layer(x)
            if use_softmax:
                y = self.activation(y)
            return y

class ClivModel(nn.Module):
    """
    The implementation of the Cliv Model. It consists of a segmentation backbone, probabilistic latent variable and
    segmentation head.
    """

    def __init__(self, input_channels=3, num_classes=1, annotators=6, latent_dim=8, init_sigma=8.0, no_head_layers=3, head_kernelsize=1,
                 head_dilation=1, seg_model: torch.nn.Module=None, pretrain_path=None, train_annotators=None, spatial_axes=[2,3]):
        super(ClivModel, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.annotators = annotators
        self.no_head_layers = no_head_layers
        self.head_kernelsize = head_kernelsize
        self.head_dilation = head_dilation
        self.train_annotators = train_annotators
        self.unet = seg_model
        self.spatial_axes = spatial_axes

        self.z = LatentVariable(len(annotators), latent_dim, init_sigma=init_sigma).to(device)
        self.head = ClivHead(16, self.latent_dim, input_channels, self.num_classes,
                                self.no_head_layers, self.head_kernelsize, self.head_dilation, use_tile=True, spatial_axes=spatial_axes).to(device)
        self.phase = 'segmentation'
        self.name = 'ClivModel'

        if pretrain_path is not None:
            weights = torch.load(pretrain_path)['state_dict']
            final_weights = {}

            # set z weights to be overwrittern afterwards
            for name, param in self.z.named_parameters():
                final_weights[f"z.{name}"] = param

            for w in weights:
                if w.startswith("model."):
                    final_weights[w[6:]] = weights[w]
                elif not w.startswith("gen_model."):
                    final_weights[w] = weights[w]
            
            self.load_state_dict(final_weights)
        self.pretrain_path = pretrain_path


    def forward(self, batch, annotator_ids: torch.Tensor, annotator_list: list = None):
        if annotator_list is not None:
            ann_ids = self.map_annotators_to_correct_id(ann_ids, annotator_list)

        self.unet_features = self.unet.forward(batch)
        z = self.z.forward(annotator_ids)
        pred = self.head.forward(self.unet_features, z, False)

        return pred

    def map_annotators_to_correct_id(self, annotator_ids: torch.Tensor, annotator_list:list = None):
        new_ids = torch.zeros_like(annotator_ids).to(device)
        for a in range(len(annotator_ids)):
            id_corresponds = (annotator_list[int(annotator_ids[a])] == np.array(self.annotators))
            if not np.any(id_corresponds):
                raise Exception('Annotator has no corresponding distribution. Annotator: ' + str(annotator_list[int(annotator_ids[a])]))
            new_ids[a] = torch.nonzero(torch.tensor(annotator_list[int(annotator_ids[a])] == np.array(self.annotators)))[0][0]
        return new_ids

    def train_step(self, images, labels, loss_fct, ann_ids):
        """
        Make one train forward, returning loss and predictions.
        """

        for p in self.z.parameters():
            w = p.data
            w = w.clamp(0.0001)
            p.data = w

        if self.train_annotators is not None:
            ann_ids = self.map_annotators_to_correct_id(ann_ids, self.train_annotators)

        y_pred = self.forward(images, annotator_ids=ann_ids)
        loss = loss_fct(y_pred, labels)

        return loss, y_pred
