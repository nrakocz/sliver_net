# SLIVER Net

This repository contains the source and analysis code for SLIVER-NET, published in the article [Automated identification of clinical features from sparsely annotated 3-dimensional medical imaging](https://www.nature.com/articles/s41746-021-00411-w) by Rakocz and colleagues in _npj Digital Medicine_.

The SLIVER-Net model was implemented in PyTorch, and utilities from [fastai](https://docs.fast.ai/) were used to train the model.

The current implementation of the model can be found in the `sliver_net` directory. To load the pytorch model, run the following:

```
from sliver_net.model import load_backbone
from sliver_net.model import SliverNet2

backbone = load_backbone("scratch") # randomly initialize a resnet18 backbone
model = SliverNet2(backbone, n_out=2) # create SLIVER-Net with n_out outputs
```


A minimal data loader example is provided in `sliver_net/data.py`. Our approach handles 3D volumes by rearranging them into a 2D "Tile" of slices and processing the entire tile as a single 2D image. Then, a 1D convolution is run over the resulting feature representation to account for the third dimension. Thus the data loader for the pytorch model should accept a 3D volumetric input (in our case, 97 x 256 x 256) and return a tiled image of shape (3 x H x W) - in our case (3 x [97x256] x 256) where 97 is the number of slices in the 3D volume. We performed contrast stretching on each slice as the only preprocessing step.

Please cite the following article if this model was useful to your work.
`Rakocz, N., Chiang, J.N., Nittala, M.G. et al. Automated identification of clinical features from sparsely annotated 3-dimensional medical imaging. npj Digit. Med. 4, 44 (2021). https://doi.org/10.1038/s41746-021-00411-w`

