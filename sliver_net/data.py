
import numpy as np

from torch.utils.data import Dataset
import PIL
from skimage import exposure


""" NOTE: This is just a sample dataset... it's not guaranteed that this will work"""

class SampleDataset(Dataset):

    def __init__(self, volumes, labels):
        """
        Args:
            volumes: list of 3-d volumes (recommend providing paths and loading them on-the-fly if the dataset is large)
            labels: list of labels(arrays or int) for each volume
        """
        self.items = list(zip(volumes, labels))
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        
        slices, l = self.items[idx]
        # v is of shape (n_slices x H x W)
        
        # contrast stretch
        slices = [pil_contrast_strech()(image) for image in slices]
        
        # assuming we had a single channel input, just concatenate them to make 3 channels
        v = torch.cat(slices,dim=1)
        # v is of shape ([n_slices * H] x W)
        v = torch.stack([v,v,v]).squeeze()
        # v is of shape(3 x [n_slices * H] x W)
        
        # return sample
        return v, l

 class  pil_contrast_strech(object):
    def __init__(self,low=2,high=98):
        self.low,self.high = low,high

    def __call__(self,img):
        # Contrast stretching
        img=np.array(img)
        plow, phigh = np.percentile(img, (self.low, self.high))
        return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))
