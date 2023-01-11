import torch
from torch.utils.data import Subset
from torchvision.datasets import CelebA
from torchvision.transforms import Resize, ToTensor, Compose
import numpy as np

all_attr = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
                 'Double_Chin',
                 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
                 ]
# class_names = ['Attractive', 'Male', 'Young']
class_names = ['Male']
class_pos = [all_attr.index(c) for c in class_names]
attr_pos = [a for a in range(len(all_attr)) if a not in class_pos]

class ToHierarchy(object):
    def __init__(self, class_pos, attr_pos):
        self.class_pos = class_pos
        self.attr_pos = attr_pos

    def __call__(self, attr):
        return attr[self.attr_pos], torch.cat([attr[self.class_pos], 1-attr[self.class_pos]])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def load_celeba(root='.', img_size=64):
    transforms = [
        Resize([img_size, img_size]),
        ToTensor(),
    ]
    dataset = CelebA(root=root, split='test',
                     target_type='attr', download=True,
                     transform=Compose(transforms),
                     target_transform=ToHierarchy(class_pos, attr_pos))
    concept_names = np.array(all_attr)[attr_pos].tolist()
    train_idx = sorted(np.random.choice(np.arange(len(dataset)), size=int(0.8*len(dataset)), replace=False))
    test_idx = np.setdiff1d(np.arange(len(dataset)), train_idx)
    train_data = Subset(dataset, train_idx)
    test_data = Subset(dataset, test_idx)
    return train_data, test_data, len(attr_pos), 2*len(class_pos), concept_names, ['Male', 'Female']
