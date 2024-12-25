import numpy as np
import torch
import torchvision.transforms as v2

################################################################################
## DATASET MANIPULATION
################################################################################


def gauss_noise_tensor(img, sigma):
    assert isinstance(img, torch.Tensor), type(img).__name__
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    out = img + sigma * torch.randn_like(img)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out

def salt_and_pepper_noise_tensor(
    img,
    s_vs_p=0.5,
    amount=0.05,
    all_channels=False,
):
    assert isinstance(img, torch.Tensor), type(img).__name__
    dtype = img.dtype
    max_elem, min_elem = 1.0, 0.0
    if not img.is_floating_point():
        img = img.to(torch.float32)/255.0

    out = img + 0.0

    # Salt mode
    shape_to_use = img.shape[:-1] if all_channels else img.shape
    num_salt = np.ceil(amount * np.prod(tuple(shape_to_use)) * s_vs_p)
    coords = [
        np.random.randint(0, i - 1, int(num_salt))
        for i in tuple(shape_to_use)
    ]
    if all_channels:
        out[coords, :] = max_elem
    else:
        out[coords] = max_elem


    # Pepper mode
    num_pepper = np.ceil(amount* np.prod(tuple(shape_to_use)) * (1. - s_vs_p))
    coords = [
        np.random.randint(0, i - 1, int(num_pepper))
        for i in tuple(shape_to_use)
    ]
    if all_channels:
        out[coords, :] = min_elem
    else:
        out[coords] = min_elem

    return out

class LambdaDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        outputs = self.subset[index]
        if isinstance(outputs, (list, tuple)):
            x = outputs[0]
        else:
            x = outputs
        if self.transform:
            x = self.transform(x)
        if isinstance(outputs, (list, tuple)):
            return (x, *outputs[1:])
        return x

    def __len__(self):
        return len(self.subset)


def transform_from_config(transform):
    """
    Simple function to generate a torchvision transform from its dictionary
    representation as provided by the `transform` input config.
    """
    if isinstance(transform, list):
        return v2.Compose([
            transform_from_config(x) for x in transform
        ])
    if (transform is None) or (transform == {}):
        return lambda x: x
    transform_name = transform["name"].lower().strip()
    if transform_name == "identity":
        return lambda x: x

    if transform_name in ["gaussian_noise", "gaussiannoise"]:
        return lambda x: gauss_noise_tensor(x, sigma=transform.get('sigma', 1))

    if transform_name in ["salt_and_pepper", "s&p", "saltandpepper"]:
        return lambda x: salt_and_pepper_noise_tensor(
            x,
            s_vs_p=transform.get('s_vs_p', 0.5),
            amount=transform.get('amount', 0.01),
            all_channels=transform.get('all_channels', False),
        )
    if transform_name == "random_noise":
        if transform['noise_level'] > 0.0:
            def _trans(x):
                mask = np.random.choice(
                    [0, 1],
                    size=x.shape,
                    p=[1 - transform['noise_level'], transform['noise_level']],
                )
                mask = torch.tensor(mask).to(x.device).type(
                    x.type()
                )
                substitutes = np.random.uniform(
                    low=0,
                    high=transform['low_noise_level'],
                    size=x.shape,
                )
                substitutes = torch.tensor(substitutes).to(x.device).type(
                    x.type()
                )
                return mask * substitutes + (1 - mask) * x
        return _trans

    if transform_name == "randomapply":
        return v2.RandomApply(
            transforms=list(map(
                transform_from_config,
                transform['transforms'],
            )),
            p=transform['p'],
        )
    if transform_name == "randomadjustsharpness":
        return v2.RandomAdjustSharpness(
            sharpness_factor=transform.get('sharpness_factor', 2),
        )
    if transform_name == "gaussianblur":
        return v2.GaussianBlur(
            kernel_size=transform.get('kernel_size', (5, 5)),
            sigma=transform.get('sigma', (0.1, 2.)),
        )
    if transform_name == "randomaffine":
        return v2.RandomAffine(
            degrees=transform.get('degrees', 0),
            translate=transform.get('translate', None),
            scale=transform.get('scale', None),
        )
    if transform_name == "randaugment":
        return v2.RandAugment(
            num_ops=transform.get('num_ops', 2),
            magnitude=transform.get('magnitude', 9),
            num_magnitude_bins=transform.get('num_magnitude_bins', 31),
        )
    if transform_name == "convertimagedtype":
        return v2.ConvertImageDtype(
            dtype=getattr(torch, transform['dtype']),
        )
    if transform_name == "normalize":
        return v2.Normalize(
            mean=transform['mean'],
            std=transform['std'],
        )
    raise ValueError(
        f'Unsupported transformation {transform_name}'
    )
