import copy
import numpy as np
import torch
import torchvision
import torchvision.transforms as v2
from torchvision.models.feature_extraction import create_feature_extractor
from pytorch_lightning import seed_everything
import collections


def freeze_model_weights(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model_weights(model):
    for param in model.parameters():
        param.requires_grad = True


def get_out_layer_name_from_config(
    model_name,
    **kwargs
):
    if "resnet" in model_name:
        if kwargs.get('add_linear_layers'):
            linear_layers = kwargs['add_linear_layers']
            if len(linear_layers) >= 2:
                return (f"fc.outlayer_{len(linear_layers) + 1}", f"fc.outlayer_{len(linear_layers)}")
            return (f"fc.outlayer_1", f"fc.outlayer")
        return ("fc", "avgpool")
    elif "alexnet" in model_name:
        return ("classifier.6", "classifier.5")
    elif "vgg" in model_name:
        return ("classifier.6", "classifier.4")
    elif "squeezenet" in model_name:
        return ("classifier.3", "classifier.2")
    elif "densenet" in model_name:
        return ("classifier", "features.norm")
    elif model_name == "mlp":
        return (str(len(kwargs['layer_sizes'])*2 +1), str(len(kwargs['layer_sizes'])*2 - 1))
    elif model_name == "cnn":
        return (str(len(kwargs['layers'])*2 + 1), str(len(kwargs['layers'])*2 - 1))
    else:
        raise ValueError(f"Unsupported architecture {model_name}")


def get_latent_dim_size_from_config(
    model_name,
    **kwargs
):
    if "resnet" in model_name:
        if "resnet50" in model_name:
            return 2048
        return 512
    elif "alexnet" in model_name:
        return 4096
    elif "vgg" in model_name:
        return 4096
    elif "squeezenet" in model_name:
        return 512
    elif "densenet" in model_name:
        return 1024
    else:
        raise ValueError(f"Unsupported TorchVision architecture {model_name}")


def load_vision_model(
    name,
    output_classes,
    imagenet_pretrained=False,
    output_last_layer=False,
    freeze_weights=False,
    add_linear_layers=None,
):
    """Loads a pretrained ImageNet vision model from the torchvision library
    and modifies its output layer so that it outputs `output_classes` instead
    so that this model can be used for training on a separate task.

    Args:
        name (str): A valid architecture name for a torchvision model. See
            official documentation for all supported architecture names.
        output_classes (int): The number of outputs we wish to have in the
            output model.
        imagenet_pretrained (bool, optional): If True, then we load the
            pretrained ImageNet weights on the requested model for all layers
            except for the last one. Defaults to False.

    Raises:
        ValueError: If provided architecture name is not a valid supported
            torchvision model.

    Returns:
        torch.Module: Initialized torch model containing the architectured
            request but with the output layer having exactly `output_classes`
            classes.
    """
    # Because we are working with torchvision 1.12, we iterate over all
    # possibilities
    if name == "resnet18":
        model = torchvision.models.resnet18(
            pretrained=imagenet_pretrained
        )
    elif name == "resnet34":
        model = torchvision.models.resnet34(
            pretrained=imagenet_pretrained
        )
    elif name == "resnet50":
        model = torchvision.models.resnet50(
            pretrained=imagenet_pretrained,
        )
    elif name == "resnet101":
        model = torchvision.models.resnet101(
            pretrained=imagenet_pretrained
        )
    elif name == "resnet152":
        model = torchvision.models.resnet152(
            pretrained=imagenet_pretrained
        )
    elif name == "alexnet":
        model = torchvision.models.alexnet(
            pretrained=imagenet_pretrained
        )
    elif name == "vgg16":
        model = torchvision.models.vgg16(
            pretrained=imagenet_pretrained
        )
    elif name == "vgg19":
        model = torchvision.models.vgg19(
            pretrained=imagenet_pretrained
        )
    elif name == "squeezenet1_0":
        model = torchvision.models.squeezenet1_0(
            pretrained=imagenet_pretrained
        )
    elif name == "densenet121":
        model = torchvision.models.densenet121(
            pretrained=imagenet_pretrained
        )
    elif name == "densenet161":
        model = torchvision.models.densenet161(
            pretrained=imagenet_pretrained
        )
    elif name == "densenet169":
        model = torchvision.models.densenet169(
            pretrained=imagenet_pretrained
        )
    elif name == "densenet201":
        model = torchvision.models.densenet201(
            pretrained=imagenet_pretrained
        )
    elif name == "inception_v3":
        model = torchvision.models.inception_v3(
            pretrained=imagenet_pretrained
        )
    elif name == "googlenet":
        model = torchvision.models.googlenet(
            pretrained=imagenet_pretrained
        )
    elif name == "shufflenet_v2_x1_0":
        model = torchvision.models.shufflenet_v2_x1_0(
            pretrained=imagenet_pretrained
        )
    elif name == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(
            pretrained=imagenet_pretrained
        )
    elif name == "mobilenet_v3_large":
        model = torchvision.models.mobilenet_v3_large(
            pretrained=imagenet_pretrained
        )
    elif name == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small(
            pretrained=imagenet_pretrained
        )
    elif name == "resnext50_32x4d":
        model = torchvision.models.resnext50_32x4d(
            pretrained=imagenet_pretrained
        )
    elif name == "wide_resnet50_2":
        model = torchvision.models.wide_resnet50_2(
            pretrained=imagenet_pretrained
        )
    elif name == "mnasnet1_0":
        model = torchvision.models.mnasnet1_0(
            pretrained=imagenet_pretrained
        )
    elif name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(
            pretrained=imagenet_pretrained
        )
    elif name == "efficientnet_b1":
        model = torchvision.models.efficientnet_b1(
            pretrained=imagenet_pretrained
        )
    elif name == "efficientnet_b2":
        model = torchvision.models.efficientnet_b2(
            pretrained=imagenet_pretrained
        )
    elif name == "efficientnet_b3":
        model = torchvision.models.efficientnet_b3(
            pretrained=imagenet_pretrained
        )
    elif name == "efficientnet_b4":
        model = torchvision.models.efficientnet_b4(
            pretrained=imagenet_pretrained
        )
    elif name == "efficientnet_b5":
        model = torchvision.models.efficientnet_b5(
            pretrained=imagenet_pretrained
        )
    elif name == "efficientnet_b6":
        model = torchvision.models.efficientnet_b6(
            pretrained=imagenet_pretrained
        )
    elif name == "efficientnet_b7":
        model = torchvision.models.efficientnet_b7(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_y_400mf":
        model = torchvision.models.regnet_y_400mf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_y_800mf":
        model = torchvision.models.regnet_y_800mf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_y_1_6gf":
        model = torchvision.models.regnet_y_1_6gf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_y_3_2gf":
        model = torchvision.models.regnet_y_3_2gf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_y_8gf":
        model = torchvision.models.regnet_y_8gf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_y_16gf":
        model = torchvision.models.regnet_y_16gf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_y_32gf":
        model = torchvision.models.regnet_y_32gf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_y_128gf":
        model = torchvision.models.regnet_y_128gf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_x_400mf":
        model = torchvision.models.regnet_x_400mf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_x_800mf":
        model = torchvision.models.regnet_x_800mf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_x_1_6gf":
        model = torchvision.models.regnet_x_1_6gf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_x_3_2gf":
        model = torchvision.models.regnet_x_3_2gf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_x_8gf":
        model = torchvision.models.regnet_x_8gf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_x_16gf":
        model = torchvision.models.regnet_x_16gf(
            pretrained=imagenet_pretrained
        )
    elif name == "regnet_x_32gf":
        model = torchvision.models.regnet_x_32gf(
            pretrained=imagenet_pretrained
        )
    elif name == "vit_b_16":
        model = torchvision.models.vit_b_16(
            pretrained=imagenet_pretrained
        )
    elif name == "vit_b_32":
        model = torchvision.models.vit_b_32(
            pretrained=imagenet_pretrained
        )
    elif name == "vit_l_16":
        model = torchvision.models.vit_l_16(
            pretrained=imagenet_pretrained
        )
    elif name == "vit_l_32":
        model = torchvision.models.vit_l_32(
            pretrained=imagenet_pretrained
        )
    else:
        raise ValueError(
            f'Unsupported pretrained model {name}'
        )
    latent_dim_size = get_latent_dim_size_from_config(name)

    if freeze_weights:
        freeze_model_weights(model)

    if "resnet" in name:
        if add_linear_layers:
            units = [latent_dim_size] + add_linear_layers + [output_classes]
            layers = []
            for i in range(1, len(units)):
                layers.append((f"nonlin_{i}", torch.nn.LeakyReLU()))
                layers.append((f"outlayer_{i}", torch.nn.Linear(units[i-1], units[i])))
            model.fc = torch.nn.Sequential(collections.OrderedDict(layers))
        else:
            model.fc = torch.nn.Linear(latent_dim_size, output_classes)
        if output_last_layer:
            logit_name, latent_name = get_out_layer_name_from_config(name)
            model = create_feature_extractor(
                model,
                return_nodes={
                    logit_name: "logits",
                    latent_name: "latent",
                },
            )
    elif "alexnet" in name:
        assert add_linear_layers is None, 'unsupported'
        model.classifier[6] = torch.nn.Linear(latent_dim_size, output_classes)
        if output_last_layer:
            logit_name, latent_name = get_out_layer_name_from_config(name)
            model = create_feature_extractor(
                model,
                return_nodes={
                    logit_name: "logits",
                    latent_name: "latent",
                },
            )
    elif "vgg" in name:
        assert add_linear_layers is None, 'unsupported'
        model.classifier[6] = torch.nn.Linear(latent_dim_size, output_classes)
        if output_last_layer:
            logit_name, latent_name = get_out_layer_name_from_config(name)
            model = create_feature_extractor(
                model,
                return_nodes={
                    logit_name: "logits",
                    latent_name: "latent",
                },
            )
    elif "squeezenet" in name:
        assert add_linear_layers is None, 'unsupported'
        model.classifier[1] = torch.nn.Conv2d(
            latent_dim_size,
            output_classes,
            kernel_size=(1,1),
            stride=(1,1),
        )
        if output_last_layer:
            logit_name, latent_name = get_out_layer_name_from_config(name)
            model = create_feature_extractor(
                model,
                return_nodes={
                    logit_name: "logits",
                    latent_name: "latent",
                },
            )
    elif "densenet" in name:
        assert add_linear_layers is None, 'unsupported'
        model.classifier = torch.nn.Linear(latent_dim_size, output_classes)
        if output_last_layer:
            logit_name, latent_name = get_out_layer_name_from_config(name)
            model = create_feature_extractor(
                model,
                return_nodes={
                    logit_name: "logits",
                    latent_name: "latent",
                },
            )
    else:
        raise ValueError(f"Unsupported pretrained architecture {name}")
    return model


def is_torchvision_model(architecture):
    """Check if given string is a supported torchvision model.
    The current implementation is a bit flimsy/error-prone, but will be soon
    updated once torchvision's local version is updated.

    Args:
        architecture (str): a valid name of a torchvision classification model.
            See official torchvision documentation for details on supported
            architectures.

    Returns:
        bool: whether or not the requested name is a valid architecture name for
            torchvision classification models.
    """
    # if using a more modern version of torchvision, this should work:
    # return architecture in torchvision.models.list_models(
    #     module=torchvision.models
    # )
    for supported in ["resnet", "alexnet", "vgg", "squeezenet", "densenet"]:
        if supported in architecture:
            return True
    return False

def construct_standard_model(
    architecture,
    input_shape,
    n_labels,
    seed=None,
    pretrained_model=None,
    output_last_layer=False,
    **kwargs,
):
    """Constructs a torch.Module with the corresponding architecture which
    takes as an input a sample of shape `input_shape` and produces `n_labels`
    outputs.

    Args:
        architecture (str): A string represented a valid and supported
            architecture. Currently this corresponds to supported vision
            architecture names in the torchvision library.
        input_shape (tuple): Shape, without batch dimension, of each input
            sample to be fed into this model.
        n_labels (int): Number of outputs we wish to have in the resulting
            model.
        seed (int, optional): Random seed to use at initialization. Defaults to
            None.
        pretrained_model (torch.Module, optional): Whether we wish to use a
            pretrained model to initialize the weights of the output model.
            If provided, we will initialize weights in both models whose names
            are shared, while silently ignoring unmatching weights. Defaults to
            None.

    Raises:
        ValueError: If provided architecture string is not a supporting
            architecture or if the input shape is not aligned with the necessary
            input shape for the requested architecture.

    Returns:
        (torch.Module, bool): A tuple (model, pretrained) containing the
            constructed `model` and a boolean indicating whether its weights
            were initialized from a pretrained model or not.
    """
    if seed:
        seed_everything(seed)
    if is_torchvision_model(architecture):
        # if (input_shape[-1] != 224) or (input_shape[-2] != 224):
        #     raise ValueError(
        #         f"To use pretrain model {architecture} the input shape must "
        #         f"be [..., 224, 224] but instead got {input_shape}"
        #     )
        model = load_vision_model(
            name=architecture,
            output_classes=kwargs.get('output_classes', n_labels),
            imagenet_pretrained=kwargs.get("imagenet_pretrained", False),
            output_last_layer=output_last_layer,
            freeze_weights=kwargs.get("freeze_weights", False),
            add_linear_layers=kwargs.get('add_linear_layers'),
        )
        pretrained = kwargs.get("imagenet_pretrained", False)

    elif architecture.lower().replace(" ", "") == "mlp":
        layers = [torch.nn.Flatten()]
        current_shape = np.prod(input_shape)
        for num_acts in kwargs['layer_sizes']:
            layers.append(torch.nn.Linear(
                current_shape,
                num_acts,
            ))
            current_shape = num_acts
            act_fn = kwargs.get(
                'activation_fn',
                'relu',
            )
            if act_fn.lower() == 'relu':
                layers.append(torch.nn.ReLU())
            else:
                raise ValueError(
                    f'Unsupported activation {kwargs["activation_fn"]}'
                )
        layers.append(torch.nn.Linear(
                current_shape,
                kwargs.get('output_classes', n_labels),
            ))
        model = torch.nn.Sequential(*layers)
        pretrained = False

    elif architecture.lower().replace(" ", "") == "cnn":
        layers = []
        current_shape = input_shape
        for layer in kwargs['layers']:
            layers.append(torch.nn.Conv2d(
                in_channels=current_shape[0],
                out_channels=layer['out_channels'],
                kernel_size=layer['filter'],
                padding='same',
            ))
            current_shape = (layer['out_channels'], *current_shape[1:])
            act_fn = layer.get(
                'activation_fn',
                kwargs.get('activation_fn', 'relu'),
            )
            if act_fn.lower() == 'relu':
                layers.append(torch.nn.ReLU())
            else:
                raise ValueError(
                    f'Unsupported activation {act_fn}'
                )
            if layer.get('max_pool_kernel', False):
                kernel = layer['max_pool_kernel']
                layers.append(
                    torch.nn.MaxPool2d(kernel)
                )
                current_shape = (
                    current_shape[0],
                    current_shape[1] - (kernel[0] - 1),
                    current_shape[2] - (kernel[1] - 1),
                )
            elif layer.get('avg_pool_kernel', False):
                kernel = layer['avg_pool_kernel']
                layers.append(
                    torch.nn.AvgPool2d(kernel)
                )
                current_shape = (
                    current_shape[0],
                    current_shape[1] - (kernel[0] - 1),
                    current_shape[2] - (kernel[1] - 1),
                )
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(
                np.prod(current_shape),
                kwargs.get('output_classes', n_labels),
            ))
        model = torch.nn.Sequential(*layers)
        pretrained = False
    else:
        # Here is where we can add support for custom MLPs, CNNs, etc etc
        raise ValueError(f"Architecture {architecture} not supported yet!")

    if pretrained_model is not None:
        # Then time to load the weights into the model!
        current_model_dict = model.state_dict()
        pretrained_state_dict = pretrained_model.state_dict()
        new_state_dict = {
            k: (
                v if v.size() == current_model_dict[k].size() else
                current_model_dict[k]
            ) for k, v in zip(
                current_model_dict.keys(),
                pretrained_state_dict.values(),
            )
        }
        model.load_state_dict(new_state_dict, strict=False)
        pretrained = True
    return model, pretrained