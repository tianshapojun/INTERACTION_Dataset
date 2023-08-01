import warnings
from typing import Dict, List
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import resnet18, resnet50

def SimpleCNN_GN(num_input_channels: int, features_dim: int) -> nn.Module:
    """A simplified feature extractor with GroupNorm.

    :param num_input_channels: the number of input channels in the input
    :param features_dim: the number of features to extract from input
    """
    model = nn.Sequential(
        nn.Conv2d(num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.GroupNorm(4, 64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.GroupNorm(2, 32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=1568, out_features=features_dim),
    )

    return model

class RasterizedPlanningModel(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True
    ) -> None:
        """Initializes the planning model.

        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        if model_arch == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(weights = models.ResNet50_Weights.DEFAULT)
            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
        elif model_arch == "simple_cnn":
            self.model = SimpleCNN_GN(self.num_input_channels, num_targets)
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        if model_arch in {"resnet18", "resnet50"} and self.num_input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = torch.cat((data_batch["box_img"],data_batch["way_img"],data_batch["rel_img"]), dim = -1)
        image_batch = image_batch.permute(0,3,1,2)
        
        # [batch_size, num_steps * 3]
        outputs = self.model(image_batch)
        batch_size = len(data_batch["box_img"])
        #print(outputs)

        if self.training:
            #print(outputs)
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
            # [batch_size, num_steps * 3]
            targets = data_batch['ego_targets'].view(batch_size, -1)
            target_weights = (data_batch['ego_availabilities'].unsqueeze(-1) * self.weights_scaling).view(batch_size, -1)
            loss = torch.mean(self.criterion(outputs, targets) * target_weights)
            train_dict = {"loss": loss}
            #print(targets)
            return train_dict
        else:
            #print(outputs)
            predicted = outputs.view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = predicted[:, :, 2:3]
            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            return eval_dict
        
