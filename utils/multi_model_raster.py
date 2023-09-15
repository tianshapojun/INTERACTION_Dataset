import warnings
from typing import Dict, List
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import resnet18, resnet50
import matplotlib.pyplot as plt
import cv2
import numpy as np

CV2_SUB_VALUES = {"shift": 9, "lineType": cv2.LINE_AA}
CV2_SHIFT_VALUE = 2 ** CV2_SUB_VALUES["shift"]

class RasterizedPlanningModel(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        ego_input_channels: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        lmbda,lmbda_occ,lmbda_ego,
        raster_size,pixel_size,ego_center,
        anchor_shape,
        pretrained: bool = True,
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
        self.ego_input_channels = ego_input_channels
        self.num_targets = num_targets
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion
        self.lmbda = lmbda 
        self.lmbda_occ = lmbda_occ 
        self.lmbda_ego = lmbda_ego
        self.raster_size = raster_size
        self.pixel_size = pixel_size
        self.ego_center = ego_center
        self.anchor_shape = anchor_shape

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        self.model = resnet50(weights = models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(in_features=2048, out_features=anchor_shape + num_targets)
        self.model.conv1 = nn.Conv2d(
            in_channels=self.num_input_channels,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
            
        #self.mlp_head = nn.Linear(2*ego_input_channels, anchor_shape + num_targets)
        self.mlp_head = nn.Sequential(
            nn.Linear(2*ego_input_channels, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, anchor_shape + num_targets),
            )

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = len(data_batch["box_img"])
        # [batch_size, channels, height, width]
        image_batch = torch.cat((data_batch["box_img"],data_batch["way_img"],data_batch["rel_img"]), dim = -1)
        image_batch = image_batch.permute(0,3,1,2)
        
        # [batch_size, num_steps * 3]
        x = self.model(image_batch)
        y = data_batch["ego_features"].view(batch_size,-1)
        outputs = x + self.lmbda_ego * self.mlp_head(y)
        #print(outputs)

        if self.training:
            # +anchor真值轨迹
            outputs[:,self.anchor_shape::3] = data_batch["anchor_ref"][:,::2] + 0.1* outputs[:,self.anchor_shape::3]
            outputs[:,self.anchor_shape+1::3] = data_batch["anchor_ref"][:,1::2] + 0.1* outputs[:,self.anchor_shape+1::3]
            outputs[:,self.anchor_shape+2] = torch.atan2(data_batch["anchor_ref"][:,1],data_batch["anchor_ref"][:,0]) + 0.1*outputs[:,self.anchor_shape+2]
            outputs[:,self.anchor_shape+5::3] = torch.atan2(data_batch["anchor_ref"][:,3::2]-data_batch["anchor_ref"][:,1:-1:2]
                    ,data_batch["anchor_ref"][:,2::2]-data_batch["anchor_ref"][:,0:-2:2]) + 0.1*outputs[:,self.anchor_shape+5::3]
            #print(outputs)
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
            # [batch_size, num_steps * 3]
            targets = data_batch['ego_targets'].view(batch_size, -1)
            target_weights = (data_batch['ego_availabilities'].unsqueeze(-1) * self.weights_scaling).view(batch_size, -1)
            #payoff of probability
            loss_prob = nn.functional.cross_entropy(outputs[:,:self.anchor_shape],data_batch["anchor_id"])
            #payoff of last frame
            #target_weights[:,-3:] *= self.lmbda
            #payoff of IOU
            predicted = outputs.view(batch_size, -1, 3)
            #predicted = data_batch['ego_targets']
            length, width = data_batch['ego_extent'][:,0].unsqueeze(-1)*1.01, data_batch['ego_extent'][:,1].unsqueeze(-1)*1.01
            lowleft = torch.concat((-length/2, -width/2), dim=-1).unsqueeze(1)
            lowright = torch.concat((length/2, -width/2), dim=-1).unsqueeze(1)
            upright = torch.concat((length/2, width/2), dim=-1).unsqueeze(1)
            upleft = torch.concat((-length/2, width/2), dim=-1).unsqueeze(1)
            box = torch.concat((lowleft, lowright, upright, upleft), dim = 1).unsqueeze(1).repeat(1,self.num_targets //3,1,1)
            rot_1 = torch.concat((torch.cos(predicted[:,:,2:3]), torch.sin(predicted[:,:,2:3])), dim=-1).unsqueeze(-2)
            rot_2 = torch.concat((-torch.sin(predicted[:,:,2:3]), torch.cos(predicted[:,:,2:3])), dim=-1).unsqueeze(-2)
            rot = torch.concat((rot_1, rot_2), dim = -2)
            #bxTx4x2
            future_box = torch.matmul(box,rot) + predicted[:,:,:2].unsqueeze(-2)
            future_box[...,0] = future_box[...,0]/self.pixel_size[0] + self.ego_center[0]*self.raster_size[0]
            future_box[...,1] = -future_box[...,1]/self.pixel_size[1] + (1 - self.ego_center[1])*self.raster_size[1]
            loss_cap = 0
            for i,img in enumerate(data_batch["rel_img"]):
                img = img.detach().numpy()
                area_1 = img.sum()
                cv2.fillPoly(img, (future_box[i].detach().numpy()* CV2_SHIFT_VALUE).astype(np.int32), color=1, **CV2_SUB_VALUES)
                area_2 = img.sum()
                #plt.imshow(img)
                #plt.show()
                loss_cap += area_2 - area_1 
            
            loss_traj = torch.sum(self.criterion(outputs[:,self.anchor_shape:], targets)* target_weights/2, dim = -1)
            loss_traj = torch.mean(loss_traj)
            loss = loss_prob + loss_traj + self.lmbda_occ * loss_cap/batch_size
            train_dict = {"loss": loss}
            #print(targets)
            return train_dict
        else:
            prob = outputs[:,:self.anchor_shape]
            #print(outputs)
            predicted = outputs[:,self.anchor_shape:].view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = predicted[:, :, 2:3]
            eval_dict = {"prob":prob,"positions": pred_positions, "yaws": pred_yaws}
            return eval_dict
        
