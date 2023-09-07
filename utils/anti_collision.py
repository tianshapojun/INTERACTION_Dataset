import numpy as np
import torch
import torch.nn as nn

def way_points(map, ego_centroid):
    distances = []
    for key,way in map.way_dict.items():
        if way["type"] not in [1,9]:
            continue
        lane_poly = map.interpolate(way["position"][:, :2], 200, "INTER_LEN") 
        distances.append(lane_poly)
    return np.array(distances)
    


def way_info(config, map, ego_centroid):
    # START WORKING ON LANES
        MAX_LANE_DISTANCE = 1
        lanes_indices = []
        distances = []
        slopes = []
        
        for key,way in map.way_dict.items():
            if way["type"] not in [1,9]:
                continue
            #lane_poly = map.interpolate(way["position"][:, :2], 1, "INTER_METER") 
            lane_poly = map.interpolate(way["position"][:, :2], 200, "INTER_LEN") 
            #if lane_poly.shape[0]<= 1:
                #lane_poly = map.interpolate(way["position"][:, :2], 2, "INTER_LEN") 
            lane_dist_all = np.zeros((lane_poly.shape[0],ego_centroid.shape[0]),dtype = np.float32)
            for idx,center in enumerate(ego_centroid):
                lane_dist_all[:,idx] = np.linalg.norm(lane_poly - center, axis=-1)
            lane_dist = np.min(lane_dist_all,axis=-1)
            lane_k = [[lane_poly[i+1,0] - lane_poly[i,0], lane_poly[i+1,1] - lane_poly[i,1]] for i in range(len(lane_poly)-1)]
            lane_k.append([lane_poly[-1,0] - lane_poly[-2,0], lane_poly[-1,1] - lane_poly[-2,1]])
            lanes_indices.append(key)
            distances.append(np.min(lane_dist))
            slopes.append(lane_k[np.argsort(lane_dist)[0]])
        
        if len(distances) ==0: #or min(distances) > MAX_LANE_DISTANCE:
            return {}
        else:
            return {"id":lanes_indices[np.argsort(distances)[0]], "distance":min(distances), "k":slopes[np.argsort(distances)[0]]}
        
class traj_modif(nn.Module):

    def __init__(self,shape,lmbda):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(shape,2))
        self.lmbda = lmbda

    def forward(self,traj,border):
        #traj,N*2
        #borer,M*200*2
        traj_n = traj + self.delta
        out = torch.zeros(traj.shape[0])
        peng_loss = 0
        for i in range(traj.shape[0]):
            temp = border - traj_n[i]
            temp = torch.norm(temp, dim=-1)
            temp = torch.min(temp)
            out[i] = temp
        peng_loss += torch.exp(-torch.min(out))
        l2_loss = torch.norm(self.delta) 
        loss = l2_loss + self.lmbda * peng_loss
        return traj_n,loss 
            
        
