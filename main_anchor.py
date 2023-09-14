#!/usr/bin/python3.10  
# -*- coding: utf-8 -*- 

import argparse
import os
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch import optim
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import datetime 
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import map_vis_without_lanelet
from utils import dataset_reader
from utils import vectorizer
from utils import map_api
from utils import visual
from utils import multi_model_raster
from utils.model_vector import VectorizedUnrollModel
from utils import model_raster_detr
from utils import model_raster_vit
from utils import model_raster_pre

N_past_track = 10
N_future_track = 10

#config_filename = "./config1.json"
parser = argparse.ArgumentParser()
parser.add_argument("--config_filename", type=str, default = "./config1.json")
args = parser.parse_args()
config_filename = args.config_filename

def is_satisfy(frame_list,past_num,future_num,frame_interval):
    temp_list = []
    frame_list.sort()
    for i in frame_list:
        temp = [j for j in range(i - past_num ,i + future_num + 1)]
        if set(temp) <= set(frame_list):
            if len(temp_list) > 0 and i < temp_list[-1] + frame_interval:
                continue
            temp_list.append(i)
    return temp_list

def plot_agent(frame_id,track_dictionary,lanelet_map_file,lat_origin,lon_origin,x,y,psd):
    fig, axes = plt.subplots(1, 1)
    #fig.canvas.FigureManagerBase.set_window_title("Interaction Dataset Visualization")
    fig.canvas.manager.set_window_title("Interaction Dataset Visualization")
    print("Loading map...")
    map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin,x,y,psd)
    map_vis_without_lanelet.update_objects_plot(frame_id,axes, x, y, psd, track_dictionary)

def rasterize(track_dictionary, map, frame_id, num_past, num_ego_past, num_future, track_id, raster_size, pixel_size, ego_center):
    history_track = track_dictionary[track_dictionary['frame_id'] <= frame_id]
    current_track = track_dictionary[track_dictionary['frame_id'] == frame_id]
    future_track = track_dictionary[track_dictionary['frame_id'] >= frame_id + 1]
    temp = current_track[current_track['track_id'] == track_id]
    x,y,psd = temp['x'].iloc[0],temp['y'].iloc[0],temp['psi_rad'].iloc[0]
    box_img = np.zeros((raster_size[1], raster_size[0], num_past + 1))
    for i in range(num_past + 1):
        tmp_dict = track_dictionary[track_dictionary['frame_id'] == frame_id - num_past + i]
        box_img[...,i] = map_vis_without_lanelet.box_raster(np.array(tmp_dict[['x','y','psi_rad','width','length']])
            , x, y, psd, raster_size, pixel_size, ego_center)
    map_img_1, map_img_2 = map_vis_without_lanelet.map_raster(map, x, y, psd, raster_size, pixel_size, ego_center)
    ego_features = vectorizer.vectorize_ego(track_id, num_ego_past, history_track,current_track,future_track)
    inact_features = vectorizer.vectorize_inact(map, history_track, num_ego_past,track_id, frame_id)
    return box_img.astype(np.float32)/255, map_img_1.astype(np.float32)/255, map_img_2.astype(np.float32)/255, ego_features, inact_features
    #return box_img.astype(np.float32)/255, map_img_1.astype(np.float32)/255, map_img_2.astype(np.float32)/255, ego_features

def vectorize(config, map, track_id, frame_id, history_track, current_track, future_track):
    ego_features = vectorizer.vectorize_ego(track_id, config["history_num_ego_frames"], history_track,current_track,future_track)
    agent_features = vectorizer.vectorize_agents(track_id, frame_id, ego_features["centroid"], ego_features["yaw"], ego_features["history_positions"]
                    , ego_features["history_yaws"], history_track, current_track, future_track
                    , config["max_agents_distance"], config["other_agents_num"], config["history_num_frames"], config["future_num_frames"])
    
    map_features = vectorizer.vectorize_map(config, map, ego_features["centroid"], ego_features["yaw"])
    #print(agent_features["all_other_agents_types"])
    #print(agent_features["all_other_agents_future_yaws"])
    return {**ego_features, **agent_features, **map_features}
    #return map_features

class RasterDataset(Dataset):
    
    def __init__(self, config, agent_sample, track_df, map, anchor):
        super().__init__()
        self.config = config
        self.agent_sample = agent_sample
        self.track_df = track_df
        self.map = map
        self.anchor = anchor
    
    def __getitem__(self, index):
        scenario,case_id,track_id,frame_id = self.agent_sample[index]['scenario'], self.agent_sample[index]['case_id'], self.agent_sample[index]['track_id'], self.agent_sample[index]['frame_id']
        track_dictionary = self.track_df[(self.track_df['case_id'] == case_id) & (self.track_df['scenario'] == scenario) 
            & (self.track_df['frame_id'] >= frame_id - self.config["history_num_frames"])
            & (self.track_df['frame_id'] <= frame_id + self.config["future_num_frames"])]
        box_img, map_img_1, map_img_2, ego_features, inact_features = rasterize(track_dictionary, self.map[scenario], frame_id, self.config["history_num_frames"], self.config["history_num_ego_frames"], self.config["future_num_frames"]
            , track_id, self.config["raster_size"], self.config["pixel_size"], self.config["ego_center"])
        label = np.concatenate((ego_features["target_positions"], ego_features["target_yaws"]), axis = -1)
        target = ego_features["target_positions"][:,0:2].flatten()
        #anchor_n = self.anchor.reshape(-1,self.config["future_num_frames"],3)
        #anchor_n = anchor_n[:,:,0:2].reshape(-1,2*self.config["future_num_frames"])
        dist = np.linalg.norm(self.anchor - target, axis=-1)
        ego_f = np.concatenate((ego_features["history_speed"][:self.config["history_num_ego_frames"]+1]
            , ego_features["history_yaws"][:self.config["history_num_ego_frames"]+1,None],inact_features["mid_feat"]), axis = -1)
        #ego_f = inact_features["mid_feat"]
        return {"box_img":box_img, "way_img":map_img_1, "rel_img":map_img_2
                , "ego_targets":label, "ego_availabilities":ego_features["target_availabilities"]
                ,"ego_extent":ego_features["extent"], "ego_features":ego_f
                , "anchor_id": np.argsort(dist)[0], "anchor_ref": self.anchor[np.argsort(dist)[0]]}
    
    def __len__(self):
        return len(agent_sample)

class VectorDataset(Dataset):
    
    def __init__(self, config, agent_sample, track_df, map):
        super().__init__()
        self.config = config
        self.agent_sample = agent_sample
        self.track_df = track_df
        self.map = map
    
    def __getitem__(self, index):
        scenario,case_id,track_id,frame_id = self.agent_sample[index]['scenario'], self.agent_sample[index]['case_id'], self.agent_sample[index]['track_id'], self.agent_sample[index]['frame_id']
        #print(case_id,track_id,frame_id)
        id_track = track_df[(track_df['scenario'] == scenario) & (track_df['case_id'] == case_id)]
        history_track = id_track[(id_track['frame_id'] >= frame_id - self.config["history_num_frames"]) & (id_track['frame_id'] <= frame_id)]
        current_track = id_track[id_track['frame_id'] == frame_id]
        future_track = id_track[(id_track['frame_id'] >= frame_id + 1) & (id_track['frame_id'] <= frame_id + self.config["future_num_frames"])]
        return vectorize(config, map[scenario], track_id, frame_id, history_track, current_track, future_track)
    
    def __len__(self):
        return len(agent_sample)

def train(config, dataset, model_name, anchor):
    #dist.init_process_group(backend='nccl') 
    #local_rank = int(os.environ["LOCAL_RANK"])
    #torch.cuda.set_device(local_rank)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device(config["device"])
    print(device)
    if model_name == "resnet":
        model = multi_model_raster.RasterizedPlanningModel(
            model_arch = config["model_architecture"],
            num_input_channels = config['history_num_frames']+1+3+1,
            ego_input_channels = config['history_num_ego_frames'] + 1,
            num_targets=3 * config["future_num_frames"],  # X, Y, Yaw * number of future states,
            weights_scaling= config["weights_scaling"],
            criterion=nn.MSELoss(reduction="none"),
            lmbda = config["lmbda"],
            lmbda_occ = config["lmbda_occ"],
            lmbda_ego = config["lmbda_ego"],
            raster_size = config["raster_size"],
            pixel_size = config["pixel_size"],
            ego_center = config["ego_center"],
            anchor_shape = anchor.shape[0]
            )
    elif model_name == "vit":
        model = model_raster_vit.RasterizedPlanningModel(
            model_arch = config["model_architecture"],
            num_input_channels = config['history_num_frames']+1+3+1,
            ego_input_channels = config['history_num_ego_frames'] + 1,
            num_targets=3 * config["future_num_frames"],  # X, Y, Yaw * number of future states,
            weights_scaling= config["weights_scaling"],
            criterion=nn.MSELoss(reduction="none"),
            lmbda = config["lmbda"],
            lmbda_occ = config["lmbda_occ"],
            lmbda_ego = config["lmbda_ego"],
            raster_size = config["raster_size"],
            pixel_size = config["pixel_size"],
            ego_center = config["ego_center"],
            )
    elif model_name == "resnet_detr":
        model = model_raster_detr.RasterizedPlanningModel(
            model_arch = config["model_architecture"],
            num_input_channels = config['history_num_frames']+1+3+1,
            num_targets=3 * config["future_num_frames"],  # X, Y, Yaw * number of future states,
            weights_scaling= config["weights_scaling"],
            criterion=nn.MSELoss(reduction="none"),
            lmbda = config["lmbda"],
            raster_size = config["raster_size"],
            pixel_size = config["pixel_size"],
            ego_center = config["ego_center"]
            )
    elif model_name == "safetynet":
        weights_scaling = config["weights_scaling"]
        _num_predicted_frames = config["future_num_frames"]
        _num_predicted_params = len(weights_scaling)
        model = VectorizedUnrollModel(
            history_num_frames_ego=config["history_num_ego_frames"],
            history_num_frames_agents=config["history_num_frames"],
            num_targets=_num_predicted_params * _num_predicted_frames,
            weights_scaling=weights_scaling,
            criterion=nn.L1Loss(reduction="none"),
            global_head_dropout=config["global_head_dropout"],
            disable_other_agents=config["disable_other_agents"],
            disable_map=config["disable_map"],
            disable_lane_boundaries=config["disable_lane_boundaries"],
            detach_unroll=config["detach_unroll"],
            warmup_num_frames=config["warmup_num_frames"],
            discount_factor=config["discount_factor"],
            lmbda = config["lmbda"]
        )

    else:
        raise IOError("You must specify a model.")
    
    model = model.to(device)
    #model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    
    # Model Output Checking
    if False:
        model.load_state_dict(torch.load("./model/2023-08-02-07:43_08.pth"))
        model.eval()
        tt = 2
        data = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in dataset[tt].items()}
        #print(dataset[tt]['target_positions'],dataset[tt]['target_yaws'])
        out = model(data)
        for key,value in out.items():
            print(key,value)
        hhh
    
    # Prepare for Training Rasterization Data
    #train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #train_dataloader = DataLoader(dataset, batch_size = config["batch_size"], 
                             #num_workers = config["num_workers"], sampler=train_sampler)
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size = config["batch_size"], 
                             num_workers = config["num_workers"])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #os.makedirs(config["base_dir"]+'cache/{}_'.format(config_filename[2:-5])+ datetime.datetime.now().strftime('%Y-%m-%d-%H'), exist_ok=True)
    writer = SummaryWriter(config["base_dir"]+'cache/mul_vit_cmb_{}_'.format(config_filename[2:-5])+datetime.datetime.now().strftime('%Y-%m-%d-%H'))
    #writer = SummaryWriter(config["base_dir"]+'cache/mul_vit_cmb_config1_2023-08-29-07')
    
    if False:
        model.load_state_dict(torch.load("./model/2023-08-04-09:57_00.pth"))
        model.eval()
        for data in train_dataloader:
            data = {k: v.to(device) for k, v in data.items()}
            out = model(data)
            print(out['positions'][:3])
            print(data["target_positions"][:3])
            hhh
    
    iter = 0
    for epoch in range(config["epoch"]):
        #train_dataloader.sampler.set_epoch(epoch)
        iterator = tqdm(train_dataloader)
        model.train()
        for data in iterator:
            data = {k: v.to(device) for k, v in data.items()}
            result = model(data)
            loss = result["loss"]
            iterator.desc = "loss = %0.3f" % loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            iter +=1
            writer.add_scalar("TotalLoss", loss.item(), iter)
        if config["model_save"] and epoch %3 == 0: 
            model_path = config["base_dir"] + 'model/{}/mul_vit_cmb_{}_'.format(datetime.datetime.now().strftime('%Y-%m-%d'),config_filename[2:-5])
            model_path += datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' +'{:02}.pth'.format(epoch)
            os.makedirs(config["base_dir"] +'model/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d')), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print('Saved checkpoints at', model_path)
    writer.close()

if __name__ == "__main__":
    
    with open(config_filename, 'r') as f:
        config = json.loads(f.read())

    if config["scenario_name"] is None:
        raise IOError("You must specify a scenario. Type --help for help.")
    
    # origin is necessary to correctly project the lat lon values of the map to the local
    # coordinates in which the tracks are provided; defaulting to (0|0) for every scenario
    lat_origin, lon_origin = config["lat_origin"], config["lon_origin"]
    
    # check folders and files
    error_string = ""

    # root directory is one above main_visualize_data.py file
    # i.e. the root directory of this project
    root_dir = os.path.dirname(os.path.abspath(__file__))

    tracks_dir = os.path.join(root_dir, "recorded_trackfiles")
    maps_dir = os.path.join(root_dir, "maps")

    lanelet_map_ending = ".osm"
    map = dict()
    
    for sc_name in config["scenario_name"]:
        print("Loading {}".format(sc_name))
        lanelet_map_file = os.path.join(maps_dir, sc_name + lanelet_map_ending)

        scenario_dir = os.path.join(tracks_dir, sc_name)

        track_file_name = os.path.join(
            scenario_dir,
            "vehicle_tracks_" + str(config["track_file_number"]).zfill(3) + ".csv"
        )
    
        # Construct map
        map[sc_name] = map_api.map_api(config, lanelet_map_file, lat_origin, lon_origin)
        # Construct samples
        print("Constructing samples...")
        track_df = pd.read_csv(track_file_name)
        track_df = track_df[track_df['agent_type'] == 'car'] 
        track_df['scenario'] = sc_name
        try:
            track_all = pd.concat([track_all,track_df],axis=0)  
        except: 
            track_all = track_df
    for sc_name in config["scenario_name"]:
        print('Scenario:{}, Total:{} scenes'.format(sc_name,len(list(track_all[track_all["scenario"]==sc_name]['case_id'].unique()))))
    agent_sample = []
    for sc_name in config["scenario_name"]:
        track_df = track_all[track_all['scenario'] == sc_name]
        for i in list(track_df['case_id'].unique()):
            #if i>1700:
                #break
        #for i in range(10):
            temp = track_df[track_df['case_id'] == i]
            for j in list(temp['track_id'].unique()):
                if j <= 5 or sc_name == "DR_USA_Intersection_EP0":
                    agent_sample += [ {'scenario':sc_name,'case_id':i,'track_id':j,'frame_id':k} for k in 
                        is_satisfy(list(temp[temp['track_id'] == j]['frame_id'].unique()), N_past_track, N_future_track, config['frame_interval'])]
                else:
                    break
    print('Total:{} samples'.format(len(agent_sample)))
    
    with open(config["anchor_name"], "rb") as f:
        anchor = pickle.load(f)
    print("Constructing RasterDataset...")
    raster_dataset = RasterDataset(config, agent_sample, track_all, map, anchor)
    assert config["history_num_frames"] <= N_past_track
    assert config["history_num_ego_frames"] <= config["history_num_frames"]
    assert config["future_num_frames"] <= N_future_track
    vector_dataset = VectorDataset(config, agent_sample, track_all, map)
    
    #print(raster_dataset[0]["ego_features"].shape)
    #print(raster_dataset[0]["box_img"].shape)
    #hhh
    
    if config["model_name"] == "safetynet":
        train(config, vector_dataset, config["model_name"], anchor)
    else: 
        train(config, raster_dataset, config["model_name"], anchor)
