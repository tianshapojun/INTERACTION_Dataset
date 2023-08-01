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


from utils import map_vis_without_lanelet
from utils import dataset_reader
from utils import vectorizer
from utils import map_api
from utils import visual
from utils import model_raster
from utils.model_vector import VectorizedUnrollModel

N_past_track = 10
N_future_track = 5

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
    return box_img.astype(np.float32)/255, map_img_1.astype(np.float32)/255, map_img_2.astype(np.float32)/255, ego_features

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
    
    def __init__(self, config, agent_sample, track_df, map):
        super().__init__()
        self.config = config
        self.agent_sample = agent_sample
        self.track_df = track_df
        self.map = map
    
    def __getitem__(self, index):
        case_id,track_id,frame_id = self.agent_sample[index]['case_id'], self.agent_sample[index]['track_id'], self.agent_sample[index]['frame_id']
        track_dictionary = self.track_df[(self.track_df['case_id'] == case_id)  & (self.track_df['frame_id'] >= frame_id - self.config["history_num_frames"])
            & (self.track_df['frame_id'] <= frame_id + self.config["future_num_frames"])]
        box_img, map_img_1, map_img_2, ego_features = rasterize(track_dictionary, self.map, frame_id, self.config["history_num_frames"], self.config["history_num_ego_frames"], self.config["future_num_frames"]
            , track_id, self.config["raster_size"], self.config["pixel_size"], self.config["ego_center"])
        label = np.concatenate((ego_features["target_positions"], ego_features["target_yaws"]), axis = -1)
        return {"box_img":box_img, "way_img":map_img_1, "rel_img":map_img_2
                , "ego_targets":label, "ego_availabilities":ego_features["target_availabilities"]}
    
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
        case_id,track_id,frame_id = self.agent_sample[index]['case_id'], self.agent_sample[index]['track_id'], self.agent_sample[index]['frame_id']
        #print(case_id,track_id,frame_id)
        id_track = track_df[track_df['case_id'] == case_id]
        history_track = id_track[(id_track['frame_id'] >= frame_id - self.config["history_num_frames"]) & (id_track['frame_id'] <= frame_id)]
        current_track = id_track[id_track['frame_id'] == frame_id]
        future_track = id_track[(id_track['frame_id'] >= frame_id + 1) & (id_track['frame_id'] <= frame_id + self.config["future_num_frames"])]
        return vectorize(config, map, track_id, frame_id, history_track, current_track, future_track)
    
    def __len__(self):
        return len(agent_sample)

def train(config, dataset, model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if model_name == "ResNet":
        model = model_raster.RasterizedPlanningModel(
            model_arch = config["model_architecture"],
            num_input_channels = config['history_num_frames']+1+3+1,
            num_targets=3 * config["future_num_frames"],  # X, Y, Yaw * number of future states,
            weights_scaling= [1., 1., 1.],
            criterion=nn.MSELoss(reduction="none"),
            )
    elif model_name == "SafetyNet":
        weights_scaling = [1.0, 1.0, 1.0]
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
            discount_factor=config["discount_factor"]
        )

    else:
        raise IOError("You must specify a model.")
    
    model = model.to(device)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    
    # Model Output Checking
    if False:
        model.load_state_dict(torch.load("D:/Code/INTERACTION_Dataset/model/2023-07-28_00.pth",map_location=torch.device('cpu')))
        model.eval()
        data = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in dataset[0].items()}
        print(dataset[0]['ego_targets'])
        out = model(data)
        #for key,value in out.items():
            #print(key,value)
        hhh
    
    # Prepare for Training Rasterization Data 
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size = config["batch_size"], 
                             num_workers = config["num_workers"])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(config["base_dir"]+'cache')
    
    iter = 0
    for epoch in range(config["epoch"]):
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
        if config["model_save"] and epoch % 2 ==0:
            model_path = config["base_dir"] + 'model/' + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M') + '_' +'{:02}.pth'.format(epoch)
            os.makedirs(config["base_dir"] +'model', exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print('Saved checkpoints at', model_path)
    writer.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_past_track", type=int, default = 10, nargs="?")
    parser.add_argument("--N_future_track", type=int, default = 1, nargs="?")
    parser.add_argument("--config_filename", type=str, default = "D:/Code/INTERACTION_Dataset/config.json")
    args = parser.parse_args()
    
    with open(args.config_filename, 'r') as f:
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
    lanelet_map_file = os.path.join(maps_dir, config["scenario_name"] + lanelet_map_ending)

    scenario_dir = os.path.join(tracks_dir, config["scenario_name"])

    track_file_name = os.path.join(
        scenario_dir,
        "vehicle_tracks_" + str(config["track_file_number"]).zfill(3) + ".csv"
    )
    
    # Construct map
    map = map_api.map_api(config, lanelet_map_file, lat_origin, lon_origin)
    # Construct samples
    print("Constructing samples...")
    track_df = pd.read_csv(track_file_name)
    track_df = track_df[track_df['agent_type'] == 'car']   
    print('Total:{} scenes'.format(len(list(track_df['case_id'].unique()))))
    agent_sample = []
    #for i in list(track_df['case_id'].unique()):
    for i in range(10):
        temp = track_df[track_df['case_id'] == i]
        for j in list(temp['track_id'].unique()):
            agent_sample += [ {'case_id':i,'track_id':j,'frame_id':k} for k in 
                 is_satisfy(list(temp[temp['track_id'] == j]['frame_id'].unique()), N_past_track, N_future_track, config['frame_interval'])]
    print('Total:{} samples'.format(len(agent_sample)))
    print("Constructing RasterDataset...")
    raster_dataset = RasterDataset(config, agent_sample, track_df, map)
    assert config["history_num_frames"] <= N_past_track
    assert config["history_num_ego_frames"] <= config["history_num_frames"]
    assert config["future_num_frames"] <= N_future_track
    vector_dataset = VectorDataset(config, agent_sample, track_df, map)
    '''
    temp = vector_dataset[25]
    fig = plt.figure()
    for i in range(config["max_num_lanes"]):
        #plt.subplot(5,2,i+1)
        plt.plot(temp['lanes'][2*i][:,0],temp['lanes'][2*i][:,1],'b')
        plt.plot(temp['lanes'][2*i+1][:,0],temp['lanes'][2*i+1][:,1],'b')
        plt.plot(temp['lanes_mid'][i][:,0],temp['lanes_mid'][i][:,1],'r--')
    plt.show()
    hhh
    '''
    
    ### One Sample
    ###
    ###
    '''
    i = 5
    case_id,track_id,frame_id = agent_sample[i]['case_id'],agent_sample[i]['track_id'],agent_sample[i]['frame_id']
    temp = track_df[(track_df['case_id'] == case_id) & (track_df['track_id'] == track_id) & (track_df['frame_id'] == frame_id)]
    print(temp['x'].iloc[0],temp['y'].iloc[0],temp['psi_rad'].iloc[0])
    print(case_id,track_id,frame_id)
    '''
    
    # load the tracks
    # plot by matplotlib
    '''
    print("Loading tracks using matplotlib...")
    track_dictionary = dataset_reader.read_tracks(track_file_name,case_id)
    plot_agent(frame_id,track_dictionary,lanelet_map_file,lat_origin,lon_origin,temp['x'].iloc[0],temp['y'].iloc[0],temp['psi_rad'].iloc[0])
    plt.show()
    '''
    
    # Rasterization Features
    '''
    i=5
    box_img = raster_dataset[i]["box_img"]
    map_img_1 = raster_dataset[i]["way_img"]
    map_img_2 = raster_dataset[i]["rel_img"]
    ego_targets = raster_dataset[i]['ego_targets']
    '''
    
    # Visualizaiton
    #visual.raster_visual_fig(config,box_img,map_img_1,map_img_2) 
    # Animation of Rasterization Images
    #visual.raster_visual_anim(config["history_num_frames"]+1, box_img, map_img_1, save_flg=False) 
    
    #'''
    #train(config, raster_dataset, "ResNet")
    train(config, vector_dataset, "SafetyNet")
