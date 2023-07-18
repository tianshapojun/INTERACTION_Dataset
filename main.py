#!/usr/bin/python3.10  
# -*- coding: utf-8 -*- 
#source D:/Software/Anaconda/etc/profile.d/conda.sh
#cd /d/Code/Interaction_Dataset
#python D:/Code/INTERACTION_Dataset/main.py


import argparse
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import json
from utils import map_vis_without_lanelet
from utils import dataset_reader

N_past_track = 10
N_future_track = 1

def is_satisfy(frame_list,past_num,future_num):
    temp_list = []
    for i in frame_list:
        temp = [j for j in range(i - past_num,i + future_num + 1)]
        if set(temp) < set(frame_list):
            temp_list.append(i)
    return temp_list

def plot_agent(frame_id,track_dictionary,lanelet_map_file,lat_origin,lon_origin,x,y,psd):
    fig, axes = plt.subplots(1, 1)
    #fig.canvas.FigureManagerBase.set_window_title("Interaction Dataset Visualization")
    fig.canvas.manager.set_window_title("Interaction Dataset Visualization")
    print("Loading map...")
    map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin,x,y,psd)
    map_vis_without_lanelet.update_objects_plot(frame_id,axes, x, y, psd, track_dictionary)

def plot_raster(track_dictionary,lanelet_map_file, lat_origin, lon_origin, x, y, psd, raster_size, pixel_size, ego_center):
    #map_vis_without_lanelet.box_raster(track_dictionary, x, y, psd, raster_size, pixel_size, ego_center)
    map_vis_without_lanelet.map_raster(lanelet_map_file, lat_origin, lon_origin, x, y, psd, raster_size, pixel_size, ego_center)

if __name__ == "__main__":

    # provide data to be visualized
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_past_track", type=int, default = 10, nargs="?")
    parser.add_argument("--N_future_track", type=int, default = 1, nargs="?")
    parser.add_argument("--config_filename", type=str, default = "D:/Code/INTERACTION_Dataset/config.json")
    args = parser.parse_args()
    
    with open(args.config_filename, 'r') as f:
        config = json.loads(f.read())

    if config["scenario_name"] is None:
        raise IOError("You must specify a scenario. Type --help for help.")
    if config["load_mode"] != "vehicle" and config["load_mode"] != "pedestrian" and config["load_mode"] != "both":
        raise IOError("Invalid load command. Use 'vehicle', 'pedestrian', or 'both'")
    
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
    pedestrian_file_name = os.path.join(
        scenario_dir,
        "pedestrian_tracks_" + str(config["track_file_number"]).zfill(3) + ".csv"
    )

    if not os.path.isdir(tracks_dir):
        error_string += "Did not find track file directory \"" + tracks_dir + "\"\n"
    if not os.path.isdir(maps_dir):
        error_string += "Did not find map file directory \"" + tracks_dir + "\"\n"
    if not os.path.isdir(scenario_dir):
        error_string += "Did not find scenario directory \"" + scenario_dir + "\"\n"
    if not os.path.isfile(lanelet_map_file):
        error_string += "Did not find lanelet map file \"" + lanelet_map_file + "\"\n"
    if not os.path.isfile(track_file_name):
        error_string += "Did not find track file \"" + track_file_name + "\"\n"
    if not os.path.isfile(pedestrian_file_name):
        flag_ped = 0
    else:
        flag_ped = 1
    if error_string != "":
        error_string += "Type --help for help."
        raise IOError(error_string)


    # Construct samples
    print("Constructing samples...")
    if config["load_mode"] == 'both':
        track_df = pd.read_csv(track_file_name)
        if flag_ped:
            pedestrian_df = pd.read_csv(pedestrian_file_name)

    elif config["load_mode"] == 'vehicle':
        track_df = pd.read_csv(track_file_name)
    elif config["load_mode"] == 'pedestrian':
        pedestrian_df = pd.read_csv(pedestrian_file_name)
        
    print('Total:{} scenes'.format(len(list(track_df['case_id'].unique()))))
    agent_sample = []
    #for i in list(track_df['case_id'].unique()):
    for i in range(3):
        temp = track_df[(track_df['case_id'] == i) & (track_df['agent_type'] == 'car')]
        for j in list(temp['track_id'].unique()):
            agent_sample += [ {'case_id':i,'track_id':j,'frame_id':k} for k in is_satisfy(list(temp[temp['track_id'] == j]['frame_id'].unique()), N_past_track, N_future_track)]
    print('Total:{} samples'.format(len(agent_sample)))
    
    
    i = 34
    case_id,track_id,frame_id = agent_sample[i]['case_id'],agent_sample[i]['track_id'],agent_sample[i]['frame_id']
    temp = track_df[(track_df['case_id'] == case_id) & (track_df['track_id'] == track_id) & (track_df['frame_id'] == frame_id)]
    print(temp['x'].iloc[0],temp['y'].iloc[0],temp['psi_rad'].iloc[0])
    print(case_id,track_id,frame_id)
    
    # load the tracks
    '''
    print("Loading tracks using matplotlib...")
    track_dictionary = dataset_reader.read_tracks(track_file_name,case_id)
    plot_agent(frame_id,track_dictionary,lanelet_map_file,lat_origin,lon_origin,temp['x'].iloc[0],temp['y'].iloc[0],temp['psi_rad'].iloc[0])
    plt.show()
    '''
    
    
    track_dictionary = track_df[(track_df['case_id'] == case_id) & (track_df['agent_type'] == 'car') & (track_df['frame_id'] == frame_id)]
    plot_raster(track_dictionary,lanelet_map_file, lat_origin, lon_origin, temp['x'].iloc[0], temp['y'].iloc[0], temp['psi_rad'].iloc[0], config["raster_size"], config["pixel_size"], config["ego_center"])
