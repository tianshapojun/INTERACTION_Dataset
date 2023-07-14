import argparse
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from utils import map_vis_without_lanelet

N_past_track = 10
N_future_track = 1

def is_satisfy(frame_list,past_num,future_num):
    temp_list = []
    for i in frame_list:
        temp = [j for j in range(i - past_num,i + future_num + 1)]
        if set(temp) < set(frame_list):
            temp_list.append(i)
    return temp_list

def plot_agent(lanelet_map_file,lat_origin,lon_origin):
    fig, axes = plt.subplots(1, 1)
    fig.canvas.set_window_title("Interaction Dataset Visualization")
    print("Loading map...")
    map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin)


if __name__ == "__main__":

    # provide data to be visualized
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
                                                        "files)", nargs="?")
    parser.add_argument("track_file_number", type=int, help="Number of the track file (int)", default=0, nargs="?")
    parser.add_argument("load_mode", type=str, help="Dataset to load (vehicle, pedestrian, or both)", default="vehicle",
                        nargs="?")
    parser.add_argument("--start_timestamp", type=int, nargs="?")
    parser.add_argument("--lat_origin", type=float,
                        help="Latitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    parser.add_argument("--lon_origin", type=float,
                        help="Longitude of the reference point for the projection of the lanelet map (float)",
                        default=0.0, nargs="?")
    args = parser.parse_args()
    #DR_CHN_Roundabout_LN,.TestScenarioForScripts,DR_CHN_Merging_ZS0,DR_USA_Intersection_EP0
    args.scenario_name = 'DR_USA_Intersection_EP0'
    N_past_track = 10
    N_future_track = 1

    if args.scenario_name is None:
        raise IOError("You must specify a scenario. Type --help for help.")
    if args.load_mode != "vehicle" and args.load_mode != "pedestrian" and args.load_mode != "both":
        raise IOError("Invalid load command. Use 'vehicle', 'pedestrian', or 'both'")

    # check folders and files
    error_string = ""

    # root directory is one above main_visualize_data.py file
    # i.e. the root directory of this project
    root_dir = os.path.dirname(os.path.abspath(__file__))

    tracks_dir = os.path.join(root_dir, "recorded_trackfiles")
    maps_dir = os.path.join(root_dir, "maps")

    lanelet_map_ending = ".osm"
    lanelet_map_file = os.path.join(maps_dir, args.scenario_name + lanelet_map_ending)

    scenario_dir = os.path.join(tracks_dir, args.scenario_name)

    track_file_name = os.path.join(
        scenario_dir,
        "vehicle_tracks_" + str(args.track_file_number).zfill(3) + ".csv"
    )
    pedestrian_file_name = os.path.join(
        scenario_dir,
        "pedestrian_tracks_" + str(args.track_file_number).zfill(3) + ".csv"
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

    # load the tracks
    print("Loading tracks...")
    if args.load_mode == 'both':
        track_df = pd.read_csv(track_file_name)
        if flag_ped:
            pedestrian_df = pd.read_csv(pedestrian_file_name)

    elif args.load_mode == 'vehicle':
        track_df = pd.read_csv(track_file_name)
    elif args.load_mode == 'pedestrian':
        pedestrian_df = pd.read_csv(pedestrian_file_name)
        
    agent_sample = []
    for i in list(track_df['case_id'].unique()):
        temp = track_df[track_df['case_id'] == i]
        for j in list(temp['track_id'].unique()):
            temp_list = list(temp[temp['track_id'] == j]['frame_id'].unique())
            agent_sample += [{'case_id':int(i),'track_id':j,'frame_id':k} for k in is_satisfy(temp_list, N_past_track, N_future_track)]
    
    lat_origin = args.lat_origin  # origin is necessary to correctly project the lat lon values of the map to the local
    lon_origin = args.lon_origin  # coordinates in which the tracks are provided; defaulting to (0|0) for every scenario
    plot_agent(lanelet_map_file,lat_origin,lon_origin)
    plt.show()
