import numpy as np
import matplotlib.pyplot as plt

def get_relative_pose(cood, center, yaw):
    return np.dot(cood - center,np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]))

def angular_distance(angle_a, angle_b):
    #A function that takes two arrays of angles in radian and compute the angular distance, wrap the angular
    #distance such that they are always in the [-pi, pi) range.
    return (angle_a - angle_b + np.pi) % (2 * np.pi) - np.pi

def filter_agents_by_distance(agents_id, agents, centroid, max_distance):
    #Filter agents by distance, cut to `max_distance` and sort the result
    agents_dist = np.linalg.norm(agents - centroid, axis=-1)
    agents_id = agents_id[agents_dist < max_distance]
    agents_dist = agents_dist[agents_dist < max_distance]
    agents_id = agents_id[np.argsort(agents_dist)]
    return agents_id

def get_other_agents_ids(all_agents_ids, priority_ids, selected_track_id, max_agents) :
    #Get ids of agents around selected_track_id. Give precedence to `priority_ids`
    #over `all_agents_ids` and cut to `max_agents`
    agents_taken = set()
    # ensure we give priority to reliable, then fill starting from the past
    for agent_id in np.concatenate([priority_ids, all_agents_ids]):
        if len(agents_taken) >= max_agents:
            break
        if agent_id != selected_track_id:
            agents_taken.add(agent_id)
    return list(agents_taken)

def indices_in_bounds(center, map, half_extent):
    """
    Get indices of elements for which the bounding box described by bounds intersects the one defined around
    center (square with side 2*half_side)
    """
    bounds = map.rel_bounds_region
    x_center, y_center = center

    x_min_in = x_center > bounds[:, 0, 0] - half_extent
    y_min_in = y_center > bounds[:, 0, 1] - half_extent
    x_max_in = x_center < bounds[:, 1, 0] + half_extent
    y_max_in = y_center < bounds[:, 1, 1] + half_extent
    #return map.rel_bounds_region[np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]]
    return map.rel_bounds_id[np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]]

def vectorize_ego(track_id, history_num_ego_frames, history_track,current_track,future_track):
    #current:28, history:28,27,26,25, future:29,30,31
    agent = current_track[current_track['track_id'] == track_id]
    agent_extent_m = np.array([agent['length'].iloc[0],agent['width'].iloc[0]], dtype = np.float32)
    agent_centroid_m = np.array([agent['x'].iloc[0],agent['y'].iloc[0]], dtype = np.float32)
    agent_yaw_rad = np.array(agent['psi_rad'].iloc[0], dtype = np.float32)
    
    history = history_track[history_track['track_id'] == track_id].sort_values(by = ['frame_id'],axis = 0,ascending = False)
    history_coords = np.array(history[['x','y']], dtype = np.float32)
    history_yaws = np.array(history['psi_rad'], dtype = np.float32)
    history_coords_offset = get_relative_pose(history_coords, agent_centroid_m, agent_yaw_rad)
    history_yaws_offset = angular_distance(history_yaws,agent_yaw_rad)
    
    future = future_track[future_track['track_id'] == track_id].sort_values(by = ['frame_id'],axis = 0,ascending = True)
    future_coords = np.array(future[['x','y']], dtype = np.float32)
    future_yaws = np.array(future['psi_rad'], dtype = np.float32)
    future_coords_offset = get_relative_pose(future_coords, agent_centroid_m, agent_yaw_rad)
    future_yaws_offset = angular_distance(future_yaws,agent_yaw_rad)
    
    history_coords_offset[history_num_ego_frames + 1:] *= 0
    history_yaws_offset[history_num_ego_frames + 1:] *= 0
    
    frame_info = {
        "extent": agent_extent_m,
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "speed": np.linalg.norm([agent['vx'].iloc[0],agent['vy'].iloc[0]]),
        "history_positions": history_coords_offset,
        "history_yaws": history_yaws_offset,
        "target_positions": future_coords_offset,
        "target_yaws": future_yaws_offset
    }
    return frame_info
    
def vectorize_agents(track_id, frame_id, ego_centroid, ego_yaw, history_ego_position, history_ego_yaws, history_track, current_track
                     , max_agents_distance, other_agents_num, history_num_frames, future_num_frames):
    # compute agent features
    # sequence_length x 2 (two being x, y)
    ego_points = history_ego_position.copy()
    # sequence_length x 1
    ego_yaws = history_ego_yaws[:,np.newaxis].copy()
    # sequence_length x xy+yaw (3)
    ego_trajectory_polyline = np.concatenate([ego_points, ego_yaws], axis=-1)

    # get agents around AoI sorted by distance in a given radius. Give priority to agents in the current time step
    history_agents_id = filter_agents_by_distance(np.array(history_track['track_id']), np.array(history_track[['x','y']]), ego_centroid, max_agents_distance)
    cur_agents_id = filter_agents_by_distance(np.array(current_track['track_id']), np.array(current_track[['x','y']]), ego_centroid, max_agents_distance)
    list_agents_to_take = get_other_agents_ids(history_agents_id, cur_agents_id, track_id, other_agents_num)
    
    # Loop to grab history and future for all other agents 
    all_other_agents_history_positions = np.zeros((other_agents_num, history_num_frames + 1, 2))
    all_other_agents_history_yaws = np.zeros((other_agents_num, history_num_frames + 1, 1))
    all_other_agents_history_extents = np.zeros((other_agents_num, history_num_frames + 1, 2))
    all_other_agents_history_availability = np.zeros((other_agents_num, history_num_frames + 1))
    all_other_agents_types = np.zeros((other_agents_num,))
    all_other_agents_track_ids = np.zeros((other_agents_num,))

    for idx, agent_id in enumerate(list_agents_to_take):
        temp = history_track[history_track['track_id'] == agent_id].sort_values(by = ['frame_id'],axis = 0,ascending = False)
        idx2 = frame_id - np.array(temp['frame_id'])
        all_other_agents_history_positions[idx,idx2,:] = get_relative_pose(np.array(temp[['x','y']]), ego_centroid, ego_yaw)
        all_other_agents_history_yaws[idx,idx2,:] = angular_distance(np.array(temp[['psi_rad']]), ego_yaw)
        all_other_agents_history_extents[idx,idx2,:] = np.array(temp[['length','width']])
        all_other_agents_history_availability[idx,idx2] = 1
        all_other_agents_types[idx] = 1
        all_other_agents_track_ids[idx] = agent_id
    
    # compute other agents features
    # num_other_agents (M) x sequence_length x 2 (two being x, y)
    agents_points = all_other_agents_history_positions.copy()
    # num_other_agents (M) x sequence_length x 1
    agents_yaws = all_other_agents_history_yaws.copy()
    # num_other_agents (M) x sequence_length x self._vector_length
    other_agents_polyline = np.concatenate([agents_points, agents_yaws], axis=-1)
    other_agents_polyline_availability = all_other_agents_history_availability.copy()
    
    agent_dict = {
            "all_other_agents_history_positions": all_other_agents_history_positions,
            "all_other_agents_history_yaws": all_other_agents_history_yaws,
            "all_other_agents_history_extents": all_other_agents_history_extents,
            "all_other_agents_history_availability": all_other_agents_history_availability.astype(bool),
            "all_other_agents_types": all_other_agents_types,
            "all_other_agents_track_ids": all_other_agents_track_ids,
            "agent_trajectory_polyline": ego_trajectory_polyline,
            "other_agents_polyline": other_agents_polyline,
            "other_agents_polyline_availability": other_agents_polyline_availability.astype(bool),
        }
    
    return agent_dict

def vectorize_map(config, map, ego_centroid, ego_yaw):
    # START WORKING ON LANES
        MAX_LANES = config["max_num_lanes"]
        MAX_POINTS_LANES = config["max_points_per_lane"]
        #MAX_CROSSWALKS = config["max_num_crosswalks"]
        #MAX_POINTS_CW = config["max_points_per_crosswalk"]

        MAX_LANE_DISTANCE = config["max_retrieval_distance"]
        STEP_INTERPOLATION = MAX_POINTS_LANES  # number of points along lane

        lanes_points = np.zeros((MAX_LANES * 2, MAX_POINTS_LANES, 2), dtype=np.float32)
        lanes_availabilities = np.zeros((MAX_LANES * 2, MAX_POINTS_LANES), dtype=np.float32)

        lanes_mid_points = np.zeros((MAX_LANES, MAX_POINTS_LANES, 2), dtype=np.float32)
        lanes_mid_availabilities = np.zeros((MAX_LANES, MAX_POINTS_LANES), dtype=np.float32)
        
        #lane_bounds: M x 2 x 2
        lanes_indices = indices_in_bounds(ego_centroid, map, MAX_LANE_DISTANCE)
        lanes_indices_lr = np.zeros(0)
        #print(lanes_indices)
        distances = []
        for lane_idx in lanes_indices:
            lane = map.rel_dict[lane_idx]['lane_interpolation']
            try:
                lane["xy_mid"]
            except:
                print('Lane {} does not have left/right way'.format(lane_idx))
                continue
            lane_dist = np.linalg.norm(lane["xy_mid"][:, :2] - ego_centroid, axis=-1)
            lanes_indices_lr = np.append(lanes_indices_lr, lane_idx)
            distances.append(np.min(lane_dist))
        lanes_indices_lr = lanes_indices_lr[np.argsort(distances)]
        
        for out_idx, lane_idx in enumerate(lanes_indices_lr[:MAX_LANES]):
            lane = map.rel_dict[lane_idx]["lane_interpolation"]
            xy_left = lane["xy_left"][:MAX_POINTS_LANES, :2]
            xy_right = lane["xy_right"][:MAX_POINTS_LANES, :2]
            # convert coordinates into local space
            xy_left = get_relative_pose(xy_left, ego_centroid, ego_yaw)
            xy_right = get_relative_pose(xy_right, ego_centroid, ego_yaw)

            num_vectors_left = len(xy_left)
            num_vectors_right = len(xy_right)

            lanes_points[out_idx * 2, :num_vectors_left] = xy_left
            lanes_points[out_idx * 2 + 1, :num_vectors_right] = xy_right

            lanes_availabilities[out_idx * 2, :num_vectors_left] = 1
            lanes_availabilities[out_idx * 2 + 1, :num_vectors_right] = 1

            midlane = lane["xy_mid"][:MAX_POINTS_LANES, :2]
            midlane = get_relative_pose(midlane, ego_centroid, ego_yaw)
            num_vectors_mid = len(midlane)

            lanes_mid_points[out_idx, :num_vectors_mid] = midlane
            lanes_mid_availabilities[out_idx, :num_vectors_mid] = 1
        
        # disable all points over the distance threshold
        valid_distances = np.linalg.norm(lanes_points, axis=-1) < MAX_LANE_DISTANCE
        lanes_availabilities *= valid_distances
        valid_mid_distances = np.linalg.norm(lanes_mid_points, axis=-1) < MAX_LANE_DISTANCE
        lanes_mid_availabilities *= valid_mid_distances
        
        # 2 MAX_LANES x MAX_VECTORS x (XY + TL-feature)
        # -> 2 MAX_LANES for left and right
        lanes = np.concatenate([lanes_points, np.zeros_like(lanes_points[..., [0]])], axis=-1)
        # MAX_LANES x MAX_VECTORS x 3 (XY + 1 TL-feature)
        lanes_mid = np.concatenate([lanes_mid_points, np.zeros_like(lanes_mid_points[..., [0]])], axis=-1)
        
        return {
            "lanes": lanes,
            "lanes_availabilities": lanes_availabilities.astype(bool),
            "lanes_mid": lanes_mid,
            "lanes_mid_availabilities": lanes_mid_availabilities.astype(bool)
        }
