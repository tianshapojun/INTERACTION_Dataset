import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np 

def raster_visual_fig(config, box_img, map_img_1, map_img_2):
    plt.subplot(2,3,1)
    plt.imshow(box_img[...,config["history_num_frames"]])
    plt.subplot(2,3,2)
    plt.imshow(map_img_2)
    plt.subplot(2,3,3)
    plt.imshow(box_img[...,config["history_num_frames"]] + map_img_1[:,:,2])
    for i in range(3):
        plt.subplot(2,3,i+4)    
        plt.imshow(map_img_1[:,:,i])
    plt.show()
    
def raster_visual_anim(N, box_img, map_img_1, save_flg = False):
    def update(n):
        fig.clear() 
        plt.imshow(box_img[...,n] + map_img_1[:,:,2])
        #plt.imshow(box_img[...,n])
        plt.title(n) 

    fig = plt.figure()
    ani = FuncAnimation(fig, update, frames=range(N), interval=100)
    if save_flg:
        print('Begin saving gif')
        ani.save('D:/Code/INTERACTION_Dataset/test1.gif', writer='imagemagick', fps=60)
    plt.show()
    
def plot_matplotlib(map, agent_x, agent_y, agent_yaw, track_list, save_flg = False):
    #fig, axes = plt.subplots(1, 1)
    #fig.canvas.manager.set_window_title("Interaction Dataset Visualization")
    #print("Loading map...")
    #draw_map_without_lanelet(map, axes)
    #update_objects_plot(axes, agent_x, agent_y, agent_yaw, track_list)
    def update(n):
        #fig, axes = plt.subplots(1, 1)
        fig.clear() 
        axes = fig.add_subplot(111)
        axes.set_title('Frame:{}'.format(n))
        #fig.canvas.manager.set_window_title("Interaction Dataset Visualization")
        draw_map_without_lanelet(map, axes)
        update_objects_plot(axes, agent_x, agent_y, agent_yaw, track_list[n])

    fig = plt.figure()
    ani = FuncAnimation(fig, update, frames=range(len(track_list)), interval=100)
    if save_flg:
        print('Begin saving gif')
        ani.save('D:/Code/INTERACTION_Dataset/test11.gif', writer='imagemagick', fps=60)
    #plt.show()
    
def set_visible_area(point_dict, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for key,point in point_dict.items():
        min_x = min(point[0], min_x)
        min_y = min(point[1], min_y)
        max_x = max(point[0], max_x)
        max_y = max(point[1], max_y)

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 10, max_x + 10])
    axes.set_ylim([min_y - 10, max_y + 10])

def draw_map_without_lanelet(map, axes):
    assert isinstance(axes, matplotlib.axes.Axes)

    axes.set_aspect('equal', adjustable='box')
    axes.patch.set_facecolor('lightgrey')

    point_dict = map.point_dict
    set_visible_area(point_dict, axes)

    unknown_linestring_types = list()

    for key,way in map.way_dict.items():
        way_type = way['type']
        if way_type is None:
            raise RuntimeError("Linestring type must be specified")
        elif way_type == "curbstone":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "line_thin":
            type_dict = dict(color="white", linewidth=1, zorder=10)
        elif way_type == "line_thick":
            type_dict = dict(color="white", linewidth=2, zorder=10)
        elif way_type == "pedestrian_marking":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        elif way_type == "bike_marking":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        elif way_type == "stop_line":
            type_dict = dict(color="white", linewidth=3, zorder=10)
        elif way_type == "virtual":
            type_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[2, 5])
        elif way_type == "road_border":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "guard_rail":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "traffic_sign":
            continue
        else:
            if way_type not in unknown_linestring_types:
                unknown_linestring_types.append(way_type)
            continue

        plt.plot(way["position"][:,0], way["position"][:,1], **type_dict)

        #for key,value in point_dict.items():
            #plt.text(value.x,value.y,key)

    if len(unknown_linestring_types) != 0:
        print("Found the following unknown types, did not plot them: " + str(unknown_linestring_types))
        
def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


def polygon_xy_from_motionstate(ms, width, length, agent_x, agent_y, agent_yaw):
    lowleft = (ms[1] - length / 2., ms[2] - width / 2.)
    lowright = (ms[1] + length / 2., ms[2] - width / 2.)
    upright = (ms[1] + length / 2., ms[2] + width / 2.)
    upleft = (ms[1] - length / 2., ms[2] + width / 2.)
    temp = rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([ms[1], ms[2]]), yaw=ms[3])
    #return np.dot(temp - np.array([agent_x, agent_y]),np.array([[np.cos(agent_yaw), -np.sin(agent_yaw)], [np.sin(agent_yaw), np.cos(agent_yaw)]]))
    return temp  
        
def update_objects_plot(axes, agent_x, agent_y, agent_yaw, track_list): 
    for value in track_list:
        width, length = value[4], value[5]

        rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(value, width, length, agent_x, agent_y, agent_yaw), closed=True,
                                                      zorder=20)
        axes.add_patch(rect)
        #temp = np.dot(np.array([value[1]-agent_x, value[2]-agent_y]),np.array([[np.cos(agent_yaw), -np.sin(agent_yaw)], [np.sin(agent_yaw), np.cos(agent_yaw)]]))
        temp =[value[1], value[2]]
        axes.text(temp[0], temp[1] + 2, str(int(value[0])), horizontalalignment='center', zorder=30)
