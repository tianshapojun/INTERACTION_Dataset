#!/usr/bin/env python

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt

import xml.etree.ElementTree as xml
import pyproj
import math
import numpy as np
import cv2

from . import dict_utils
from .dataset_types import Track, MotionState

CV2_SUB_VALUES = {"shift": 9, "lineType": cv2.LINE_AA}
CV2_SHIFT_VALUE = 2 ** CV2_SUB_VALUES["shift"]

##### Matplotlib
#####
##### 
#####
#####

class Point:
    def __init__(self):
        self.x = None
        self.y = None


class LL2XYProjector:
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1  # works for most tiles, and for all in the dataset
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]

def xy2agent(point,agent_x,agent_y,agent_yaw):
    temp_x = np.dot(np.array([point.x-agent_x,point.y-agent_y]),np.array([np.cos(agent_yaw),np.sin(agent_yaw)]))
    temp_y = np.dot(np.array([point.x-agent_x,point.y-agent_y]),np.array([-np.sin(agent_yaw),np.cos(agent_yaw)]))
    point.x,point.y = temp_x,temp_y

def get_type(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "type":
            return tag.get("v")
    return None


def get_subtype(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "subtype":
            return tag.get("v")
    return None


def get_x_y_lists(element, point_dict):
    x_list = list()
    y_list = list()
    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        point = point_dict[pt_id]
        x_list.append(point.x)
        y_list.append(point.y)
    return x_list, y_list


def set_visible_area(point_dict, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for id, point in dict_utils.get_item_iterator(point_dict):
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 10, max_x + 10])
    axes.set_ylim([min_y - 10, max_y + 10])


def draw_map_without_lanelet(filename, axes, lat_origin, lon_origin,agent_x,agent_y,agent_yaw):
    assert isinstance(axes, matplotlib.axes.Axes)
    assert agent_yaw == float(agent_yaw)

    axes.set_aspect('equal', adjustable='box')
    axes.patch.set_facecolor('lightgrey')

    projector = LL2XYProjector(lat_origin, lon_origin)

    e = xml.parse(filename).getroot()

    point_dict = dict()
    for node in e.findall("node"):
        point = Point()
        point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
        xy2agent(point,agent_x,agent_y,agent_yaw)
        point_dict[int(node.get('id'))] = point

    set_visible_area(point_dict, axes)

    unknown_linestring_types = list()

    for way in e.findall('way'):
        way_type = get_type(way)
        if way_type is None:
            raise RuntimeError("Linestring type must be specified")
        elif way_type == "curbstone":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "line_thin":
            way_subtype = get_subtype(way)
            if way_subtype == "dashed":
                type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="white", linewidth=1, zorder=10)
        elif way_type == "line_thick":
            way_subtype = get_subtype(way)
            if way_subtype == "dashed":
                type_dict = dict(color="white", linewidth=2, zorder=10, dashes=[10, 10])
            else:
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

        x_list, y_list = get_x_y_lists(way, point_dict)
        plt.plot(x_list, y_list, **type_dict)

        #for key,value in point_dict.items():
            #plt.text(value.x,value.y,key)

    if len(unknown_linestring_types) != 0:
        print("Found the following unknown types, did not plot them: " + str(unknown_linestring_types))

def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


def polygon_xy_from_motionstate(ms, width, length, agent_x, agent_y, agent_yaw):
    assert isinstance(ms, MotionState)
    lowleft = (ms.x - length / 2., ms.y - width / 2.)
    lowright = (ms.x + length / 2., ms.y - width / 2.)
    upright = (ms.x + length / 2., ms.y + width / 2.)
    upleft = (ms.x - length / 2., ms.y + width / 2.)
    temp = rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([ms.x, ms.y]), yaw=ms.psi_rad)
    #print(temp)
    return np.dot(temp - np.array([agent_x, agent_y]),np.array([[np.cos(agent_yaw), -np.sin(agent_yaw)], [np.sin(agent_yaw), np.cos(agent_yaw)]]))


def polygon_xy_from_motionstate_pedest(ms, width, length):
    assert isinstance(ms, MotionState)
    lowleft = (ms.x - length / 2., ms.y - width / 2.)
    lowright = (ms.x + length / 2., ms.y - width / 2.)
    upright = (ms.x + length / 2., ms.y + width / 2.)
    upleft = (ms.x - length / 2., ms.y + width / 2.)
    return np.array([lowleft, lowright, upright, upleft])


def update_objects_plot(frame_id, axes, agent_x, agent_y, agent_yaw, track_dict=None): 
    if track_dict is not None:

        for key, value in track_dict.items():
            assert isinstance(value, Track)
            try:
                ms = value.motion_states[frame_id]
            except:
                continue
            
            assert isinstance(ms, MotionState)
            #print(ms.x,ms.y,key)

            width = value.width
            length = value.length

            rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width, length, agent_x, agent_y, agent_yaw), closed=True,
                                                      zorder=20)
            axes.add_patch(rect)
            temp = np.dot(np.array([ms.x-agent_x, ms.y-agent_y]),np.array([[np.cos(agent_yaw), -np.sin(agent_yaw)], [np.sin(agent_yaw), np.cos(agent_yaw)]]))
            axes.text(temp[0], temp[1] + 2, str(key), horizontalalignment='center', zorder=30)

##### Rasterization
#####
##### 
#####
#####

def polygon_xy_from_df(x, y, psi_rad, width, length, agent_x, agent_y, agent_yaw):
    lowleft = (x - length / 2., y - width / 2.)
    lowright = (x + length / 2., y - width / 2.)
    upright = (x + length / 2., y + width / 2.)
    upleft = (x - length / 2., y + width / 2.)
    temp = rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([x, y]), yaw = psi_rad)
    return np.dot(temp - np.array([agent_x, agent_y]),np.array([[np.cos(agent_yaw), -np.sin(agent_yaw)], [np.sin(agent_yaw), np.cos(agent_yaw)]]))

def box_raster(track_dictionary,x,y,psd,raster_size,pixel_size,ego_center):
    world_center_coods = np.array(track_dictionary[['x','y','psi_rad','width','length']])
    hh = np.array([polygon_xy_from_df(i[0],i[1],i[2],i[3],i[4],x,y,psd) for i in world_center_coods]) #Nx4x2
    hh2 = np.array([ [[j[0]/pixel_size[0] + ego_center[0]*raster_size[0],-j[1]/pixel_size[1] + ego_center[1]*raster_size[1]] for j in i] for i in hh])
    im = np.zeros((raster_size[1],raster_size[0]))
    
    cv2.fillPoly(im, (hh2* CV2_SHIFT_VALUE).astype(np.int32), color=255,**CV2_SUB_VALUES)
    plt.imshow(im)
    plt.show()
    
def get_x_y_array(element, point_dict):
    x_y_list = []
    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        point = point_dict[pt_id]
        x_y_list.append([point.x,point.y])
    return np.array(x_y_list)

def rotational_sort(list_of_xy_coords):
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(angles)
    return list_of_xy_coords[indices]

def map_raster(filename, lat_origin, lon_origin,agent_x,agent_y,agent_yaw,raster_size,pixel_size,ego_center):
    assert agent_yaw == float(agent_yaw)

    projector = LL2XYProjector(lat_origin, lon_origin)
    e = xml.parse(filename).getroot()
    #im = np.zeros((raster_size[1],raster_size[0]))
    im = 255 * np.ones(shape=(raster_size[1], raster_size[0], 3), dtype=np.uint8)

    point_dict = dict()
    for node in e.findall("node"):
        point = Point()
        point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
        xy2agent(point,agent_x,agent_y,agent_yaw)
        point.x, point.y = point.x/pixel_size[0] + ego_center[0]*raster_size[0],-point.y/pixel_size[1] + ego_center[1]*raster_size[1]
        point_dict[int(node.get('id'))] = point

    unknown_linestring_types = list()
    way_dict = dict()
    for way in e.findall('way'):
        x_y_array = get_x_y_array(way, point_dict) # Nx2
        way_dict[int(way.get('id'))] = x_y_array
        way_type = get_type(way)
        if way_type is None:
            raise RuntimeError("Linestring type must be specified")
        elif way_type in ["curbstone", "road_border", "guard_rail"]:
            COLOR = (255, 217, 82)
        elif way_type in ["line_thin", "line_thick", "pedestrian_marking", "bike_marking", "stop_line"]:
            COLOR = (17, 17, 31)
        elif way_type == "virtual":
            COLOR = (255, 117, 69)
        elif way_type == "traffic_sign":
            continue
        else:
            if way_type not in unknown_linestring_types:
                unknown_linestring_types.append(way_type)
            continue

        cv2.polylines(im, [(x_y_array* CV2_SHIFT_VALUE).astype(np.int32)], False, color=COLOR, thickness = 1, **CV2_SUB_VALUES)
        
    for relation in e.findall('relation'):
        rel_list = np.zeros([0,2]) 
        for mb in relation.findall("member"):
            if mb.get("type") == 'way':
                mb_id = int(mb.get("ref"))
                rel_list = np.concatenate((way_dict[mb_id],rel_list),axis = 0)
                
        #print(relation.get("id"),rel_list.shape)
        if rel_list.shape[0] > 0:
            cv2.fillPoly(im, [(rotational_sort(rel_list) * CV2_SHIFT_VALUE).astype(np.int32)], color=(0, 255, 0),**CV2_SUB_VALUES)
          
    for i in range(3):
        plt.subplot(1,3,i+1)    
        plt.imshow(im[:,:,i])
    plt.show()
    
    if len(unknown_linestring_types) != 0:
        print("Found the following unknown types, did not plot them: " + str(unknown_linestring_types))
