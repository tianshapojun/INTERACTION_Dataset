import xml.etree.ElementTree as xml
import numpy as np

from .map_vis_without_lanelet import LL2XYProjector,get_type,rotational_sort

def get_x_y_array(element, point_dict):
    x_y_list = []
    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        point = point_dict[pt_id]
        x_y_list.append(point)
    return np.array(x_y_list)

class map_api:
    def __init__(self, config, filename, lat_origin, lon_origin):
        self.lat_origin, self.lon_origin = lat_origin, lon_origin
        self.e = xml.parse(filename).getroot()
        self.config = config
        print("Loading points...")
        self.point_dict = self.get_point_dict()
        print("Loading ways...")
        self.way_dict = self.get_way_dict()
        print("Loading relations...")
        self.rel_dict = self.get_rel_dict()
        print("Loading bounds...")
        self.rel_bounds_region, self.rel_bounds_id = self.get_bounds()
        
    def get_point_dict(self):
        projector = LL2XYProjector(self.lat_origin, self.lon_origin)
        point_dict = dict()
        for node in self.e.findall("node"):
            x, y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
            point_dict[int(node.get('id'))] = [x,y]
        return point_dict
    
    def get_way_dict(self):
        unknown_linestring_types = list()
        way_dict = dict()
        for way in self.e.findall('way'):
            x_y_array = get_x_y_array(way, self.point_dict) # Nx2
            x_y_inter = self.interpolate(x_y_array, self.config['max_points_per_lane'], self.config['interpolation_method']) 
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
                COLOR = 'None'
            else:
                if way_type not in unknown_linestring_types:
                    unknown_linestring_types.append(way_type)
                COLOR = 'None'
            way_dict[int(way.get('id'))] = {"position":x_y_array, "type":way_type, "color":COLOR, "interpolation":x_y_inter}
            
        if len(unknown_linestring_types) != 0:
            print("Found the following unknown types, did not plot them: " + str(unknown_linestring_types))
            
        return way_dict
    
    def get_rel_dict(self):
        rel_dict = dict()
        for relation in self.e.findall('relation'):
            way_position = []
            way_id = []
            way_type = []
            way_role = []
            for mb in relation.findall("member"):
                #限定way和left/right线
                if mb.get("type") == 'way' and mb.get("role") in ['left','right']:
                    mb_id = int(mb.get("ref"))
                    way_position.append(self.way_dict[mb_id]['position'])
                    way_id.append(mb_id)
                    way_type.append(self.way_dict[mb_id]['type'])
                    way_role.append(mb.get("role"))
            if len(way_position) > 0:   
                x_min = np.min(np.concatenate(way_position)[:,0])
                y_min = np.min(np.concatenate(way_position)[:,1])
                x_max = np.max(np.concatenate(way_position)[:,0])
                y_max = np.max(np.concatenate(way_position)[:,1])
                rel_dict[int(relation.get("id"))] = {"position":way_position, "id":way_id, "type":way_type, "role":way_role
                    ,"bounds":np.array([[x_min, y_min], [x_max, y_max]])
                    , "lane_interpolation":self.get_lane_as_interpolation(way_id, way_role, self.config['max_points_per_lane'])}
                #print(rel_dict[int(relation.get("id"))]["lane_interpolation"])
                #hhh
            
        return rel_dict
    
    def get_bounds(self):
        temp_array = np.zeros([len(self.rel_dict),2,2])
        temp_ids = []
        for idx,(key,value) in enumerate(self.rel_dict.items()):
            temp_array[idx] = value['bounds']
            temp_ids.append(int(key))
        return temp_array, np.array(temp_ids)
    
    def interpolate(self, xyz, step, method):
        #Interpolate points based on cumulative distances from the first one. 
        cum_dist = np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=-1))
        cum_dist = np.insert(cum_dist, 0, 0)

        if method == "INTER_LEN":
            step = int(step)
            assert step > 1, "step must be at least 2 with INTER_ENSURE_LEN"
            steps = np.linspace(cum_dist[0], cum_dist[-1], step)

        elif method == "INTER_METER":
            assert step > 0, "step must be greater than 0 with INTER_FIXED"
            steps = np.arange(cum_dist[0], cum_dist[-1], step)
        else:
            raise NotImplementedError(f"interpolation method should be precise")

        xyz_inter = np.empty((len(steps), 2), dtype=xyz.dtype)
        xyz_inter[:, 0] = np.interp(steps, xp=cum_dist, fp=xyz[:, 0])
        xyz_inter[:, 1] = np.interp(steps, xp=cum_dist, fp=xyz[:, 1])
        return xyz_inter
    
    #'''
    def get_lane_as_interpolation(self, way_id, way_role, step):
        #Perform an interpolation of the left and right lanes and compute the midlane.
        for i in range(len(way_role)):
            if way_role[i] == 'left':
                id_left = way_id[i]
            elif way_role[i] == 'right':
                id_right = way_id[i]
        try:
            id_left
            id_right
        except: return dict()
        lane_dict = dict()
        lane_dict["xy_left"] = self.way_dict[id_left]['interpolation']
        lane_dict["xy_right"] = self.way_dict[id_right]['interpolation']
        temp = list(rotational_sort(np.array([lane_dict["xy_left"][0], lane_dict["xy_left"][-1], lane_dict["xy_right"][0], lane_dict["xy_right"][-1]]))[0])
        if abs(temp.index(0)-temp.index(2)) == 1 or abs(temp.index(1)-temp.index(3)) == 1: 
               xy_midlane = (lane_dict["xy_left"] + lane_dict["xy_right"]) / 2
        else:
            xy_midlane = (lane_dict["xy_left"] + lane_dict["xy_right"][::-1]) / 2

        # interpolate xyz for midlane with the selected interpolation
        lane_dict["xy_mid"] = self.interpolate(xy_midlane, step, self.config['interpolation_method'])
        return lane_dict
        #'''
