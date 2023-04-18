import re
import numpy as np
import struct as st
import matplotlib.pyplot as plt
import cv2
import copy
import sys

class SerialDataModel(object):
    def __init__(self):
        self.data_str_table = {'SYS-T': 0,
                               'V-BAT': 1,
                               'CURRENT': 2,
                               'YAW': 3,
                               'PITCH': 4,
                               'ROLL': 5,
                               'TCRT-ON-L': 6,
                               'TCRT-ON-M': 7,
                               'TCRT-ON-R': 8,
                               'TCRT-OFF-L': 9,
                               'TCRT-OFF-M': 10,
                               'TCRT-OFF-R': 11,
                               'APDS-L': 12,
                               'APDS-R': 13,
                               'COL-S-F-R': 14,
                               'COL-S-F-G': 15,
                               'COL-S-F-B': 16,
                               'COL-S-B-R': 17,
                               'COL-S-B-G': 18,
                               'COL-S-B-B': 19,
                               'COL-S-L-R': 20,
                               'COL-S-L-G': 21,
                               'COL-S-L-B': 22,
                               'COL-S-R-R': 23,
                               'COL-S-R-G': 24,
                               'COL-S-R-B': 25,
                               'Energy': 26,
                               'FAvoid': 27,
                               'FGather': 28,
                               'State':29
                               }
        # data
        # last 7B is reserved
        # '=' means that Size and alignment, bytes
        self.pack_code = '=1B1L2B3h6H2H12H9f7B'
        self.num_package = 0
        # robot
        self.robot_data = {}

    def data_transfer(self, b):
        # unpack data
        data = st.unpack(self.pack_code, b)
        self.num_package += 1
        # re-arrange data based on robot ID
        # already exits
        if data[0] in self.robot_data.keys():
            self.robot_data[data[0]].append(data[1:])
        else:
            # add new id to robot data
            self.robot_data.update({data[0]: [data[1:]]})

    def get_robots_data(self, data_str, t=None):
        data = {}
        if t is not None:
            list_data = [v[t][self.data_str_table[data_str]] for k, v in self.robot_data.items()]
        else:
            list_data = [[v[t][self.data_str_table[data_str]] for t in range(len(v))] for k, v in self.robot_data.items()]
        for i, k in enumerate(self.robot_data.keys()):
            data.update({k: list_data[i]})
        return data

    def get_robot_data(self, robot_id, data_str, t=None):
        data = list(zip(*self.robot_data[robot_id]))
        return data[self.data_str_table[data_str]] if t is None else data[self.data_str_table[data_str]][t]


class LedImageProcess:
    def __init__(self):
        pass
    
    def generate_test_img(self, img_w, img_h, fold_x, fold_y, pattern=None):
        '''
        :param img_w:
        :param img_h:
        :param pattern: None->default, 0->gradient, 1->Net
        :return:
        '''
        # get the real image size
        img_w_real = int(img_w * (fold_x + 1))
        img_h_real = int(img_h / (fold_x + 1))
        img_real = np.zeros((img_h_real, img_w_real, 3), np.uint8)
        if pattern is None:
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, 'OpenCV', (10, 10), font, 4, (255, 255, 255), 2)
            # cv2.circle(img, (50, 50), 20, (255, 255, 255), -1)
            cv2.rectangle(img_real, (10, 10), (img_w_real - 10, img_h_real - 10), (255, 0, 255), 3)
        elif pattern == 0:
            pass
        elif pattern == 1:
            pass
        else:
            pass
        img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
        # fold the image
        img_f = np.zeros((img_h, img_w, 3), np.uint8)
        for i in range(fold_x + 1):
            img_f[img_h_real*i:img_h_real*(i+1), :, :] = img_real[:, img_w*i:img_w*(i+1), :]

        return img_f
    
    def generate_grating(self, img_w, img_h, fold_x, fold_y, strip_w, v, roll=0, half_cover=False):
        img_w_real = int(img_w * (fold_x + 1))
        img_h_real = int(img_h / (fold_x + 1))
        img_real = np.zeros([img_h_real, img_w_real])
        v = 0.005 * v * np.pi / 180
        strip_w = strip_w * np.pi / 180
        w = 2.0 * np.pi * v / strip_w
        for i in range(img_w_real):
            img_real[:, i] = 255 * (np.sin(w * (2 * i * np.pi / 180.0 / v)) + 1) / 2.0
        temp = img_real[:, 1200:int(img_w_real/2)+1200]
        # roll
        img_real = np.roll(img_real, int(roll/np.pi*2 * img_w_real))
        if half_cover:
            img_real[:, 1200:int(img_w_real/2)+1200] = temp
        # fold the image
        img_f = np.zeros((img_h, img_w), np.uint8)
        for i in range(fold_x + 1):
            img_f[img_h_real * i:img_h_real * (i + 1), :] = img_real[:, img_w * i:img_w * (i + 1)]

        return img_f

    
    def transfer_loaded_image(self, source_img, img_w, img_h, fold_x, fold_y, gray=False, resize=True, fold=True):
        img_w_real = int(img_w * (fold_x + 1))
        img_h_real = int(img_h / (fold_x + 1))
        # convert color
        if len(source_img.shape) == 3:
            img_real = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            
        if resize:
            img_real = cv2.resize(img_real, (img_w_real, img_h_real))
            
        if gray:
            if len(img_real.shape) == 3:
                img_real = cv2.cvtColor(img_real, cv2.COLOR_RGB2GRAY)

        if fold:
            # fold the image
            if len(img_real.shape) == 3:
                img_f = np.zeros((img_h, img_w, 3), np.uint8)
                for i in range(fold_x + 1):
                    img_f[img_h_real*i:img_h_real*(i+1), :, :] = img_real[:, img_w*i:img_w*(i+1), :]
            else:
                img_f = np.zeros((img_h, img_w), np.uint8)
                for i in range(fold_x + 1):
                    img_f[img_h_real * i:img_h_real * (i + 1), :] = img_real[:, img_w * i:img_w * (i + 1)]
            
            return img_f
        else:
            return img_real
    
    def generate_video(self, img, img_w, img_h, fold_x, fold_y, animation=None):
        '''
        :param img: source image to animate
        :param animation: dict, like {'roll': 20}
        :return:
        '''
        if animation is not None:
            if 'roll' in animation.keys():
                # roll
                img = np.roll(img, animation['roll'], axis=1)
            else:
                #TODO: other transform
                pass
            
        img_w_real = int(img_w * (fold_x + 1))
        img_h_real = int(img_h / (fold_x + 1))
        # fold the image
        if len(img.shape) == 3:
            img_f = np.zeros((img_h, img_w, 3), np.uint8)
            for i in range(fold_x + 1):
                img_f[img_h_real*i:img_h_real*(i+1), :, :] = img[:, img_w*i:img_w*(i+1), :]
        else:
            img_f = np.zeros((img_h, img_w), np.uint8)
            for i in range(fold_x + 1):
                img_f[img_h_real * i:img_h_real * (i + 1), :] = img[:, img_w * i:img_w * (i + 1)]
        
        return img_f

class PheromoneModel:
    def __init__(self):
        self.pixel_height = 480
        self.pixel_width = 640
        self.d_kernel_s_factor = np.ones(3, dtype=int)*51
        self.diffusion_factor = np.zeros(3)
        self.evaporation_factor = np.zeros(3)
        self.injection_factor = np.zeros(3)
        self.radius_factor = np.ones(3).astype('int')
        self.dt = 0.1
        self.update_parameters()
        
        # pheromone
        self.color_channel = {'red':0, 
                              'green':1, 
                              'blue':2}
        self.pheromone_field = np.zeros([self.pixel_height, 
                                         self.pixel_width, 
                                         3])

    def update_parameters(self):
        # diffusion_kernel
        # self.diffusion_kernel = np.ones([3,3,3])
        # for i, d in enumerate(self.diffusion_factor):
        #     self.diffusion_kernel[i] *= (1-d)/8
        #     self.diffusion_kernel[i,1,1] = d - 1
            # self.diffusion_kernel[i,0,0],self.diffusion_kernel[i,0,2],self.diffusion_kernel[i,2,0],self.diffusion_kernel[i,2,2] = 0,0,0,0

        # diffusion_kernel
        self.diffusion_kernel = [[]]*3
        for n, d in enumerate(self.diffusion_factor):
            sigma = self.d_kernel_s_factor[n] / 2
            s = 2*(sigma**2)
            x0 = int((self.d_kernel_s_factor[n]-1)/2)
            y0 = int((self.d_kernel_s_factor[n]-1)/2)
            diffusion_kernel = np.zeros([self.d_kernel_s_factor[n],
                                        self.d_kernel_s_factor[n]])
            for i in range(self.d_kernel_s_factor[n]):
                for j in range(self.d_kernel_s_factor[n]):
                    x = i - x0
                    y = j - y0
                    diffusion_kernel[i,j] = np.exp(-(x**2 + y**2)/s)
            diffusion_kernel = diffusion_kernel / (np.sum(diffusion_kernel) - \
                diffusion_kernel[x0,y0]) * (1-d)
            diffusion_kernel[x0,y0] = d - 1
            self.diffusion_kernel[n] = diffusion_kernel.copy()
            # self.diffusion_kernel[n] *= (1-d)/120
            # self.diffusion_kernel[n,5,5] = d - 1
            
        # injection_kernel
        self.injection_kernel = [[]]*3
        for n in range(3):
            r = self.radius_factor[n]
            injection_kernel = np.zeros([2*r, 2*r])
            for i in range(2*r):
                for j in range(2*r):
                    if (i-r)**2 + (j-r)**2 <= r**2:
                        injection_kernel[i, j] = 1
            self.injection_kernel[n] = injection_kernel.copy()
        
        # pheromone field
        self.pheromone_field = np.zeros([self.pixel_height, 
                                         self.pixel_width,
                                         3])
    def transform_loaded_image(self, source_img, img_w, img_h, gray=False, resize=True):
        if len(source_img.shape) == 3:
            img_real = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            
        if resize:
            img_real = cv2.resize(img_real, (img_w, img_h))
            
        if gray:
            if len(img_real.shape) == 3:
                img_real = cv2.cvtColor(img_real, cv2.COLOR_RGB2GRAY)
        
        return img_real
    
    def render_pheromone(self, robot_pos, channel, arena_length, arena_width):
        injection = np.zeros([self.pixel_height, self.pixel_width, 3])
        for k, v in robot_pos.items():
            if (v[0] >= 0) and (v[0] <= arena_length) and \
                (v[1] >= 0) and (v[1] <= arena_width):
                y = int(v[0]/arena_length*self.pixel_width)
                x = int(v[1]/arena_width*self.pixel_height)
                if str(k) in channel.keys():
                    ind = self.color_channel[channel[str(k)]]
                else:
                    ind = self.color_channel[channel['other']]
                r = int(self.radius_factor[ind])
                s_x = np.max([0, x-r])
                s_y = np.max([0, y-r])
                e_x = np.min([x+r, self.pixel_height])
                e_y = np.min([y+r, self.pixel_width])
                s_k_x = 0 if x-r > 0 else r-x
                s_k_y = 0 if y-r > 0 else r-y
                injection[s_x:e_x, s_y:e_y, ind] += np.array(self.injection_kernel[ind])[s_k_x:e_x-s_x+s_k_x,s_k_y:e_y-s_y+s_k_y]
        # evaporation
        e = -(1/(self.evaporation_factor)) * self.pheromone_field
        # injection
        i = self.injection_factor * injection
        # diffusion
        # !should separate channels and process individually
        r, g, b = cv2.split(self.pheromone_field)
        r = cv2.filter2D(r, -1, self.diffusion_kernel[0])
        g = cv2.filter2D(g, -1, self.diffusion_kernel[1])
        b = cv2.filter2D(b, -1, self.diffusion_kernel[2])
        d = cv2.merge([r, g, b])
        # update
        self.pheromone_field = self.pheromone_field + (e + i + d) * self.dt
        # print('max',self.pheromone_field.max(),'dmin',d.min(), 'dmax', d.max(), 'e', e.min(), e.max(), 'i', i.min(), i.max())
        return np.clip(self.pheromone_field, 0, 255).astype(np.uint8)
        # if self.pheromone_field.max() > 255:
        #     return (self.pheromone_field / self.pheromone_field.max() * 255).astype(np.uint8)
        # else:
        #     return self.pheromone_field.astype(np.uint8)
    
    def generate_calibration_pattern(self, arena_length):
        image = np.zeros([self.pixel_height, self.pixel_width, 3])
        # radius of pattern is 2cm
        radius = int(0.02 * (self.pixel_width/arena_length))
        for xi, yi in [(radius, radius),
                       (self.pixel_width-radius, radius),
                       (radius, self.pixel_height-radius),
                       (self.pixel_width-radius, self.pixel_height-radius)]:
            # outer white circle
            image = cv2.circle(image, (xi, yi), int(radius*7/8), (255, 255, 255), int(radius/4))
            # inner circle
            image = cv2.circle(image, (xi, yi), int(radius/4), (255, 255, 255), -1)
        return image.astype(np.uint8)


class LocDataModel:
    def __init__(self):
        self.re_pattern_loc = re.compile(r'\d+ \d*.\d* \d*.\d* \d*.\d*')
        self.re_pattern_num = re.compile(r'Detected \d+ of \d+')
        self.loc_data_str = []
        self.robot_num = 0
        self.robot_num_detected = 0
        self.latest_pos = {}
        self.data_length = 0

    def get_loc_data(self, raw_data):
        data_str = raw_data.decode()
        # get number of robot
        data_info = re.findall(self.re_pattern_num, data_str)
        if len(data_info) > 0:
            try:
                self.robot_num_detected = int(data_info[0].split()[1])
                self.robot_num = int(data_info[0].split()[3])
            except:
                print('cannot find')
        # get position of robots
        data = re.findall(self.re_pattern_loc, data_str)
        self.data_length += len(data)
        if len(self.loc_data_str) >= self.robot_num*100:
            del(self.loc_data_str[:len(data)])
        self.loc_data_str += data
        return data

    def get_last_pos(self, r_id=None, data=None, x_only=False, y_only=False):
        self.latest_pos = {}
        if data is None:
            data = self.loc_data_str
        if r_id is None:
            for i in range(-1, -len(data)-1, -1):
                r_id = int(data[i][:3])
                if r_id not in self.latest_pos.keys():
                    new_value = [float(data[i][4:9]), float(data[i][10:15])]
                    if x_only:
                        new_value = new_value[0]
                    if y_only:
                        new_value = new_value[1]
                    self.latest_pos.update({r_id: new_value})
                if len(self.latest_pos.keys()) >= self.robot_num:
                    break
            return self.latest_pos
        else:
            for i in range(-1, -len(data), -1):
                r_id = int(data[i][:3])
                if r_id == id:
                    if x_only:
                        return float(data[i][4:9])
                    if y_only:
                        return float(data[i][10:15])
                    return [float(data[i][4:9]), float(data[i][10:15])]

class Pattern(object):
    def __init__(self, id, ideal_r1, ideal_r0, IDTable):
        self.id = id
        self.thresh = 127
        self.search_img= []
        self.innersize = 0
        self.outersize = 0
        self.inner = []
        self.outer = []
        self.inner_ellipse = []
        self.outer_ellipse = []
        self.ideal_r0 = ideal_r0
        self.ideal_r1 = ideal_r1
        self.r_tolerence = 0.004
        self.r0 = 0
        self.r1 = 0
        self.IDTable = IDTable
        self.lost_frames = 61

        self.cam_pos = [[0,0]]
        self.previous_search = [False]
        self.angle = 0
        self.world_pos = [[0,0]]


class LocalizationModel(object):
    def __init__(self):
        is_corrected = True
        if is_corrected == False:
            print('The camera and coordination is not calibrated, please calibrate it first')
            sys.exit()
        else:
            with np.load('./camera/calibration_data.npz') as X:
                self.mtx, self.dist, rvecs, tvecs, self.PresM = [X[i] for i in ('mtx','dist','rvecs','tvecs','PresM')]
        temp_array = np.array([0,0,0])
        mtx1 = np.insert(self.mtx,3,temp_array,axis=1)
        rotM = cv2.Rodrigues(rvecs)[0]
        rot_trans_M = np.insert(np.insert(rotM,3,tvecs.T,axis=1),3,np.array([0,0,0,1]),axis=0)

        self.w2p_M = np.matrix(np.matmul(mtx1,rot_trans_M))
        self.p2w_M = self.w2p_M.I
        self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (1920,1200), 0, (1920,1200))
        
        IDTable = np.loadtxt('./temps/ID.txt')

        self.r_output = open("./temps/loc_data.txt",'w+') 

        self.num_of_pattern = 16
        self.arena_size = (1.4,0.8)
        self.image_size = (1920,1200)
        self.zoom_ratio = 2
        self.pattern_size = 0.04
        self.tolerence = 0.4
        self.r_tolerence = 0.004
        self.IDTable = IDTable

        self.pat_ratio = (60,145,280,340,400)

        self.pat_img_r = int((self.image_size[0]/self.arena_size[0]*self.pattern_size*0.9)/2)

        outer_r0 = self.pat_img_r*self.pat_ratio[2]/self.pat_ratio[4]
        outer_r1 = self.pat_img_r*self.pat_ratio[3]/self.pat_ratio[4]
        self.outer_area = np.pi * outer_r0 * outer_r1
        self.outer_r_ratio = outer_r0/outer_r1
        self.search_area_num = 0

        self.pattern = []
        for [id, r1, r0] in IDTable:
            self.pattern.append(Pattern(int(id), r1, r0, IDTable))

        self.cnt = 0
    
    def __del__(self):
        self.r_output.close()

    def get_possible_posi(self,img):
        dim = (int(img.shape[1] /self.zoom_ratio), int(img.shape[0] /self.zoom_ratio))
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img_gray = img
        cnt = 0
        try_thresh = 255
        possib_pos = []
        for try_i in range(0,1):
            try_thresh = int(try_thresh/2)
            ret, thresh = cv2.threshold(img_gray,try_thresh,255,cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            outer_area = self.outer_area/(self.zoom_ratio**2)
            outer_r_ratio = self.outer_r_ratio
            area_tolerence = outer_area*0.4
            ratio_tolerence = outer_r_ratio*0.2
            
            for i, hier in enumerate(hierarchy[0]):
                if hier[2] == -1 and hier[3] != -1:
                    if contours[hierarchy[0][i][3]].shape[0]>5:
                        outer_ellipse = cv2.fitEllipse(contours[hierarchy[0][i][3]])
                        outerArea = cv2.contourArea(contours[hierarchy[0][i][3]])
                        elli_ratio2 = outer_ellipse[1][0]/outer_ellipse[1][1]
                        if abs(outerArea - outer_area) < area_tolerence and abs(elli_ratio2 - outer_r_ratio) < ratio_tolerence:
                            temp = ((np.array(outer_ellipse[0]))* self.zoom_ratio).astype(int)
                            temp1 = []
                            cnt_i = 0
                            if cnt > 0:
                                for cnt_ii in range(0, cnt):
                                    if np.linalg.norm(temp-possib_pos[cnt_i])<self.pat_img_r:
                                        break
                                    cnt_i += 1
                            if cnt_i != cnt:
                                continue
                            cnt+=1
                            if cnt < 16:
                                possib_pos.append(temp)
                            elif int(cnt) == 16:
                                possib_pos.append(temp)
                                return possib_pos
                            elif int(cnt) > 16:
                                return possib_pos
        return possib_pos
    
    def segment(self, img, pos, ii):
        search_size = self.pat_img_r*(1+0)
        search_start_y, search_start_x = max(int(pos[1] - search_size),0), max(int(pos[0] - search_size),0)
        search_end_y, search_end_x = min(int(pos[1] + search_size),img.shape[0]), min(int(pos[0] + search_size),img.shape[1])
        seg_img = copy.deepcopy(img[search_start_y:search_end_y,search_start_x:search_end_x])
        img_gray = seg_img
        thre = img_gray.mean()
        ret, thresh = cv2.threshold(img_gray,thre,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, None
        outer_area = self.outer_area
        outer_r_ratio = self.outer_r_ratio
        area_tolerence = outer_area*0.4
        ratio_tolerence = outer_r_ratio*0.2
        maxdis = 1
        iid=-2

        for i,hier in enumerate(hierarchy[0]):
            if hier[2] == -1 and hier[3] != -1:
                if contours[i].shape[0] > 5 and contours[hierarchy[0][i][3]].shape[0]>5:
                    inner_ellipse = cv2.fitEllipse(contours[i])
                    outer_ellipse = cv2.fitEllipse(contours[hierarchy[0][i][3]])
                    outerArea = cv2.contourArea(contours[hierarchy[0][i][3]])
                    elli_ratio2 = outer_ellipse[1][0]/outer_ellipse[1][1]
                    if abs(outerArea - outer_area) < area_tolerence and abs(elli_ratio2 - outer_r_ratio) < ratio_tolerence:
                        offset = 2
                        if abs(outer_ellipse[1][0]/outer_ellipse[1][1] -1) < 0.1:
                            r0 = (inner_ellipse[1][0] - offset)/(outer_ellipse[1][0] + offset)
                            r1 = (inner_ellipse[1][1] - offset)/(outer_ellipse[1][1] + offset)
                        elif abs(inner_ellipse[2] - outer_ellipse[2]) <= 45 or abs(inner_ellipse[2] - outer_ellipse[2]) >= 135:
                            r0 = (inner_ellipse[1][0] - offset)/(outer_ellipse[1][0] + offset)
                            r1 = (inner_ellipse[1][1] - offset)/(outer_ellipse[1][1] + offset)
                        elif abs(abs(inner_ellipse[2] - outer_ellipse[2]) - 90) < 45:
                            r0 = (inner_ellipse[1][1] - offset)/(outer_ellipse[1][0] + offset)
                            r1 = (inner_ellipse[1][0] - offset)/(outer_ellipse[1][1] + offset)
                        # get id
                        for i, [id, ideal_r1, ideal_r0] in enumerate(self.IDTable):
                            dis = (ideal_r0 - r0)**2 + (ideal_r1 - r1)**2
                            if dis < maxdis:
                                maxdis = dis
                                iid = id
                        # get heading direction
                        if inner_ellipse[0][1] - outer_ellipse[0][1] > 0:
                            angle_ = np.deg2rad(outer_ellipse[-1])
                        else:
                            angle_ = np.deg2rad(outer_ellipse[-1] + 180)
                        return np.array([iid, search_start_x + outer_ellipse[0][0], search_start_y + outer_ellipse[0][1]]).astype(int), angle_
        return None, None

    def trans_corrdi_p2w(self, p2w_M, PresM, p_pos):
        id = p_pos[0]
        p_pos[:-1] = p_pos[1:]
        p_pos[-1] = 1
        w_pos = np.matmul(p2w_M, p_pos)
        w_pos = (w_pos[0]/w_pos[0,3]).tolist()
        w_pos[0].pop(2)
        w_pos = np.matmul(PresM,w_pos[0])
        w_pos = w_pos/w_pos[2]
        w_pos[1:] = w_pos[:-1]
        w_pos[0] = int(id)
        return w_pos

    def search_pattern(self, img_ori):
        possib_pos = self.get_possible_posi(img_ori)
        self.id_cam_pos = []
        self.world_pos = []
        self.heading = []
        for i, pos in enumerate(possib_pos):
            id_cam_pos, h = self.segment(img_ori, pos, i)
            if id_cam_pos is not None:
                self.id_cam_pos.append(copy.deepcopy(id_cam_pos))
                self.world_pos.append(self.trans_corrdi_p2w(self.p2w_M, self.PresM, id_cam_pos))
                self.heading.append(h)
        return self.world_pos, self.id_cam_pos, self.heading

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from scipy.stats import norm
    phero_model = PheromoneModel()
    phero_model.evaporation_factor = np.array([100000,100000,100000])
    phero_model.diffusion_factor = np.array([0.2,0.1,0.2])
    phero_model.d_kernel_s_factor = np.array([51,91,21])
    phero_model.injection_factor = np.array([200.0,200.0,300.0])
    phero_model.radius_factor = np.ones(3).astype('int')*10
    phero_model.update_parameters()
    data = []
    while True:
        img = phero_model.render_pheromone({0:[0.0,0.0],1:[0.2,0.4],2:[0.6,0.5]}, 
                                           {'0':'red','1':'green','2':'blue'}, 
                                           0.8, 0.6)
        cv2.imshow('phero', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        K = cv2.waitKey(10)                # 等待一个键盘的输入
        if K == 27:                       # 若键入ESC后退出
            cv2.destroyAllWindows()       # 销毁我们创建的所有窗口
            break
        time.sleep(0.1)
        print(np.max(img))
        data.append(img[int(phero_model.pixel_height/2),:,0])
    
    # data = img[int(phero_model.pixel_height/2),:,2]
    # plt.plot(data, color="red")
    fig, ax = plt.subplots()
    lines = []
    for d in data:
        l, = ax.plot(d, color="red")
        lines.append([l])
    ani = animation.ArtistAnimation(fig, lines, interval=20)
    
    
    # Gaussian curve fit
    # mu = np.mean(data)
    # sigma = np.std(data)
    # n, bins, patches = plt.hist(data, 30, alpha=0.2)
    # y = norm.pdf(bins, mu, sigma)
    # plt.plot(y)
    plt.grid(True)
    plt.show()



