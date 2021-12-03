import re
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
        self.diffusion_factor = np.zeros(3)
        self.evaporation_factor = np.zeros(3)
        self.injection_factor = np.zeros(3)
        self.radius_factor = np.ones(3).astype('int')
        self.dt = 0.1
        self.update_parameters()
        
        # pheromone
        self.color_channel = {'red':0, 'green':1, 'blue':2}
        self.pheromone_field = np.zeros([self.pixel_height, self.pixel_width, 3])
        
    def update_parameters(self):
        # diffusion_kernel
        self.diffusion_kernel = np.ones([3,3,3])
        for i, d in enumerate(self.diffusion_factor):
            self.diffusion_kernel[i] *= (1-d)/4
            self.diffusion_kernel[i,1,1] = d - 1
            # self.diffusion_kernel[i,0,0],self.diffusion_kernel[i,0,2],self.diffusion_kernel[i,2,0],self.diffusion_kernel[i,2,2] = 0,0,0,0
        self.pheromone_field = np.zeros([self.pixel_height, self.pixel_width, 3])
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
            y = int(v[0]/arena_length*self.pixel_width)
            x = int(v[1]/arena_width*self.pixel_height)
            if str(k) in channel.keys():
                ind = self.color_channel[channel[str(k)]]
            else:
                ind = self.color_channel[channel['other']]
            r = int(self.radius_factor[ind])
            try:
                # print(self.injection_kernel)
                injection[x-r:x+r, y-r:y+r, ind] += self.injection_kernel[ind]
            # get rid of the boundary effect
            except:
                injection[x-r:x+r, y-r:y+r, ind] += 1
        # evaportion
        e = -(1/(self.evaporation_factor*100)) * self.pheromone_field 
        # injection
        i = self.injection_factor * injection
        # diffusion
        # !should separate channels and process individually
        r,g,b = cv2.split(self.pheromone_field)
        r = cv2.filter2D(r, -1, self.diffusion_kernel[0])
        g = cv2.filter2D(g, -1, self.diffusion_kernel[1])
        b = cv2.filter2D(b, -1, self.diffusion_kernel[2])
        d = cv2.merge([r,g,b])
        # update
        self.pheromone_field = self.pheromone_field + (e + i + d) * self.dt
        print(self.pheromone_field.max(), np.sum(injection))
        return np.clip(self.pheromone_field, 0, 255).astype(np.uint8)
    
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