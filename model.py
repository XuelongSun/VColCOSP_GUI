import re
import numpy as np
import matplotlib.pyplot as plt
import cv2

from viewer import PheroBgInfoSetting

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
        
        self.kernel_size = 71
        self.diffusion_kernel = np.ones([3,self.kernel_size,
                                         self.kernel_size])
        sigma = self.kernel_size/2
        s = 2*(sigma**2)
        x0 = int((self.kernel_size-1)/2)
        y0 = int((self.kernel_size-1)/2)
        for n, d in enumerate(self.diffusion_factor):
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    x = i - x0
                    y = j - y0
                    self.diffusion_kernel[n,i,j] = np.exp(-(x**2 + y**2)/s)
            self.diffusion_kernel[n] = self.diffusion_kernel[n] / (np.sum(self.diffusion_kernel[n]) - self.diffusion_kernel[n,x0,y0]) * (1-d)
            self.diffusion_kernel[n,x0,y0] = d - 1
            # self.diffusion_kernel[n] *= (1-d)/120
            # self.diffusion_kernel[n,5,5] = d - 1
            print(np.sum(self.diffusion_kernel[0]))
        self.pheromone_field = np.zeros([self.pixel_height, 
                                         self.pixel_width, 
                                         3])
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
            s_x = np.max([0, x-r])
            s_y = np.max([0, y-r])
            e_x = np.min([x+r, self.pixel_height])
            e_y = np.min([y+r, self.pixel_width])
            s_k_x = 0 if x-r>0 else r-x
            s_k_y = 0 if y-r>0 else r-y
            injection[s_x:e_x, s_y:e_y, ind] += np.array(self.injection_kernel[ind])[s_k_x:e_x-s_x+s_k_x,s_k_y:e_y-s_y+s_k_y]
        # evaportion
        e = -(1/(self.evaporation_factor)) * self.pheromone_field
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

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from scipy.stats import norm
    phero_model = PheromoneModel()
    phero_model.evaporation_factor = np.array([100000,100000,100000])
    phero_model.diffusion_factor = np.array([0.2,0.7,0.2])
    phero_model.injection_factor = np.array([20.0,20.0,30.0])
    phero_model.radius_factor = np.ones(3).astype('int')*16
    phero_model.update_parameters()
    data = []
    while True:
        img = phero_model.render_pheromone({0:[0.4,0.3]}, 
                                           {'0':'red','other':'green'}, 
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