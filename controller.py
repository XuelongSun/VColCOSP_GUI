import sys
import os
import scipy.io as sio
import configparser
from multiprocessing import Process
from itertools import combinations
import cv2
import numpy as np
import struct as st
import time
import serial
import serial.tools.list_ports
import threading
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QColorDialog
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from model import LedImageProcess, PheromoneModel, LocDataModel, SerialDataModel, LocalizationModel,\
    distance
from viewer import LEDScreen, Viewer
from camera import HighFpsCamera

class SocketDataReceiver(QThread):
    singal = pyqtSignal(str)
    
    def __init__(self,socket):
        super().__init__()
        self.socket = socket
        self.socket_is_connect = False
        self.stop = True
        self.loc_data_model = LocDataModel()
        self.loc_data_now = None
        
    def run(self):
        while self.socket_is_connect:
            if not self.stop:
                try:
                    raw_data = self.socket.recv(2048)
                    self.loc_data_now = self.loc_data_model.get_loc_data(raw_data)
                except TimeoutError:
                    print('tcp timeout')
            else:
                pass
                
        
class Controller:
    def __init__(self):
        self.viewer = Viewer()
        
        #* login
        self.viewer.login.pushButton_login.clicked.connect(self.login)
        
        #* main menu
        self.viewer.main_menu.pb_vscene.clicked.connect(self.vscene_show_window)
        self.sys_time = time.time()
        self.sys_timer = QTimer()
        self.sys_timer.timeout.connect(self.system_frequency_task)
        
        #* vscene
        self.vscene_model = LedImageProcess()
        self.viewer.vscene.signal.connect(self.vscene_event_handle)
        self.viewer.vscene.pb_show.clicked.connect(self.vscene_show_screen)
        self.viewer.vscene.pushButton_led_v_loadpic.clicked.connect(self.vscene_load_picture)
        self.viewer.vscene.pb_stop_video.clicked.connect(self.vscene_stop_video)
        self.viewer.vscene.pb_start_video.clicked.connect(self.vscene_start_video)
        self.viewer.vscene.pb_pause_video.clicked.connect(self.vscene_pause_video)
        self.viewer.vscene.pb_load_config.clicked.connect(self.vscene_load_config)
        self.viewer.vscene.pb_save_config.clicked.connect(self.vscene_save_config)
        ## timer for video player
        self.vscene_timer = QTimer()
        self.vscene_timer.timeout.connect(self.vscene_video_player)
        ## default test image
        self.vscene_refresh_parameters()
        self.viewer.vscene.pb_refresh.clicked.connect(self.vscene_refresh_parameters)
        self.vscene_image = self.vscene_model.generate_test_img(self.vscene_img_width,
                                                             self.vscene_img_height,
                                                             self.vscene_fold_row,
                                                             self.vscene_fold_column)
        self.vscene_video_image = None
        ## led screen window
        self.vscene_screen = LEDScreen()
        self.vscene_loaded_image_path = None
        self.vscene_frame_num = 0
        self.vscene_roll = 0
        
        #* pheromone
        self.phero_model = PheromoneModel()
        self.viewer.main_menu.pb_phero.clicked.connect(self.phero_show_window)
        self.viewer.phero.signal.connect(self.phero_event_handle)
        self.viewer.phero.pb_start.clicked.connect(self.phero_start_render)
        self.viewer.phero.pb_stop.clicked.connect(self.phero_stop_render)
        self.viewer.phero.pb_pause.clicked.connect(self.phero_pause_render)
        self.viewer.phero.pb_show.clicked.connect(self.phero_show_pheromone)
        self.viewer.phero.pb_refresh.clicked.connect(self.phero_refresh_parameter)
        self.viewer.phero.pb_load_phero_image.clicked.connect(self.phero_load_picture)
        self.viewer.phero.pb_load_phero_video.clicked.connect(self.phero_load_video)
        self.viewer.phero.pb_save_config.clicked.connect(self.phero_save_config)
        self.viewer.phero.pb_load_config.clicked.connect(self.phero_load_config)
        self.viewer.phero.pb_bg_load_img.clicked.connect(self.phero_load_bg_img)
        self.viewer.phero.pb_screen_shot.clicked.connect(self.phero_screenshot)
        
        self.phero_screen = LEDScreen()
        #* pheromone background settings
        self.phero_background_image = np.zeros([self.phero_model.pixel_height,
                                                self.phero_model.pixel_width, 3]).astype(np.uint8)
        self.phero_bg_sc = (0,0,0) # black
        self.phero_bg_loaded_image = None
        self.phero_bg_drawn_image = None
        self.phero_bg_info_paras = {'pos_text_color':(255,255,255),
                                    'pos_text_width':2,
                                    'pos_line_style':'solid',
                                    'pos_line_color':(125,125,125),
                                    'pos_line_width':2,
                                    'pos_marker_style':'circle',
                                    'pos_marker_width':2,
                                    'pos_marker_color':(0,255,255), 
                                    'pos_marker_radius':30,
                                    'arena_border_color':(255,255,255),
                                    'arena_border_width':4,
                                    'arena_border_margin':2}
        self.viewer.phero.pb_bg_setting.clicked.connect(lambda:self.viewer.phero_bg_setting.show())
        self.viewer.phero_bg_setting.signal.connect(self.phero_bg_setting_message_handle)
        self.phero_image = None
        self.phero_timer = QTimer()
        self.phero_timer.timeout.connect(self.phero_render)
        self.phero_is_rendering = False
        self.phero_loaded_image_path = None
        self.phero_bg_loaded_image_path = None
        self.phero_loaded_video_path = None
        self.phero_video_capture = None
        self.phero_field_display = np.zeros([self.phero_model.pixel_height,
                                             self.phero_model.pixel_width, 3]).astype(np.uint8)
        self.phero_frame_num = 0
        self.arena_length = 0.8
        self.arena_width = 0.6
        
        #* localization
        self.viewer.main_menu.pb_loc.clicked.connect(self.loc_show_window)
        self.loc_model = LocalizationModel()
        self.loc_refresh_timer = QTimer()
        self.loc_refresh_timer.timeout.connect(self.loc_display)
        self.loc_camera = HighFpsCamera.Camera()
        self.loc_camera_opened = False
        self.loc_has_started = False
        self.loc_is_running = False
        self.loc_camera_is_calibrating = False
        self.loc_current_image = None
        self.loc_img_display = None
        self.loc_world_locations = {}
        self.loc_image_location = []
        self.loc_world_location = []
        self.loc_heading = []
        self.viewer.loc_embedded.signal.connect(self.loc_event_handle)
        self.viewer.loc_embedded.pb_check_camera.clicked.connect(self.loc_check_camera)
        self.viewer.loc_embedded.pb_start.clicked.connect(self.loc_start)
        self.viewer.loc_embedded.hS_exposure.valueChanged.connect(self.loc_camera_setting)
        self.viewer.loc_embedded.pb_generate_pattern.clicked.connect(self.loc_generate_pattern)
        self.viewer.loc_embedded.pb_start_calibration.clicked.connect(self.loc_start_camera_calibration)
        self.viewer.loc_embedded.pb_start_capture.clicked.connect(self.loc_open_camera)
        self.viewer.loc_embedded.pb_save_as_img.clicked.connect(self.loc_save_as_imgae)
        self.viewer.loc_embedded.pb_cal_show_default.clicked.connect(self.loc_dispaly_calibration_data)
        # self.viewer.loc_embedded.pb_cal_load_data.clicked.connect(self.loc_save_as_imgae)
        #* communication
        self.viewer.main_menu.pb_com.clicked.connect(lambda: self.viewer.com.show())
        # robot data via USB-serial
        self.serial_data_model = SerialDataModel()
        # data interface
        self.thread_robot_data_capture = Process(target=self.serial_run_capture)
        self.serial_port = serial.Serial(baudrate=115200,
                                         parity=serial.PARITY_NONE,
                                         stopbits=serial.STOPBITS_ONE,
                                         bytesize=serial.EIGHTBITS)
        self.serial_send_package_len = 32
        self.serial_recv_package_len = 96
        self.serial_port_timer = QTimer()
        self.robot_data_buff = []
        self.serial_data_is_running = False
        self.valid_robot_ids = []
        self.viewer.com.pb_scan_port.clicked.connect(self.serial_ports_scan)
        self.viewer.com.pb_open_port.clicked.connect(self.serial_port_open)
        self.viewer.com.pb_close_port.clicked.connect(self.serial_port_close)
        self.robot_motion_ctl_table = {'forward':'MFD', 'backward':'MBK', 'left':'MML', 'right': 'MMR',
                                       'start': 'MST', 'stop':'MSP'}
        for k in self.robot_motion_ctl_table.keys():
            eval('self.viewer.com.pb_motion_' + k + '.clicked.connect(self.serial_send_motion)')
        self.viewer.com.pb_start_capture.clicked.connect(self.serial_start_capture)
        self.viewer.com.pb_request_update.clicked.connect(self.serial_request_data)
        self.viewer.com.pb_raw_send.clicked.connect(self.serial_send_raw_data)
        self.viewer.com.pb_raw_clear.clicked.connect(lambda:self.viewer.com.text_edit_recv_raw.clear())
        self.viewer.com.pb_send_ch_data.clicked.connect(self.serial_port_send_parameter)
        self.viewer.com.pb_clear_cache.clicked.connect(self.serial_clear_data_cached)
        
        # visualization plots
        self.viewer.main_menu.pb_add_plot.clicked.connect(self.exp_visualization_add_plot)
        self.viewer.main_menu.pb_add_map.clicked.connect(self.exp_visualization_add_plot)
        self.viewer.main_menu.pb_add_distribution.clicked.connect(self.exp_visualization_add_plot)
        
        #* experiment
        self.exp_start_time = time.time()
        self.exp_is_running = False
        self.exp_thread = threading.Thread(target=self.exp_task)
        # experiment data
        self.exp_available_data_key = []
        self.exp_detected_robot_ids = []
        ## data saving
        self.exp_save_data_selected_id = []
        self.exp_save_data_selected_data = []
        self.exp_data_to_save = []
        self.exp_save_data_file_type = ".txt"
        self.exp_save_data_max_l = 500
        self.exp_save_data_interval = 500
        self.viewer.main_menu.pb_save_data_setting.clicked.connect(self.exp_save_data_setting)
        self.viewer.data_save_setting.answer.connect(self.exp_save_data_setting_update)
        
        self.viewer.main_menu.pb_start_exp.clicked.connect(self.exp_start)
        self.viewer.main_menu.pb_save_data.clicked.connect(self.exp_save_data)
        
        self.viewer.main_menu.pb_load_config.clicked.connect(self.exp_load_config)
        self.viewer.main_menu.pb_save_config.clicked.connect(self.exp_save_config)
        
        self.sys_timer.start(100)
        
    def system_frequency_task(self):
        # 1.system time
        t = time.time() - self.sys_time
        h = int(t / 3600)
        m = int((t - h * 3600) / 60)
        s = t - h * 3600 - m * 60
        self.viewer.main_menu.et_sys_timer.setText('SystemTime: %2d H %2d M% 2.2f S' % (h, m, s))
        
        # 2.update robot ids and data
        self.exp_available_data_key = []
        self.exp_detected_robot_ids = []
        ids_from_loc = []
        ids_from_com = []
        ## ids from localization
        if self.loc_world_locations:
            ids_from_loc = list(self.loc_world_locations.keys())
            self.exp_available_data_key += ['POS_X', 'POS_Y']

        ## ids from communication
        if self.serial_data_is_running:
            ids_from_com = list(self.serial_data_model.robot_data.keys())
            self.exp_available_data_key += list(self.serial_data_model.data_str_table.keys())
            
        ## merge
        if self.loc_world_locations and self.serial_data_is_running:
            self.exp_detected_robot_ids = ids_from_com.copy()
        elif self.loc_world_locations:
            self.exp_detected_robot_ids = ids_from_loc.copy()
        else:
            self.exp_detected_robot_ids = ids_from_com.copy()
        
        # 3.update localization data counter
        if self.serial_data_is_running:
            if ids_from_com:
                s = '%d robots, %d frame have received\n' % (len(ids_from_com),
                                                            self.serial_data_model.num_package)
                self.viewer.com.et_data_flow_info.setText(s)
        else:
            s = 'Waiting for data...'
            self.viewer.com.et_data_flow_info.setText(s)
        
        # 4. update visualization plots
        for plot in self.viewer.plots:
            if plot.type == 'plot':
                for k, v in plot.lines.items():
                    _id, ds = k.split('/')
                    r_id = _id[-1]
                    if ds == 'POS_X':
                        v.setData(x=np.arange(len(self.loc_world_locations[int(r_id)])),
                                y=np.array(self.loc_world_locations[int(r_id)], dtype=np.float)[:,0])
                    elif ds == 'POS_Y':
                        v.setData(x=np.arange(len(self.loc_world_locations[int(r_id)])),
                                y=np.array(self.loc_world_locations[int(r_id)], dtype=np.float)[:,1])
                    else:
                        d = self.serial_data_model.get_robot_data(int(r_id), ds)
                        v.setData(x=np.arange(len(d)),
                                  y=np.array(d))
            elif plot.type == 'map':
                data = []
                for k, v in plot.texts.items():
                    x = self.loc_world_locations[int(k)][-1][0]
                    y = self.loc_world_locations[int(k)][-1][1]
                    data.append((x,y))
                    v.setPos(x, y)
                plot.scatter_plot.setData(pos=data)
            
            elif plot.type == 'distribution':
                for k, v in plot.bars.items():
                    d = self.serial_data_model.get_robots_data(k, t=-1)
                    x, h = np.histogram(np.array([v_ for k_, v_ in d.items()]), bin=20)
                    v.setOpts(x=x[1:], height=h, width=0.2)
        
    def login(self):
        self.viewer.login.close()
        self.viewer.main_menu.show()
    
    def load_picture(self, window):
        filename, _ = QFileDialog.getOpenFileName(window, 'Load Picture', 
                                                  './', 'Picture File(*.jpg; *.png)')
        if filename:
            print(filename)
            return filename
        else:
            return None
        
    def vscene_show_window(self):
        self.viewer.main_menu.pb_vscene.setDisabled(True)
        self.viewer.vscene.show()
    
    def vscene_event_handle(self, signal):
        if signal == "close":
            self.viewer.main_menu.pb_vscene.setDisabled(False)
            self.vscene_screen.hide()

    def vscene_show_screen(self):
        if (self.viewer.vscene.pb_show.text() == "Show") and (self.vscene_image is not None):
            self.vscene_screen.show_label_img(self.vscene_screen_start_pos[0],
                                              self.vscene_screen_start_pos[1],
                                              self.vscene_img_width,
                                              self.vscene_img_height,
                                              self.vscene_image)
            self.viewer.vscene.pb_show.setText("Hide")
            self.vscene_screen.show()
        elif self.viewer.vscene.pb_show.text() == "Hide":
            self.viewer.vscene.pb_show.setText("Show")
            self.vscene_screen.hide()
        else:
            pass
    
    def vscene_refresh_parameters(self):
        self.vscene_led_unit_width = self.viewer.vscene.spinBox_sled_w.value()
        self.vscene_led_unit_height = self.viewer.vscene.spinBox_sled_h.value()
        self.vscene_led_unit_row = self.viewer.vscene.spinBox_sled_r.value()
        self.vscene_led_unit_column = self.viewer.vscene.spinBox_sled_c.value()
        self.vscene_screen_start_pos = [int(self.viewer.vscene.spinBox_sp_x.value()),
                                        int(self.viewer.vscene.spinBox_sp_y.value())]
        self.vscene_fold_row = int(self.viewer.vscene.spinBox_sled_fold_r.value())
        self.vscene_fold_column = int(self.viewer.vscene.spinBox_sled_fold_c.value())
        self.vscene_img_width = int(self.vscene_led_unit_width * self.vscene_led_unit_row)
        self.vscene_img_height = int(self.vscene_led_unit_height * self.vscene_led_unit_column)

        self.vscene_display_mode = self.viewer.vscene.comboBox_led_mode.currentText()
        self.vscene_video_image = None
        if self.vscene_display_mode == "Default":
            self.vscene_image = self.vscene_model.generate_test_img(self.vscene_img_width,
                                                self.vscene_img_height,
                                                self.vscene_fold_row,
                                                self.vscene_fold_column)
        elif self.vscene_display_mode == "Video":
            if self.viewer.vscene.tabWidget_led_v.currentIndex() == 0:
                #* grating
                strip_w = self.viewer.vscene.spinBox_led_v_grating_sw.value()
                speed = self.viewer.vscene.spinBox_led_v_grating_speed.value()
                self.vscene_image = self.vscene_model.generate_grating(self.vscene_img_width,
                                                                    self.vscene_img_height,
                                                                    self.vscene_fold_row, 
                                                                    self.vscene_fold_column,
                                                                    strip_w, speed, 
                                                                    half_cover=True)
            elif self.viewer.vscene.tabWidget_led_v.currentIndex() == 1:
                #* load image
                if self.vscene_loaded_image_path is not None:
                    image = cv2.imread(self.vscene_loaded_image_path)
                    if image is None:
                        QMessageBox.warning(self.viewer.vscene, 
                                            'Error', 
                                            'Cannot load image:{}'.format(self.vscene_loaded_image_path))
                    else:
                        gray = self.viewer.vscene.checkBox_led_v_gray.isChecked()
                        self.vscene_image = self.vscene_model.transfer_loaded_image(image,
                                                                                 self.vscene_img_width,
                                                                                 self.vscene_img_height,
                                                                                 self.vscene_fold_row,
                                                                                 self.vscene_fold_column,
                                                                                 gray=gray)
                else:
                    QMessageBox.warning(self.viewer.vscene, 
                                        'ERROR', 
                                        'No Picture loaded, please load picture first.')
            else:
                #TODO: other modes
                self.vscene_image = None
        if (self.viewer.vscene.pb_show.text() == "Hide") and (self.vscene_image is not None):
            self.vscene_screen.show_label_img(self.vscene_screen_start_pos[0],
                                           self.vscene_screen_start_pos[1],
                                           self.vscene_img_width,
                                           self.vscene_img_height,
                                           self.vscene_image)
            
    def vscene_load_picture(self):
        self.vscene_loaded_image_path = self.load_picture(self.viewer.vscene) 
        
    def vscene_video_player(self):
        self.vscene_frame_rate = self.viewer.vscene.spinBox_frame_rate.value()
        if self.vscene_display_mode == "Video":
            #* grating
            if self.viewer.vscene.tabWidget_led_v.currentIndex() == 0:
                strip_w = self.viewer.vscene.spinBox_led_v_grating_sw.value()
                speed = self.viewer.vscene.spinBox_led_v_grating_speed.value()
                roll = (speed / self.vscene_frame_rate) * self.vscene_frame_num
                self.vscene_image = self.vscene_model.generate_grating(self.vscene_img_width,
                                                                    self.vscene_img_height,
                                                                    self.vscene_fold_row, 
                                                                    self.vscene_fold_column,
                                                                    strip_w, speed,
                                                                    roll=roll,
                                                                    half_cover=False)
            #* loaded image
            elif self.viewer.vscene.tabWidget_led_v.currentIndex() == 1:
                if self.vscene_video_image is None:
                    if self.vscene_loaded_image_path is not None:
                        image = cv2.imread(self.vscene_loaded_image_path)
                        if image is None:
                            QMessageBox.warning(self.viewer.vscene, 
                                                'Error', 
                                                'Cannot load image:{}'.format(self.vscene_loaded_image_path))
                            self.vscene_timer.stop()
                        else:
                            gray = self.viewer.vscene.checkBox_led_v_gray.isChecked()
                            self.vscene_video_image = self.vscene_model.transfer_loaded_image(image,
                                                                                           self.vscene_img_width,
                                                                                           self.vscene_img_height,
                                                                                           self.vscene_fold_row,
                                                                                           self.vscene_fold_column,
                                                                                           gray=gray, fold=False)
                    else:
                        QMessageBox.warning(self.viewer.vscene, 
                                            'ERROR', 
                                            'No Picture loaded, please load picture first.')
                        self.vscene_timer.stop()
                else:
                    if self.viewer.vscene.rb_led_v_animate.isChecked():
                        # animated
                        if self.viewer.vscene.rb_c_roll.isChecked():
                            # constant roll
                            self.vscene_roll = (self.viewer.vscene.sb_led_v_r_speed.value()/self.vscene_frame_rate) * self.vscene_frame_num
                        elif self.viewer.vscene.rb_r_pos.isChecked():
                            # random positioned
                            ctime = self.viewer.vscene.sb_r_time.value()
                            if (self.vscene_frame_num * (1/self.vscene_frame_rate))%(ctime) <= 1/self.vscene_frame_rate:
                                self.vscene_roll = np.random.randint(0, self.vscene_img_width*(self.vscene_fold_row+1)-1)
                        else:
                            QMessageBox.warning(self.viewer.vscene, 
                                            'ERROR', 
                                            'Please choose animation option then restart video.')
                            self.vscene_timer.stop()
                        
                        self.vscene_image = self.vscene_model.generate_video(self.vscene_video_image,
                                                            self.vscene_img_width,
                                                            self.vscene_img_height,
                                                            self.vscene_fold_row,
                                                            self.vscene_fold_column,
                                                            animation={'roll':int(self.vscene_roll)})
            else:
                #TODO: other video modes
                self.vscene_image = None
                
        else:
            QMessageBox.warning(self.viewer.vscene, 
                                'Error', 
                                'Not in the Video Mode', 
                                QMessageBox.Yes, QMessageBox.Yes)
            self.vscene_timer.stop()
        self.vscene_frame_num += 1
        if (self.viewer.vscene.pb_show.text() == "Hide") and (self.vscene_image is not None):
            self.vscene_screen.show_label_img(self.vscene_screen_start_pos[0],
                                           self.vscene_screen_start_pos[1],
                                           self.vscene_img_width,
                                           self.vscene_img_height,
                                           self.vscene_image)
        
    def vscene_start_video(self):
        self.vscene_frame_rate = self.viewer.vscene.spinBox_frame_rate.value()
        self.vscene_timer.start(int(1000/self.vscene_frame_rate))
    
    def vscene_stop_video(self):
        self.vscene_frame_num = 0
        self.vscene_timer.stop()
    
    def vscene_pause_video(self):
        self.vscene_timer.stop()
    
    def _vscene_save_config(self, config=None):
        if config is None:
            config = configparser.ConfigParser()
        config.add_section('VScene')
        # get config and write to the config file
        config_dict = {'mode':str(self.viewer.vscene.comboBox_led_mode.currentIndex())}
        for name in ['sled_w','sled_h','sled_r','sled_c',
                        'sp_x','sp_y','sled_fold_c','sled_fold_r',
                        'frame_rate','pc_display_rate']:
            exec('config_dict[name] = str(self.viewer.vscene.spinBox_{}.value())'.format(name))
        for k,v in config_dict.items():
            config.set('VScene', k, v)
        return config

    def vscene_save_config(self):
        filename, _ = QFileDialog.getSaveFileName(self.viewer.vscene, 
                                                 'save pheromone config',
                                                 './', 'ini(*.ini)')
        if len(filename) != 0:
            cfgfile = open(filename, 'w')
            config = self._vscene_save_config()
            config.write(cfgfile)
            cfgfile.close()
            self.viewer.system_logger('Save config {} successfully'.format(filename))
        else:
            QMessageBox.warning(self.viewer.vscene, 'Error',
                                'Not a valid filename!')
    
    def _set_vscene_config(self, config):
        if config.has_section('VScene'):
            options = config.options('VScene')
            if 'mode' in options:
                mode = config.getint('VScene', 'Mode')
                self.viewer.vscene.comboBox_led_mode.setCurrentIndex(mode)
            for name in ['sled_w','sled_h','sled_r','sled_c',
                            'sp_x','sp_y','sled_fold_c','sled_fold_r',
                            'frame_rate','pc_display_rate']:
                if name in options:
                    value = config.getint('VScene', name)
                    eval('self.viewer.vscene.spinBox_{}.setValue(value)'.format(name))
        else:
            self.viewer.system_logger('No VScene Section Found', 'warning')
            
    def vscene_load_config(self):
        filename, _ = QFileDialog.getOpenFileName(self.viewer.vscene, 
                                                  'Load config File', './')
        if filename:
            self.viewer.system_logger('Successful loaded config file:{}'.format(filename))
            Config = configparser.ConfigParser()
            Config.read(filename)
            # set values
            self._set_vscene_config(Config)

    def phero_show_window(self):
        self.viewer.main_menu.pb_phero.setDisabled(True)
        self.viewer.phero.show()
    
    def phero_event_handle(self,signal):
        if signal == "close":
            self.viewer.main_menu.pb_phero.setDisabled(False)
            self.viewer.phero.hide()
            self.phero_screen.hide()
            # kill pheromone rendering thread
            if self.phero_is_rendering:
                self.phero_is_rendering = False
    
    def phero_bg_setting_message_handle(self,message):
        if message == "OK":
            # update phero background param
            self.phero_bg_info_paras['pos_text_color'] = self.viewer.phero_bg_setting.pos_text_color
            self.phero_bg_info_paras['pos_line_color'] = self.viewer.phero_bg_setting.pos_line_color
            self.phero_bg_info_paras['pos_marker_color'] = self.viewer.phero_bg_setting.pos_marker_color
            self.phero_bg_info_paras['pos_marker_radius'] = self.viewer.phero_bg_setting.sb_pos_marker_radius.value()
            self.phero_bg_info_paras['arena_border_color'] = self.viewer.phero_bg_setting.arena_border_color
            self.phero_bg_info_paras['pos_text_width'] = self.viewer.phero_bg_setting.sb_pos_text_width.value()
            self.phero_bg_info_paras['pos_line_width'] = self.viewer.phero_bg_setting.sb_pos_line_width.value()
            self.phero_bg_info_paras['pos_marker_width'] = self.viewer.phero_bg_setting.sb_pos_marker_width.value()
            self.phero_bg_info_paras['arena_border_width'] = self.viewer.phero_bg_setting.sb_arena_border_width.value()
            self.phero_bg_info_paras['arena_border_margin'] = self.viewer.phero_bg_setting.sb_arena_border_margin.value()
            self.viewer.phero_bg_setting.hide()
        elif message == "Cancel":
            self.viewer.phero_bg_setting.hide()
        else:
            pass
        
    def phero_refresh_parameter(self):
        self.phero_led_unit_width = self.viewer.phero.spinBox_sled_w.value()
        self.phero_led_unit_height = self.viewer.phero.spinBox_sled_h.value()
        self.phero_led_unit_row = self.viewer.phero.spinBox_sled_r.value()
        self.phero_led_unit_column = self.viewer.phero.spinBox_sled_c.value()
        self.phero_screen_start_pos = [int(self.viewer.phero.spinBox_sp_x.value()),
                                       int(self.viewer.phero.spinBox_sp_y.value())]
        self.phero_img_width = int(self.phero_led_unit_width * self.phero_led_unit_row)
        self.phero_img_height = int(self.phero_led_unit_height * self.phero_led_unit_column)
        self.phero_screen.set_window_position(self.phero_screen_start_pos[0],
                                              self.phero_screen_start_pos[1],
                                              self.phero_img_width,
                                              self.phero_img_height,)
        self.phero_mode = self.viewer.phero.comboBox_led_mode.currentText()
        
        self.phero_model.dt = 1.0/self.viewer.phero.spinBox_frame_rate.value()
        
        self.arena_length = self.viewer.phero.spinBox_arena_l.value()
        self.arena_width = self.viewer.phero.spinBox_arena_w.value()
        
        self.phero_image = None
        
        if self.phero_mode == "Static":
            #* static: just show the loaded picture
            if self.phero_loaded_image_path is None:
                QMessageBox.warning(self.viewer.phero,
                                    'Error',
                                    'Please load image first.')
            else:
                image = cv2.imread(self.phero_loaded_image_path)
                if image is None:
                    QMessageBox.warning(self.viewer.vscene, 
                                        'Error', 
                                        'Cannot load image:{}'.format(self.phero_loaded_image_path))
                else:
                    gray = self.viewer.phero.cb_img_gray.isChecked()
                    self.phero_image = self.phero_model.transform_loaded_image(image,
                                                                                self.phero_img_width,
                                                                                self.phero_img_height,
                                                                                gray=gray)
                    self.viewer.phero.show_label_image(self.viewer.phero.label_image, 
                                                       self.phero_image)
                    p, f = os.path.split(self.phero_loaded_image_path)
                    self.viewer.phero.label_img_filename.setText(f)
                    
        elif self.phero_mode == "Video":
            #* video: show the first frame of a video
            if self.phero_video_capture is None:
                QMessageBox.warning(self.viewer.phero,
                                    'Error',
                                    'Please load video first.')
            else:
                self.phero_video_capture.set(1, 0)
                ret, frame = self.phero_video_capture.read()
                self.phero_video_t_frame = int(self.phero_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if ret:
                    gray = self.viewer.phero.cb_video_gray.isChecked()
                    self.phero_image = self.phero_model.transform_loaded_image(frame,
                                                                               self.phero_img_width,
                                                                               self.phero_img_height,
                                                                               gray=gray)
                    if self.viewer.phero.cb_video_preview.isChecked():
                        self.viewer.phero.show_label_image(self.viewer.phero.label_video, 
                                                           self.phero_image)
                    p, f = os.path.split(self.phero_loaded_video_path)
                    self.viewer.phero.label_video_filename.setText(f)
        elif self.phero_mode == "Dy-Localization":
            self.arena_length = self.viewer.phero.spinBox_arena_l.value()
            self.arena_width = self.viewer.phero.spinBox_arena_w.value()
            self.phero_model.pixel_width = self.phero_img_width
            self.phero_model.pixel_height = self.phero_img_height
            self.phero_frame_rate = self.viewer.phero.spinBox_frame_rate.value()
            self.phero_model.dt =  1/self.phero_frame_rate
            # pheromone parameters
            for p in ['diffusion','evaporation','injection','radius','d_kernel_s']:
                s = """self.phero_model.{}_factor = np.array([self.viewer.phero.sp_{}_r.value(), 
                     self.viewer.phero.sp_{}_g.value(),
                     self.viewer.phero.sp_{}_b.value()])""".format(p,p,p,p)
                exec(s)
            self.phero_model.update_parameters()
            # channel
            phero_channel = {}
            ch_str = self.viewer.phero.te_diversity.toPlainText()
            if len(ch_str) == 0:
                QMessageBox.warning(self.viewer.phero, 'Error','Please define diversity.')
                return
            for s in ch_str.split(';'):
                id_s = s.split(':')[0]
                if id_s.isnumeric():
                    # '2':red
                    phero_channel.update({id_s:s.split(':')[1]})
                elif ',' in id_s:
                    # '1,3:blue'
                    for id_ in id_s.split(','):
                        if '-' in id_:
                            for i in range(int(id_.split('-')[0]),int(id_.split('-')[1])+1):
                                phero_channel.update({str(i):s.split(':')[1]})
                        else:
                            phero_channel.update({id_:s.split(':')[1]})
                elif '-' in id_s:
                    # '2-5:red'
                    for i in range(int(id_s.split('-')[0]),int(id_s.split('-')[1])+1):
                        phero_channel.update({str(i):s.split(':')[1]})
                elif id_s == 'other':
                    phero_channel.update({'other':s.split(':')[1]})
                else:
                    QMessageBox.warning(self.viewer.phero, 'Error', 'Not a valid Diversity define, please check!')
            if 'other' not in ch_str:
                phero_channel.update({'other':'red'})
                
            self.phero_channel = phero_channel
            # pheromone background
            if self.viewer.phero.radioButton_sc.isChecked():
                # solid color fill
                self.phero_background_image = np.ones([self.phero_model.pixel_height,
                                                        self.phero_model.pixel_width, 3],
                                                      dtype=np.uint8)*self.phero_bg_sc
            elif self.viewer.phero.radioButton_image.isChecked():
                self.phero_background_image = self.phero_bg_loaded_image.copy()
            elif self.viewer.phero.radioButton_draw.isChecked():
                self.phero_background_image = self.phero_bg_drawn_image.copy()
            elif self.viewer.phero.radioButton_info.isChecked():
                # dynamically updated in phero_render()
                pass
        
        elif self.phero_mode == 'Loc-Calibration':
            # self.phero_image = self.phero_model.generate_calibration_pattern(self.arena_length)
            c_c = self.viewer.loc_embedded.sp_chessboard_c.value()
            c_r = self.viewer.loc_embedded.sp_chessboard_r.value()
            offsety = int(self.viewer.loc_embedded.sp_cal_offset_y.value()/self.arena_width*self.phero_img_height)
            offsetx = int(self.viewer.loc_embedded.sp_cal_offset_x.value()/self.arena_length*self.phero_img_width)
            chess_size = min([self.phero_img_height//c_r, self.phero_img_width//c_c])
            border = self.viewer.loc_embedded.sp_cal_border_w.value()
            self.phero_image = self.loc_model.draw_chess_board((self.phero_img_height, self.phero_img_width),
                                                               c_c, c_r, chess_size, offsetx, offsety, border)
        
        # if no pheromone image defined, generate onr
        if self.phero_image is None:
            self.phero_image = np.zeros([self.phero_img_height, self.phero_img_width, 3], dtype=np.uint8)
            str_para = {'font':cv2.FONT_HERSHEY_SIMPLEX,
                        'font_scale':self.phero_img_width/960,
                        'thickness':1,
                        'text':"Welcome to VColCOSP, Happy Swarming! ^V^ "}
            str_size1 = cv2.getTextSize(str_para['text'], str_para['font'], str_para['font_scale'], str_para['thickness'])[0]
            self.phero_image = cv2.putText(self.phero_image, str_para['text'],
                                           ((self.phero_img_width - str_size1[0])//2, (self.phero_img_height - str_size1[1])//2),
                                           str_para['font'],
                                           str_para['font_scale'],
                                           (255, 255, 0),
                                           str_para['thickness'])
            str_para['text'] = "Press [Start] to start rendering dynamically or select static mode..."
            str_para['font_scale'] = self.phero_img_width/1920
            str_size2 = cv2.getTextSize(str_para['text'], str_para['font'], str_para['font_scale'], str_para['thickness'])[0]
            self.phero_image = cv2.putText(self.phero_image, str_para['text'],
                                           ((self.phero_img_width - str_size2[0])//2, (self.phero_img_height + str_size2[1])//2 + str_size1[1]),
                                           str_para['font'], str_para['font_scale'],
                                           (255, 255, 255), str_para['thickness'])
            
        if (self.viewer.phero.pb_show.text() == "Hide") and (self.phero_image is not None):
            self.phero_screen.show_label_img(self.phero_image)
        
    def phero_load_picture(self):
        self.phero_loaded_image_path = self.load_picture(self.viewer.phero)
    
    def phero_load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self.viewer.phero, 
                                                  'Load Video', 
                                                  './', 'Video File(*.mp4;)')
        if filename:
            print(filename)
            self.phero_loaded_video_path = filename
            self.phero_video_capture = cv2.VideoCapture(filename)
            self.phero_video_t_frame = int(self.phero_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def phero_load_bg_img(self):
        self.phero_bg_loaded_image_path = self.load_picture(self.viewer.phero)

    def phero_show_pheromone(self):
        if self.phero_image is None:
            self.phero_refresh_parameter()
        if self.viewer.phero.pb_show.text() == "Show":
            self.phero_screen.show_label_img(self.phero_image)
            self.phero_screen.show()
            self.viewer.phero.pb_show.setText("Hide")
        else:
            self.phero_screen.hide()
            self.viewer.phero.pb_show.setText("Show")
    
    def phero_compute_pheromone(self):
        while self.phero_is_rendering:
            if self.phero_mode == "Dy-Localization":
                if self.loc_is_running:
                    info = {}
                    for v in self.loc_world_location:
                        # id,x,y,h
                        info.update({int(v[0]):[v[1],v[2],v[3]]})
                    if self.viewer.phero.radioButton_info.isChecked():
                        image = np.zeros([self.phero_model.pixel_height,
                                self.phero_model.pixel_width, 3], dtype=np.uint8)
                        # arena border
                        if self.viewer.phero_bg_setting.groupBox_arena_border.isChecked():
                            m = self.phero_bg_info_paras['arena_border_margin']
                            image = cv2.rectangle(image,
                                                    (m,m),
                                                    (self.phero_model.pixel_width-m, 
                                                    self.phero_model.pixel_height-m),
                                                    self.phero_bg_info_paras['arena_border_color'],
                                                    self.phero_bg_info_paras['arena_border_width'])
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        for k, v in info.items():
                            if (v[0] >= 0) and (v[0] <= self.arena_length) and \
                                (v[1]>=0) and (v[1] <= self.arena_width):
                                y = int(v[0]/self.arena_length*self.phero_model.pixel_width)
                                x = int(v[1]/self.arena_width*self.phero_model.pixel_height)
                                # offset by the height
                                y += int((self.phero_model.pixel_width/2 - y)*0.023)
                                x += int((self.phero_model.pixel_height/2 - x)*0.023)
                                # pos-text
                                if self.viewer.phero_bg_setting.groupBox_pos_text.isChecked():
                                    image = cv2.putText(image, 
                                                        "{:d}:({:.2f},{:.2f},{:.2f})".format(k,v[0],v[1],np.rad2deg(v[2])),
                                                        (y+self.phero_bg_info_paras['pos_marker_radius']+2,
                                                        x+self.phero_bg_info_paras['pos_marker_radius']+2),
                                                        font,0.5,
                                                        self.phero_bg_info_paras['pos_text_color'],1)
                                # pos-cross-line
                                if self.viewer.phero_bg_setting.groupBox_pos_cline.isChecked():
                                    image = cv2.line(image, (y, 0), (y, self.phero_model.pixel_height),
                                                        self.phero_bg_info_paras['pos_line_color'],
                                                        self.phero_bg_info_paras['pos_line_width'])
                                    image = cv2.line(image, (0, x), (self.phero_model.pixel_width, x),
                                                        self.phero_bg_info_paras['pos_line_color'],
                                                        self.phero_bg_info_paras['pos_line_width'])
                                # marker
                                if self.viewer.phero_bg_setting.groupBox_pos_marker.isChecked():
                                    image = cv2.circle(image, (y,x),
                                                        int(self.phero_bg_info_paras['pos_marker_radius']),
                                                        self.phero_bg_info_paras['pos_marker_color'],
                                                        self.phero_bg_info_paras['pos_marker_width'])
                                    image = cv2.arrowedLine(image, (y, x),
                                                            (int(y + (self.phero_bg_info_paras['pos_marker_radius']+10)*np.cos(v[2])),
                                                            int(x + (self.phero_bg_info_paras['pos_marker_radius']+10)*np.sin(v[2]))),
                                                            self.phero_bg_info_paras['pos_marker_color'],
                                                            4)
                            else:
                                self.viewer.system_logger('Invalid position value from LOCALIZATION:({},{}) of ID:({})'.format(v[0],v[1],k),
                                                        log_type='warning')                    
                        self.phero_background_image = image.copy()
                    elif self.viewer.phero.radioButton_image.isChecked():
                        # use background image
                        if self.phero_bg_loaded_image_path is None:
                            QMessageBox.warning(self.viewer.phero, 
                                                'Error', 
                                                'Please load background image first!')
                            self.phero_timer.stop()
                        else:
                            image = cv2.imread(self.phero_bg_loaded_image_path)
                        if image is None:
                            QMessageBox.warning(self.viewer.vscene, 
                                                'Error', 
                                                'Cannot load image:{}'.format(self.phero_bg_loaded_image_path))
                            self.phero_timer.stop()
                        else:
                            self.phero_background_image = image.copy()
                    elif self.viewer.phero.radioButton_sc.isChecked():
                        # use singel color
                        pass
                    elif self.viewer.phero.radioButton_draw.isChecked():
                        # drawing a background image
                        pass
                    else:
                        pass
                    
                    # if 'only-background' checked
                    if self.viewer.phero.cb_bg_only.isChecked():
                        phero_image = np.zeros([self.phero_model.pixel_height, 
                                                self.phero_model.pixel_width, 3]).astype(np.uint8)
                    else:
                        # phero_image = self.phero_field_display.copy()
                        phero_image = self.phero_model.render_pheromone(info, self.phero_channel, self.arena_length, self.arena_width)
                    #     try:
                    #         phero_image = self.phero_model.render_pheromone(info, self.phero_channel,
                    #                                                     self.arena_length, self.arena_width)
                    #     except:
                    #         QMessageBox.warning(self.viewer.phero, 'error', 'Please try to refresh parameters.')
                    #         self.phero_stop_render()
                    if self.phero_background_image is None:
                        QMessageBox.warning(self.viewer.phero,'Error','No valide background image!')
                        self.phero_timer.stop()
                        return
                    else:
                        # merge the background and pheromone, pheromone z-index is lower
                        img_temp = (phero_image/np.max(phero_image)*255).astype(np.uint8)
                        mask = cv2.bitwise_and(cv2.cvtColor(img_temp,cv2.COLOR_RGB2GRAY),
                                            cv2.cvtColor(self.phero_background_image,cv2.COLOR_RGB2GRAY))
                        if np.sum(mask) > 0:
                            phero_image[np.where(mask>0)] = self.phero_background_image[np.where(mask>0)]
                            phero_image[np.where(mask<=0)] = self.phero_background_image[np.where(mask<=0)] + phero_image[np.where(mask<=0)]
                        self.phero_image = phero_image.copy()
                    # time.sleep(1/self.phero_frame_rate)
                else:
                    self.phero_stop_render()
                    break

            elif self.phero_mode == "Video":
                 if self.phero_video_capture is None:
                     QMessageBox.warning(self.viewer.phero,
                                         'Error',
                                         'Please load video first.')
                     self.phero_timer.stop()
                     self.phero_is_rendering = False
                     break
                 else:
                    self.phero_video_capture.set(1, self.phero_frame_num)
                    ret, frame = self.phero_video_capture.read()
                    if ret:
                        gray = self.viewer.phero.cb_video_gray.isChecked()
                        self.phero_image = self.phero_model.transform_loaded_image(frame,
                                                                                self.phero_img_width,
                                                                                self.phero_img_height,
                                                                                gray=gray)
                    time.sleep(1/self.phero_frame_rate)
    
    def phero_render(self):
        # if self.phero_mode == "Static":
        #     #* static: just show the loaded picture
        #     if self.phero_image is None:
        #         if self.phero_loaded_image_path is None:
        #             QMessageBox.warning(self.viewer.phero,
        #                                 'Error',
        #                                 'Please load image first.')
        #             self.phero_timer.stop()
        #         else:
        #             image = cv2.imread(self.vscene_loaded_image_path)
        #             if image is None:
        #                 QMessageBox.warning(self.viewer.vscene, 
        #                                     'Error', 
        #                                     'Cannot load image:{}'.format(self.vscene_loaded_image_path))
        #                 self.phero_timer.stop()
        #             else:
        #                 gray = self.viewer.phero.cb_img_gray.isChecked()
        #                 self.phero_image = self.phero_model.transform_loaded_image(image,
        #                                                                            self.phero_img_width,
        #                                                                            self.phero_img_height,
        #                                                                            gray=gray)
        # elif self.phero_mode == 'Loc-Calibration':
        # # for localization camera calibration
        #     c_w = self.viewer.loc_embedded.sp_chessboard_w.value()
        #     c_h = self.viewer.loc_embedded.sp_chessboard_h.value()
        #     self.phero_image = self.loc_model.draw_chess_board(c_w, c_h, int(self.phero_img_height/20))

        if self.phero_mode == "Video":
            #* video: play the video
            if self.viewer.phero.cb_video_preview.isChecked():
                self.viewer.phero.show_label_image(self.viewer.phero.label_video,
                                                    self.phero_image)
            p, f = os.path.split(self.phero_loaded_video_path)
            self.viewer.phero.label_video_filename.setText("{}: {}/{}".format(f, 
                                                                            self.phero_frame_num, 
                                                                            self.phero_video_t_frame))
        elif self.phero_mode == "Dy-Localization":
            # * dynamically interact with the localization system
            if not self.loc_is_running:
                # the latest frame data
                # pos = self.loc_data_thread.loc_data_model.get_last_pos()
                QMessageBox.warning(self.viewer.phero,'Error','Please read the localization data first!')
                self.phero_timer.stop()

        elif self.phero_mode == "Dy-Customized":
            #* user defined
            pass

        self.phero_frame_num += 1
        
        if (self.viewer.phero.pb_show.text() == "Hide") and (self.phero_image is not None):
            self.phero_screen.show_label_img(self.phero_image)
        
    def phero_start_render(self):
        self.phero_frame_rate = self.viewer.phero.spinBox_frame_rate.value()
        self.phero_model.dt = 1.0/self.phero_frame_rate
        self.phero_is_rendering = True
        self.phero_render_start_t = time.time()
        self.phero_render_thread = threading.Thread(target=self.phero_compute_pheromone)
        self.phero_render_thread.start()
        self.phero_timer.start(int(1000/self.phero_frame_rate))
        self.viewer.phero.pb_start.setDisabled(True)
        self.viewer.phero.pb_pause.setDisabled(False)
        self.viewer.phero.pb_stop.setDisabled(False)
    
    def phero_stop_render(self):
        self.phero_timer.stop()
        self.phero_frame_num = 0
        self.phero_is_rendering = False
        self.viewer.phero.pb_start.setDisabled(False)
        self.viewer.phero.pb_pause.setDisabled(True)
        self.viewer.phero.pb_stop.setDisabled(True)
    
    def phero_pause_render(self):
        self.phero_timer.stop()
        self.viewer.phero.pb_start.setDisabled(False)
        self.viewer.phero.pb_pause.setDisabled(True)
        self.viewer.phero.pb_stop.setDisabled(False)
    
    def _phero_save_config(self, config=None):
        if config is None:
            config = configparser.ConfigParser()
        config.add_section('Pheromone')
        # get config and write to the config file
        config_dict = {'mode':str(self.viewer.phero.comboBox_led_mode.currentIndex())}
        for name in ['sled_w','sled_h','sled_r','sled_c',
                        'sp_x','sp_y','arena_l','arena_w',
                        'frame_rate']:
            exec('config_dict[name] = str(self.viewer.phero.spinBox_{}.value())'.format(name))
        for p in ['d_kernel_s','diffusion','evaporation','injection','radius']:
            for c in ['r','g','b']:
                value = eval('self.viewer.phero.sp_' + p + '_' + c + '.value()')
                config_dict[p + '_' + c] = str(value)
        diversity = self.viewer.phero.te_diversity.toPlainText()
        config_dict['diversity_string'] = diversity
        for k,v in config_dict.items():
            config.set('Pheromone', k, v)
        return config
        
    def phero_save_config(self):
        filename, _ = QFileDialog.getSaveFileName(self.viewer.phero, 
                                                 'save pheromone config',
                                                 './', 'ini(*.ini)')
        if len(filename) != 0:
            cfgfile = open(filename, 'w')
            config = self._phero_save_config()
            config.write(cfgfile)
            cfgfile.close()
            self.viewer.system_logger('Save config {} successfully'.format(filename))
        else:
            QMessageBox.warning(self.viewer.phero, 'Error',
                                'Not a valid filename!')

    def _set_phero_config(self, config):
        if config.has_section('Pheromone'):
            options = config.options('Pheromone')
            if 'mode' in options:
                mode = config.getint('Pheromone', 'Mode')
                self.viewer.phero.comboBox_led_mode.setCurrentIndex(mode)
            for p in ['d_kernel_s','diffusion','evaporation','injection','radius']:
                for c in ['r','g','b']:
                    if p + '_' + c in options:
                        if (p == "radius") or (p == "d_kernel_s"):
                            value = config.getint('Pheromone', p + '_' + c)
                        else:
                            value = config.getfloat('Pheromone', p + '_' + c)
                        eval('self.viewer.phero.sp_'+ p + '_' + c +'.setValue(value)')
            for name in ['sled_w','sled_h','sled_r','sled_c',
                            'sp_x','sp_y','arena_l','arena_w',
                            'frame_rate']:
                if name in options:
                    if name == 'arena_l' or name == 'arena_w':
                        value = config.getfloat('Pheromone', name)
                    else:
                        value = config.getint('Pheromone', name)
                    eval('self.viewer.phero.spinBox_{}.setValue(value)'.format(name))
            if 'diversity_string' in options:
                self.viewer.phero.te_diversity.clear()
                self.viewer.phero.te_diversity.insertPlainText(config.get('Pheromone',
                                                                        'diversity_string'))
            self.phero_refresh_parameter()
        else:
            self.viewer.system_logger('No Pheromone Section Found', 'warning')

    def phero_load_config(self):
        filename, _ = QFileDialog.getOpenFileName(self.viewer.phero, 
                                                  'Load config File', './')
        if filename:
            self.viewer.system_logger('Successful loaded config file:{}'.format(filename))
            Config = configparser.ConfigParser()
            Config.read(filename)
            # set values
            self._set_phero_config(Config)
    
    def phero_screenshot(self):
        filename, _ = QFileDialog.getSaveFileName(self.viewer.phero, 
                                                 'save pheromone screenshot',
                                                 './', 'picture (*.png;*jpg;)')
        if len(filename) != 0:
            try:
                cv2.imwrite(filename, self.phero_image)
            except:
                QMessageBox.warning(self.viewer.phero, "Error", "Cannot save screenshot as file:" + filename)
                self.viewer.system_logger("Cannot save screenshot as file:" + filename)
            self.viewer.system_logger("Successfully save screenshot as file:" + filename)
    
    def loc_show_window(self):
        self.viewer.main_menu.pb_loc.setDisabled(True)
        self.viewer.loc_embedded.show()
    
    def loc_event_handle(self, signal):
        if signal == "close":
            self.viewer.main_menu.pb_loc.setDisabled(False)
            self.viewer.loc_embedded.hide()
            if self.loc_camera_opened:
                # closing camera
                self.loc_close_camera()
    
    def loc_check_camera(self):
        cameraCnt, cameraList = self.loc_camera.enumCameras()
        if cameraCnt is None:
            self.viewer.show_message_box('No camera founded')
        else:
            camera = cameraList[0]
            self.viewer.show_message_box('Camera {} founded'.format(camera.getKey(camera)))
    
    def loc_start_camera_capture(self):
        if self.loc_camera.start_grab_img() == 0:
            time.sleep(0.5)
            self.viewer.system_logger('Start grabing image success')
            return 0
        else:
            self.viewer.show_message_box("Error: cannot start capture image.")
            return -1
            
    def loc_open_camera(self):
        text = self.viewer.loc_embedded.sender().text()
        if text == "Start Capture":
            if not self.loc_camera_opened:
                if self.loc_start_camera_capture() == 0:
                    self.loc_camera_opened = True
                    self.viewer.loc_embedded.pb_start_capture.setText('Stop Capture')
                    return 0
                else:
                    return -1
            else:
                print('Already in capturing')
                return -1
        elif text == "Stop Capture":
            if self.loc_camera_opened:
                # closing camera
                self.loc_close_camera()
                self.viewer.loc_embedded.pb_start_capture.setText('Start Capture')
                return 0
    
    def loc_close_camera(self):
        if self.loc_camera.stop_grab_img() == 0:
            self.viewer.system_logger('camera stopped grabing images')
        # if self.loc_camera.closeCamera(self.loc_camera) == 0:
        #     self.viewer.system_logger('camera closed')
        self.loc_camera_opened = False
        
    def loc_start(self):
        text = self.viewer.loc_embedded.sender().text()
        if text == 'Start':
            if self.loc_camera_is_calibrating:
                self.loc_camera_is_calibrating = False
            if not self.loc_camera_opened:
                if not self.loc_open_camera() == 0:
                    return
            self.loc_is_running = True
            self.loc_thread_computing = threading.Thread(target=self.loc_computing)
            self.loc_thread_computing.start()
            # display
            self.loc_refresh_timer.start(100)
            self.viewer.loc_embedded.pb_start.setText("Pause")
        elif text == 'Pause':
            self.loc_is_running = False
            # if self.loc_camera.stop_grab_img() == 0:
            #     time.sleep(0.5)
            #     print('Grabbing image stopped')
            self.viewer.loc_embedded.pb_start.setText("Start")
            self.loc_refresh_timer.stop()
        else:
            pass

    def loc_computing(self):
        while self.loc_is_running:
            # grab image from camera
            self.loc_current_image = self.loc_camera.get_gray_image()
            # calculate locations
            self.loc_world_location, self.loc_image_location, self.loc_heading = self.loc_model.search_pattern(self.loc_current_image)
            for info, h in zip(self.loc_world_location, self.loc_heading):
                # world location
                if info[0] in self.loc_world_locations.keys():
                    self.loc_world_locations[int(info[0])].append([info[1], info[2], h])
                else:
                    self.loc_world_locations.update({int(info[0]):[info[1], info[2], h]})

    def loc_display(self):
        self.loc_img_display = self.loc_camera.get_BGR_image()
        if self.loc_is_running:
            # color image
            for info, h in zip(self.loc_image_location, self.loc_heading):
                if self.viewer.loc_embedded.cb_show_id.isChecked():
                    self.loc_img_display = cv2.putText(self.loc_img_display, str(info[0]), (info[1], info[2]),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0), 3)
                if self.viewer.loc_embedded.cb_show_marker.isChecked():
                    self.loc_img_display = cv2.circle(self.loc_img_display, (info[1], info[2]), 20, (255, 0, 0), 4)
                    self.loc_img_display = cv2.arrowedLine(self.loc_img_display, (info[1], info[2]),
                                                (int(info[1] + 26*np.cos(h)),
                                                int(info[2] + 26*np.sin(h))),
                                                (255, 255, 0), 4)
                if self.viewer.loc_embedded.cb_show_location.isChecked():
                    self.loc_img_display = cv2.putText(self.loc_img_display,
                                            '[{:3d}, {:3d}, {:.1f}]'.format(info[1], info[2], np.rad2deg(h)),
                                            (info[1], info[2]+40),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 255, 255), 3)
                # if self.viewer.loc_embedded.cb_show_trajectory.isChecked():
                #     end_ind = np.min([20, len(self.loc_image_location)])
                #     for t in range(end_ind):
                #         for i, info in enumerate(self.loc_image_location[-1:-end_ind+1]):
                #             pass 
        elif self.loc_camera_is_calibrating:
            if self.loc_model.calibrate_info:
                if self.viewer.loc_embedded.cb_cal_show_corner.isChecked():
                    # show chessboard corners
                    c_col = self.viewer.loc_embedded.sp_chessboard_c.value()
                    c_row = self.viewer.loc_embedded.sp_chessboard_r.value()
                    if self.loc_model.chessboard_corners[0]:
                        self.loc_img_display = cv2.drawChessboardCorners(self.loc_img_display,
                                                                (c_row, c_col),
                                                                self.loc_model.chessboard_corners[1],
                                                                self.loc_model.chessboard_corners[0])
                if self.viewer.loc_embedded.rb_cal_show_axes.isChecked():
                    # make sure that the screen still displays chessboard
                    self.phero_refresh_parameter()
                    # show world axes in the image
                    self.loc_img_display = self.loc_model.draw_world_axes_in_image_plane(self.loc_img_display,
                                                                                self.loc_model.calibrate_info[3][0],
                                                                                self.loc_model.calibrate_info[4][0],
                                                                                self.loc_model.calibrate_info[1],
                                                                                self.loc_model.calibrate_info[2],
                                                                                axis_len=(self.arena_length,
                                                                                          self.arena_width,
                                                                                          10))
                elif self.viewer.loc_embedded.rb_cal_show_points.isChecked():
                    c_col = self.viewer.loc_embedded.sp_chessboard_c.value()
                    c_row = self.viewer.loc_embedded.sp_chessboard_r.value()
                    s_ = min([self.phero_img_height//c_row, self.phero_img_width//c_col])
                    size = s_/self.phero_img_height*self.arena_width
                    # comparing the results with groud truth
                    # point = np.array([[3*self.arena_width/4,0,0],
                    #                   [self.arena_length/2,self.arena_width/2,0],
                    #                   [0,3*self.arena_width/4,0],
                    #                   [3*self.arena_length/4,self.arena_width/4,0]
                    #                   ], dtype=np.float32)
                    point = np.array([[size,size,0],
                                      [size,self.arena_width-size,0],
                                      [self.arena_length-size,size,0],
                                      [self.arena_length-size,self.arena_width-size,0]
                                      ], dtype=np.float32)
                    # re-calculate points in image plane
                    self.loc_img_display = self.loc_model.draw_world_points_in_image_plane(self.loc_img_display, point,
                                                                                  self.loc_model.calibrate_info[3][0],
                                                                                  self.loc_model.calibrate_info[4][0],
                                                                                  self.loc_model.calibrate_info[1],
                                                                                  self.loc_model.calibrate_info[2])
                    # showing points in the arena screen
                    img = np.zeros([self.phero_img_height, self.phero_img_width, 3], dtype=np.uint8)
                    for p in point:
                        img = cv2.circle(img,
                                         (int(p[0]/self.arena_width*self.phero_img_height),
                                          int(p[1]/self.arena_length*self.phero_img_width)),
                                         30, (0,255,0), 4)
                    self.phero_screen.show_label_img(img)
            else:
                c_col = self.viewer.loc_embedded.sp_chessboard_c.value()
                c_row = self.viewer.loc_embedded.sp_chessboard_r.value()
                s_ = np.array([self.phero_img_height//c_row, self.phero_img_width//c_col], dtype=np.uint8)
                if np.argmin(s_) == 0:
                    c_row -= 1
                else:
                    c_col -= 1
                self.loc_img_display = cv2.putText(self.loc_img_display, 'cannot found {}x{} corners'.format(c_row, c_col),
                                            (20, self.loc_img_display.shape[0]//10),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0), 2)

        self.viewer.loc_embedded.update_localization_display(self.loc_img_display)
    
    def loc_dispaly_calibration_data(self):
        if self.loc_camera_opened:
            self.loc_img_display = self.loc_camera.get_BGR_image()
        else:
            self.loc_img_display = np.zeros([1200, 1920, 3], dtype=np.uint8)
        c_col = self.viewer.loc_embedded.sp_chessboard_c.value()
        c_row = self.viewer.loc_embedded.sp_chessboard_r.value()
        s_ = min([self.phero_img_height//c_row, self.phero_img_width//c_col])
        size = s_/self.phero_img_height*self.arena_width
        point = np.array([[size,size,0],
                            [size,self.arena_width-size,0],
                            [self.arena_length-size,size,0],
                            [self.arena_length-size,self.arena_width-size,0]
                            ], dtype=np.float32)
        self.loc_img_display = self.loc_model.draw_world_points_in_image_plane(self.loc_img_display, point,
                                                                      self.loc_model.rvecs,
                                                                      self.loc_model.tvecs,
                                                                      self.loc_model.mtx,
                                                                      self.loc_model.dist)
        img = np.zeros([self.phero_img_height, self.phero_img_width, 3], dtype=np.uint8)
        for p in point:
            img = cv2.circle(img,
                                (int(p[0]/self.arena_width*self.phero_img_height),
                                int(p[1]/self.arena_length*self.phero_img_width)),
                                30, (0,255,0), 4)
        self.phero_screen.show_label_img(img)
        self.viewer.loc_embedded.update_localization_display(self.loc_img_display)
    
    def loc_camera_setting(self):
        sender = self.viewer.loc_embedded.sender()
        if sender.objectName() == 'hS_exposure':
            exp = self.viewer.loc_embedded.hS_exposure.value() * 100
            if self.loc_camera.setExposureTime(self.loc_camera.camera, exp) == 0:
                time.sleep(0.1)
    
    def loc_generate_pattern(self):
        image = np.ones((2100, 2970, 3), np.uint8) * 255
        ID_table = open("ID.txt",'w+')
        pattern_pos = [0,0]
        pattern_size = 440
        offset = 100
        v = self.viewer.loc_embedded.sp_pattern_num_r.value()
        h = self.viewer.loc_embedded.sp_pattern_num_c.value()
        vi=100/v
        hj=140/h
        id = 0
        bias = 10
        for j in range(v):
            for i in range(h):
                pattern_pos = [int(offset + pattern_size/2 + pattern_size*i), int(offset + pattern_size/2+ + pattern_size*j) ]
                cv2.circle(image, pattern_pos, int(pattern_size/2), (0,0,0), -1)
                cv2.circle(image, pattern_pos, int(pattern_size/2), (255,255,255), 1)
                # draw 4*4 ellipse with center at pattern_pos
                cv2.ellipse(image, pattern_pos, (140, 170), 0, 0, 360, (255,255,255), -1)
                r_v = int(30+vi*i)
                r_h = int(40+hj*j)
                cv2.ellipse(image, (pattern_pos[0],pattern_pos[1]+bias), (r_v, r_h), 0, 0, 360, (0,0,0), -1)
                # cv2.ellipse(image, pattern_pos, (r_v, r_h), 0, 0, 360, (0,0,0), -1)
                r0 = r_v/150
                r1 = r_h/180
                
                cv2.putText(image, str(id), pattern_pos, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255), 1)
                ID_table.write(str(id)+" "+str(round(r1,3))+' '+str(round(r0,3))+'\n')
                id += 1
        try:
            cv2.imwrite("pattern1.png", image)
            cv2.imshow('Pattern {:d} x {:d}'.format(v, h), cv2.resize(image, (2100/4, 2970/4)))
            ID_table.close()
        except:
            self.viewer.system_logger("Cannot Write File [ID.txt] or [Pattern.png]",
                                      log_type='err', out='sys')
            self.viewer.show_message_box('Cannot Write File [ID.txt] or [Pattern.png]', 'err')
            return
        self.viewer.system_logger("Write File [ID.txt] or [Pattern.png] successfully",
                                      log_type='err', out='sys')
        self.viewer.show_message_box('Success: ID info was write to [ID.txt], pattern saved as [Pattern.png]','err')
    
    def loc_start_camera_calibration(self):
        text = self.viewer.loc_embedded.sender().text()
        if text == 'Run':
            if self.loc_is_running:
                self.loc_is_running = False
            if not self.loc_camera_opened:
                if not self.loc_open_camera() == 0:
                    return
            self.loc_camera_is_calibrating = True
            self.loc_thread_calibration = threading.Thread(target=self.loc_calibrate_camera)
            self.loc_thread_calibration.start()
            # display
            self.loc_refresh_timer.start(100)
            self.viewer.loc_embedded.pb_start_calibration.setText("Stop")

        elif text == 'Stop':
            self.loc_camera_is_calibrating = False
            self.viewer.loc_embedded.pb_start_calibration.setText("Run")
            self.loc_refresh_timer.stop()
        else:
            pass
        
    def loc_calibrate_camera(self):
        while self.loc_camera_is_calibrating and self.loc_camera_opened:
            c_c = self.viewer.loc_embedded.sp_chessboard_c.value()
            c_r = self.viewer.loc_embedded.sp_chessboard_r.value()
            offsety = self.viewer.loc_embedded.sp_cal_offset_y.value()
            offsetx = self.viewer.loc_embedded.sp_cal_offset_x.value()
            border = self.viewer.loc_embedded.sp_cal_border_w.value()/self.phero_img_height*self.arena_width
            s_ = np.array([self.phero_img_height//c_r, self.phero_img_width//c_c], dtype=np.uint8)
            i_ = np.argmin(s_)
            c_size = s_[i_]/self.phero_img_height*self.arena_width
            if i_ == 0:
                c_r -= 1
            else:
                c_c -= 1
            # grab image from camera
            img = self.loc_camera.get_gray_image()
            if self.loc_model.run_calibration(img, c_c, c_r, c_size, offsetx, offsety, border):
                # has found enougn corner, calibration finished
                break

    def loc_load_calibration_data(self):
        filename, _ = QFileDialog.getOpenFileName(self.viewer.loc_embedded, 
                                                  'Load Calibration File', './')
        if filename:
            with np.load(filename) as X:
                for i in ('mtx','dist','rvecs','tvecs'):
                    self.loc_model.calibrate_info[i+1] = X[i]

    def loc_save_as_imgae(self):
        filename, _ = QFileDialog.getSaveFileName(self.viewer.phero, 
                                            'save localization image',
                                            './', 'image(*.png; *.jpg)')
        if len(filename) != 0:
            try:
                cv2.imwrite(filename, self.loc_img_display)
            except:
                self.viewer.system_logger('Cannot write image to: ' + filename)
        self.viewer.system_logger('Successfully write image to: ' + filename)
        
    def serial_ports_scan(self):
        # get the valid serial ports as a list
        port_list = serial.tools.list_ports.comports()
        self.viewer.com.comboBox_com_port.clear()
        for pl in port_list:
            self.viewer.com.comboBox_com_port.addItem(pl[0])
        if len(port_list) > 0:
            self.viewer.system_logger("Serial Port scanned, got %d ports"%(len(port_list)), out='com')
        else:
            self.viewer.system_logger("Serial Port scanned, no port is available",
                                      log_type='warning', out='com')
    
    def serial_port_open(self):
        if self.serial_port.isOpen():
            QMessageBox.warning(self.viewer.com, 'Error', 'Port is already opened')
        else:
            com_name = self.viewer.com.comboBox_com_port.currentText()
            if com_name != '':
                self.serial_port.port = com_name
                try:
                    self.serial_port.open()
                except:
                    QMessageBox.warning(self.viewer.com, 'Error', 'Cannot open port.')
                    self.viewer.system_logger('try to open ' + com_name + ', but failed', out='com')
                if self.serial_port.isOpen():
                    self.viewer.system_logger(com_name + ' is successfully opened', out='com')
            else:
                QMessageBox.warning(self.viewer.com, 'Error', 'No Port is available')
    
    def serial_port_close(self):
        if self.serial_port.isOpen():
            self.serial_port.close()
            com_name = self.viewer.com.comboBox_com_port.currentText()
            if not self.serial_port.isOpen():
                self.viewer.system_logger(com_name + ' is closed', out='com')
        else:
            QMessageBox.warning(self.viewer.com, 'Error', 'No port is open')
    
    def serial_run_capture(self):
        while self.serial_data_is_running:
            if self.serial_port.isOpen():
                d_display = b''
                d = self.serial_port.read(1)
                if d:
                    d_display += d
                    # check frame head and end
                    if d == b'\x01':
                        d = self.serial_port.read(1)
                        d_display += d
                        if d == b'\xfe':
                            # got frame head, then read {self.serial_package_len} bytes
                            # print('got frame head')
                            data = self.serial_port.read(int(self.serial_recv_package_len))
                            # print(len(data), data)
                            d_display += data
                            # check frame end
                            d = self.serial_port.read(2)
                            if d == b'\xfe\x01':
                                # print(data)
                                # got frame end
                                # print('got frame end')
                                # update robot data
                                d_display += d
                                self.serial_data_model.data_transfer(data)
                                # self.serial_port.reset_input_buffer()
                    # raw data display
                    if self.viewer.com.cb_show_raw.isChecked():
                        raw_data_str = ''
                        if self.viewer.com.cb_show_raw_hex.isChecked():
                            for c in d_display:
                                raw_data_str += ('0'*(c <= 15) + str(hex(c))[2:]) + ' '
                        else:
                            raw_data_str = d_display.decode('utf-8', errors='replace')
                        self.viewer.com.raw_data_insert_text(raw_data_str)
                    
                    # update robot ID combox
                    self.serial_update_robot_id()
    
    def serial_update_robot_id(self):
        if self.viewer.com.cbox_request_id.count() != len(self.serial_data_model.robot_data.keys()):
            self.viewer.com.cbox_request_id.clear()
            self.viewer.com.cbox_send_robot_id.clear()
        for k in self.serial_data_model.robot_data.keys():
            self.viewer.com.cbox_request_id.addItem(str(k))
            self.viewer.com.cbox_send_robot_id.addItem(str(k))

    def serial_port_send(self, data, r_id=None):
        header = b''
        # add robot ID
        # if not given, get from the GUI
        if r_id is None:
            if self.viewer.com.cb_send_to_all.isChecked():
                r_id = 200
            else:
                r_id = self.viewer.com.cbox_send_robot_id.currentText()
                if r_id == '':
                    QMessageBox.warning(self.viewer.com, 'Error', 'No valid Robot ID')
                    self.viewer.system_logger('Try to send data, but failed with no valid Robot ID', 'error',
                                              out='com')
                    return
                else:
                    r_id = int(r_id)
        # header : ID (1 bytes) + SYS_TIME(4 bytes)
        header = st.pack('B', r_id)
        # add time stamp - ms
        t = int((time.time() - self.sys_time) * 1000)
        header += st.pack('L', t)

        data = header + data
        # data package must be {self.serial_package_len} bytes
        if len(data) <= self.serial_send_package_len:
            data += b'\x00' * (self.serial_send_package_len-len(data))
        else:
            pass
        # add frame head and end and then send
        try:
            len_bytes = self.serial_port.write(b'\x01\xfe' + data + b'\xfe\x01')
        except:
            QMessageBox.warning(self.viewer.com, 'Error', 'Cannot send data')
            self.viewer.system_logger('Try to send data, but failed -_-', 'error', out='com')
            return
        self.viewer.system_logger('Send %d bytes data via serial port: \n %s' % (len_bytes, data),
                                  out='com')
    
    def serial_send_raw_data(self):
        send_str = self.viewer.com.text_edit_send_raw.toPlainText()
        if self.viewer.com.cb_send_hex.isChecked():
            # '12 34 ac ed' -> b'\x12\x34\xac\xed'
            send_data = bytes().fromhex(send_str.replace(" ", ""))
            if self.viewer.com.cb_send_newline.isChecked():
                send_data += b'\x0d\x0a'
        else:
            send_data = b''
            for s in send_str.split(" "):
                print(s)
                # num - int
                if s.isdigit():
                    send_data += st.pack('f', int(s))
                # num - float
                elif '.' in s:
                    s_s = s.split('.')
                    if s_s[0].isdigit and s_s[1].isdigit:
                        send_data += st.pack('f', (float(s)))
                # letters and markers
                else:
                    send_data += bytes(s, encoding='utf-8')
        self.serial_port_send(send_data)
    
    def serial_send_motion(self):
        m = self.viewer.com.sender().objectName().split("_")[-1] 
        speed = self.viewer.com.sp_motion_speed.value()
        data = self.robot_motion_ctl_table[m].encode('utf-8') + st.pack('f', speed)
        self.serial_port_send(data)
    
    def serial_port_send_parameter(self):
        data = b'DWP'
        for i in range(5):
            if eval('self.viewer.com.cb_send_p_' + str(i+1) + '.isChecked()'):
                f = eval('self.viewer.com.sp_send_p_' + str(i+1) + '.value()')
                data += st.pack('f', f)
        # print(data)
        self.serial_port_send(data)

    def serial_request_data(self):
        data_request_num = self.viewer.com.sp_num_packs.value()
        
        if self.viewer.com.cb_request_all.isChecked():
            r_id = 200
        else:
            r_id = self.viewer.com.cbox_request_id.currentText()
            if r_id == '':
                QMessageBox.warning(self.viewer.com, 'Error', 'No valid Robot ID')
                self.viewer.system_logger('Try to send data, but failed with no valid Robot ID', 'error',
                                          out='com')
                return
            else:
                r_id = int(r_id)

        data = 'DR'.encode('utf-8')
        data += st.pack('B', data_request_num)
        self.serial_port_send(data, r_id=r_id)

    def serial_start_capture(self):
        if self.viewer.com.pb_start_capture.text() == 'Start \n Capture':
            if self.serial_port.isOpen():
                print('start capture...')
                self.viewer.system_logger('start capturing robot data...', out='com')
                self.serial_data_is_running = True
                if not self.thread_robot_data_capture.is_alive():
                    self.thread_robot_data_capture = threading.Thread(target=self.serial_run_capture)
                    self.thread_robot_data_capture.start()
                self.viewer.com.pb_start_capture.setText('Stop \n Capture')
                self.viewer.com.pb_open_port.setDisabled(True)
                self.viewer.com.pb_close_port.setDisabled(True)
                self.viewer.com.pb_scan_port.setDisabled(True)
            else:
                QMessageBox.warning(self.viewer.com, 'Error',
                                    'The serial port is not open, please open the port and retry')

        elif self.viewer.com.pb_start_capture.text() == 'Stop \n Capture':
            self.viewer.system_logger('stop capturing robot data', out='com')
            self.serial_data_is_running = False
            self.viewer.com.pb_start_capture.setText('Start \n Capture')
            self.viewer.com.pb_open_port.setDisabled(False)
            self.viewer.com.pb_close_port.setDisabled(False)
            self.viewer.com.pb_scan_port.setDisabled(False)
    
    def serial_clear_data_cached(self):
        len_ = len(self.serial_data_model.robot_data)
        self.serial_data_model.clear_data()
        if len(self.serial_data_model.robot_data) == 0:
            self.viewer.system_logger('Cleared {} Robot Data.'.format(len))
            self.serial_update_robot_id()
    
    def exp_visualization_add_plot(self):
        name = self.viewer.main_menu.sender().objectName().split("_")[-1]
        self.viewer.add_visualization_figure(self.exp_available_data_key,
                                             [str(i) for i in self.exp_detected_robot_ids],
                                             name)
        self.viewer.plots[-1].signal.connect(self.exp_visualization_plot_callback)
    
    def exp_visualization_plot_callback(self, signal):
        if signal.startswith('close'):
            # close the plot window
            for p in self.viewer.plots:
                if p.plot_index == int(signal.split('_')[1]):
                    self.viewer.plots.remove(p)
                    break
        elif signal.startswith('add'):
            # add plotting item
            for p in self.viewer.plots:
                if p.plot_index == int(signal.split('_')[1]):
                    p.add_plots(p.cbox_data.currentText())
                    break
        elif signal.startswith('remove'):
            # remove plotting item
            for p in self.viewer.plots:
                if p.plot_index == int(signal.split('_')[1]):
                    p.remove_plots(p.cbox_data.currentText())
                    break
    
    def exp_save_data_setting(self):
        self.viewer.data_save_setting.update_data_checkbox(self.exp_available_data_key,
                                                           self.exp_detected_robot_ids)
        self.viewer.data_save_setting.show()
    
    def exp_save_data_setting_update(self, answer):
        if answer == 'ok':
            self.exp_save_data_selected_id = []
            self.exp_save_data_selected_data = []
            for k, v in self.viewer.data_save_setting.id_check_box_list.items():
                if v.isChecked():
                    self.exp_save_data_selected_id.append(k)
            for k, v in self.viewer.data_save_setting.data_check_box_list.items():
                if v.isChecked():
                    self.exp_save_data_selected_data.append(k)
            self.exp_save_data_file_type = self.viewer.data_save_setting.cbox_file_type.currentText()
            self.exp_save_data_interval = self.viewer.data_save_setting.spinBox_time_interval.value()
            self.exp_save_data_max_l = self.viewer.data_save_setting.spinBox_max_data_length.value()
            self.viewer.data_save_setting.hide()
            self.viewer.system_logger('Save data option Changed to: file type: %s, time interval: %d,'
                                          'max length %d robot: %s data: %s' %(self.exp_save_data_file_type,
                                                                               self.exp_save_data_interval,
                                                                               self.exp_save_data_max_l,
                                                                               str(self.exp_save_data_selected_id),
                                                                               str(self.exp_save_data_selected_data)))
            self.exp_data_to_save = []
        elif answer == 'cancel':
            self.viewer.data_save_setting.hide()
    
    def exp_save_data(self):
        exp_t = time.time() - self.exp_start_time
        current_data = []
        for d_s in self.exp_save_data_selected_data:
            robot_data = {}
            if d_s == 'POS_X':
                for k in self.loc_world_locations.keys():
                    robot_data.update({k:self.loc_world_locations[k][-1][0]})
                # for r_info in self.loc_world_locations[-1]:
                #     robot_data.update({r_info[0]:r_info[1]})
            elif d_s == 'POS_Y':
                for k in self.loc_world_locations.keys():
                    robot_data.update({k:self.loc_world_locations[k][-1][1]})
                # for r_info in self.loc_world_location[-1]:
                #     robot_data.update({r_info[0]:r_info[2]})
            else:
                robot_data = self.serial_data_model.get_robots_data(d_s, t=-1)
            current_data.append(robot_data)

        if self.exp_save_data_file_type == '.txt':
            # if over max length, then write to file
            if len(self.exp_data_to_save) >= self.exp_save_data_max_l:
                print(self.exp_data_to_save)
                # txt
                exp_name = self.viewer.main_menu.lineEdit_exp_name.text()
                filename = exp_name + '_' + str(int(self.exp_start_time)) + '_' + \
                           '_'.join(self.exp_save_data_selected_data) + '.txt'
                with open(filename, mode='a') as self.save_data_txt_f:
                    self.save_data_txt_f.write('\n'.join(self.exp_data_to_save) + '\n')
                    self.viewer.system_logger(
                        'Write %d lines of data to txt file successfully' % len(self.exp_data_to_save),
                        out='exp')
                    self.exp_data_to_save = []
            for r in self.exp_save_data_selected_id:
                data_by_id = [str(exp_t), str(r)]
                for d in current_data:
                    try:
                        data_by_id.append(str(d[r]))
                    except:
                        self.viewer.system_logger(
                            'lost data of robot(ID=%d)' % r, log_type='error', out='exp')
                data_by_id = ','.join(data_by_id)
                self.exp_data_to_save.append(data_by_id)

        # save mat file
        elif self.exp_save_data_file_type == '.mat':
            if len(self.exp_data_to_save) >= self.exp_save_data_max_l:
                # mat
                exp_name = self.viewer.main_menu.lineEdit_exp_name.text()
                filename = exp_name + '_' + str(int(self.exp_start_time)) + '_' + \
                           '_'.join(self.exp_save_data_selected_data) + '.mat'
                try:
                    saved_dic = sio.loadmat(filename)
                    save_dic = {'data': np.vstack([saved_dic['data'], self.exp_data_to_save])}
                except:
                    # create a new mat file
                    save_dic = {'data': self.exp_data_to_save}
                sio.savemat(filename, save_dic)
                self.viewer.system_logger('Write %d long list to mat file successfully' % len(self.exp_data_to_save),
                                              out='exp')
                self.exp_data_to_save = []
            for r in self.exp_save_data_selected_id:
                data_by_id = [exp_t, r]
                for d in current_data:
                    try:
                        data_by_id.append(d[r])
                    except:
                        self.viewer.system_logger(
                            'lost data of robot(ID=%d)' % r, log_type='error', out='exp')
                        break
                self.exp_data_to_save.append(data_by_id)

            print(len(self.exp_data_to_save))
        # save csv file
        elif self.exp_save_data_file_type == '.csv':
            pass
        else:
            print('error')

    def exp_save_config(self):
        filename, _ = QFileDialog.getSaveFileName(self.viewer.main_menu, 
                                                 'save pheromone config',
                                                 './', 'ini(*.ini)')
        if len(filename) != 0:
            cfgfile = open(filename, 'w')
            config = self._phero_save_config()
            config = self._vscene_save_config(config)
            config.write(cfgfile)
            cfgfile.close()
            self.viewer.system_logger('Save config {} successfully'.format(filename))
        else:
            QMessageBox.warning(self.viewer.vscene, 'Error',
                                'Not a valid filename!')
    
    def exp_load_config(self):
        filename, _ = QFileDialog.getOpenFileName(self.viewer.main_menu, 
                                                  'Load config File', './')
        if filename:
            self.viewer.system_logger('Successful loaded config file:{}'.format(filename))
            config = configparser.ConfigParser()
            config.read(filename)
            # set values
            self._set_phero_config(config)
            self._set_vscene_config(config)

    def exp_start(self):
        print(self.viewer.main_menu.pb_start_exp.text())
        if self.viewer.main_menu.pb_start_exp.text() == "Start \n Experiment":
            name = self.viewer.main_menu.lineEdit_exp_name.text()
            if name == "":
                self.viewer.show_message_box("Please provide experiment name", 'warning')
                return
            self.exp_start_time = time.time()
            self.exp_is_running = True
            if not self.exp_thread.is_alive():
                self.exp_thread = threading.Thread(target=self.exp_task)
                self.exp_thread.start()
            self.viewer.main_menu.pb_start_exp.setText("Stop \n Experiment")
            self.viewer.main_menu.lineEdit_exp_name.setDisabled(True)
            self.viewer.system_logger('Experiment ' + name + ' started', out='exp')
        elif self.viewer.main_menu.pb_start_exp.text() == "Stop \n Experiment":
            self.viewer.main_menu.lineEdit_exp_name.setDisabled(False)
            self.exp_is_running = False
            name = self.viewer.main_menu.lineEdit_exp_name.text()
            self.viewer.system_logger('Experiment ' + name + ' stopped', out='exp')
            self.viewer.main_menu.pb_start_exp.setText("Start \n Experiment")

    def exp_task(self):
        self.exp_predator_id = 0
        self.exp_prey_ids = range(10)
        self.exp_prey_cluster = {}
        self.exp_prey_cluster_id = dict.fromkeys(self.exp_prey_ids, 1)
        self.exp_prey_robot_size = dict.fromkeys(self.exp_prey_ids, 0.04)

        while self.exp_is_running:
            # timer
            t = time.time() - self.exp_start_time
            h = int(t / 3600)
            m = int((t - h * 3600) / 60)
            s = t - h * 3600 - m * 60
            self.viewer.main_menu.et_exp_timer.setText('ExperimentTime: %2d H %2d M% 2.2f S' % (h, m, s))
            # task
            print('i am running')
            self.exp_prey_cluster = {}
            # robot if defined: predator: 0, prey: 1,2...10
            # 1. receive data from prey and predator (energy, f_avoid, f_gather, px, py)
            pd_energy = self.serial_data_model.get_robot_data(self.exp_predator_id, 'Energy')
            pd_position = self.loc_world_locations[self.exp_predator_id][:2]
            pd_heading = self.loc_world_locations[self.exp_predator_id][2]
            pe_energys = {}
            pe_positions = {}
            pe_energy_sum = 0
            for id_ in self.exp_prey_ids:
                pe_energys.update({id_:self.serial_data_model.get_robot_data(id_, 'Energy')})
                pe_positions.update({id_:self.loc_world_locations[id_]})
                # prey in escaping state
                if self.serial_data_model.get_robot_data(id_, 'state') == 2:
                    pe_energy_sum += pe_energys[id_]
            
            # 2. justify if each prey's energy is less than zero (death):
            # if so, send the value of f_avoid, f_gather
            for d_id in np.argwhere(np.array(pe_energys)<0):
                send_data = b'DWD'
                # generate f_avoid and f_gather value
                if np.random.uniform(0, 1) > 0.1:
                    temp_a_e = [(a, a.energy) for a in self.alive_preys]
                    temp_a_e = sorted(temp_a_e, key=lambda x:x[1])
                    sum_ = sum([a_[1] for a_ in temp_a_e])
                    partial_p = [a_[1]/sum_ for a_ in temp_a_e]
                    p_ = np.random.rand(1)[0]
                    ind = np.where(partial_p < p_)[0][-1] if len(np.where(partial_p < p_)[0]) > 0 else -1
                    f_gather = temp_a_e[min(ind + 1, len(temp_a_e)-1)][0].f_gather
                    f_avoid = temp_a_e[min(ind + 1, len(temp_a_e)-1)][0].f_avoid
                else:
                    f_gather = np.random.uniform(0.02, 0.32)
                    f_avoid = np.random.uniform(0.01, 1.01)
                send_data += st.pack('2f', f_avoid, f_gather)
                print('send to prey {}:'.format(d_id), send_data)
                self.serial_port_send(send_data, r_id=d_id)
            # 3. clusterring the preys
            self.exp_prey_cluster = {}
            ## 3.1 roughly arange by distance
            for id_ in self.exp_prey_ids:
                if self.exp_prey_cluster_id[id_] is None:
                    self.exp_prey_cluster_id[id_] = len(self.exp_prey_cluster) + 1
                    self.exp_prey_cluster.update()
                    a_k = []
                    for k, v in self.exp_prey_cluster_id.items():
                        if v is None:
                            a_k.append(k)
                    cl_k = []
                    for k in a_k:
                        if k != id_:
                            if distance(pe_positions[k], pe_positions[id_]) <= \
                                (self.exp_prey_robot_size[k] + self.exp_prey_robot_size[id_])*2:
                                    cl_k.append(k)
                    self.exp_prey_cluster.update({self.exp_prey_cluster_id[id_]:a_k + [id_]})
                    for k in a_k:
                        self.exp_prey_cluster_id[k] = self.exp_prey_cluster_id[id_]
            # 3.2 re-arange cluster using rectangle box
            rect = {}
            margin = 10
            for ind, k in self.exp_prey_cluster.items():
                if len(k) >=2:
                    x = np.array([pe_positions[k_][0] for k_ in k])
                    y = np.array([pe_positions[k_][1] for k_ in k])
                    rect.update({ind:[x.min()-margin, x.max()+margin,
                                        y.min()-margin, y.max()+margin]})
            for c_c in combinations(rect.keys(), 2):
                r1l = rect[c_c[1]][0]
                r1r = rect[c_c[1]][1]
                r2l = rect[c_c[0]][0]
                r2r = rect[c_c[0]][1]
                r1b = rect[c_c[1]][2]
                r1t = rect[c_c[1]][3]
                r2b = rect[c_c[0]][2]
                r2t = rect[c_c[0]][3]
                if not ((r1l > r2r) or (r1t < r2b) or (r2l > r1r) or (r2t < r1b)):
                    for k in self.exp_prey_cluster[c_c[1]]:
                        self.exp_prey_cluster_id[k] = c_c[0]
            self.exp_prey_cluster = {}
            for id_ in self.exp_prey_ids:
                if self.exp_prey_cluster_id[id_] in self.exp_prey_cluster.keys():
                    self.exp_prey_cluster[self.exp_prey_cluster_id[id_]].append(id_)
                elif self.exp_prey_cluster_id[id_] is not None:
                    self.exp_prey_cluster.update({self.exp_prey_cluster_id[id_]:[id_]})
                    
            # 4. send predator its position, angle and the goal position
            send_data = b'DWD'
            # calculate goal position based on the prey's cluster info
            sorted_c = sorted(self.exp_prey_cluster.items(), key=lambda v:len(v[1]))
            _, v1 = sorted_c[-1]
            if len(v1) >= 5:
                if len(v1) >= len(self.exp_prey_ids) - 2:
                    i = np.random.randint(0, len(v1))
                    g_px = pe_positions[v1[i]][0]
                    g_py = pe_positions[v1[i]][1]
                else:
                    if len(v1) >= self.last_cluster_prey_num+2:
                        g_px = np.array([pe_positions[a][0] for a in v1]).mean()
                        g_py = np.array([pe_positions[a][1] for a in v1]).mean()
                    else:
                        # ?: how to send gpx gpy data when there is no change to the previous one? 
                        g_px = pd_position[0]
                        g_py = pd_position[1]
            send_data += st.pack('6f', pd_position[0],  pd_position[1], pd_heading, g_px, g_py, pe_energy_sum)
            print('send to predator:', send_data)
            self.serial_port_send(send_data, r_id=d_id)
            
            # data send format
            # to prey: DWD + 2f:'f_avoid, f_gather'
            # to predator: DWD + 6f:'p_x, p_y, h, g_px, g_py, prey_energy'
            
            # save data
            if self.viewer.main_menu.cb_auto_save.isChecked():
                self.exp_save_data()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    App = QApplication(sys.argv)
    controller = Controller()
    controller.viewer.login.show()
    sys.exit(App.exec_())
