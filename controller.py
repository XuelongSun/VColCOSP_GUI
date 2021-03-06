import sys
import os
import socket
import configparser
from multiprocessing import Process
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QColorDialog
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from model import LedImageProcess, PheromoneModel, LocDataModel
from viewer import LEDScreen, Viewer

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
        self.phero_screen = LEDScreen()
        # pheromone background settings
        self.phero_background_image = None
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
                                    'pos_marker_radius':20,
                                    'arena_border_color':(255,255,255),
                                    'arena_border_width':4,
                                    'arena_border_margin':2}
        self.viewer.phero.pb_bg_setting.clicked.connect(lambda:self.viewer.phero_bg_setting.show())
        self.viewer.phero_bg_setting.signal.connect(self.phero_bg_setting_message_handle)
        self.phero_image = None
        self.phero_timer = QTimer()
        self.phero_timer.timeout.connect(self.phero_render)
        self.phero_loaded_image_path = None
        self.phero_bg_loaded_image_path = None
        self.phero_loaded_video_path = None
        self.phero_video_capture = None
        self.phero_frame_num = 0
        self.arena_length = 0.8
        self.arena_width = 0.6
        
        #* localization
        self.viewer.main_menu.pb_loc.clicked.connect(self.loc_show_window)
        self.viewer.loc.signal.connect(self.loc_event_handle)
        self.loc_tcp_is_connected = False
        self.viewer.loc.pb_connect_to_loc.clicked.connect(self.loc_tcp_connect)
        self.loc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.loc_data_thread = SocketDataReceiver(self.loc_socket)
        self.viewer.loc.pb_read_data.clicked.connect(self.loc_toggle_read_data)
        self.viewer.loc.pb_read_data.setDisabled(True)
        self.viewer.loc.pb_disconnect_to_loc.setDisabled(True)
        self.viewer.loc.pb_disconnect_to_loc.clicked.connect(self.loc_socket_disconnect)
        
        self.loc_data_display_timer = QTimer()
        self.loc_data_display_timer.timeout.connect(self.loc_data_display)
        
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
            self.viewer.vscene.pb_show.setText("Hide")
            self.vscene_screen.show_label_img(self.vscene_screen_start_pos[0],
                                           self.vscene_screen_start_pos[1],
                                           self.vscene_img_width,
                                           self.vscene_img_height,
                                           self.vscene_image)
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
    
    def vscene_save_config(self):
        filename, _ = QFileDialog.getSaveFileName(self.viewer.vscene, 
                                                 'save pheromone config',
                                                 './', 'ini(*.ini)')
        if len(filename) != 0:
            cfgfile = open(filename, 'w')
            Config = configparser.ConfigParser()
            Config.add_section('VScene')
            # get config and write to the config file
            config_dict = {'mode':str(self.viewer.vscene.comboBox_led_mode.currentIndex())}
            for name in ['sled_w','sled_h','sled_r','sled_c',
                         'sp_x','sp_y','sled_fold_c','sled_fold_r',
                         'frame_rate','pc_display_rate']:
                exec('config_dict[name] = str(self.viewer.vscene.spinBox_{}.value())'.format(name))
            for k,v in config_dict.items():
                Config.set('VScene', k, v)
            Config.write(cfgfile)
            cfgfile.close()
            self.viewer.system_logger('Save config {} successfully'.format(filename))
        else:
            QMessageBox.warning(self.viewer.vscene, 'Error',
                                'Not a valid filename!')
        
    def vscene_load_config(self):
        filename, _ = QFileDialog.getOpenFileName(self.viewer.vscene, 
                                                  'Load config File', './')
        if filename:
            Config = configparser.ConfigParser()
            Config.read(filename)
            # set values
            options = Config.options('VScene')
            if 'mode' in options:
                mode = Config.getint('VScene', 'Mode')
                self.viewer.vscene.comboBox_led_mode.setCurrentIndex(mode)
            
            for name in ['sled_w','sled_h','sled_r','sled_c',
                         'sp_x','sp_y','sled_fold_c','sled_fold_r',
                         'frame_rate','pc_display_rate']:
                if name in options:
                    value = Config.getint('VScene', name)
                    eval('self.viewer.vscene.spinBox_{}.setValue(value)'.format(name))
            self.viewer.system_logger('Successful loaded config file:{}'.format(filename))
    
    def phero_show_window(self):
        self.viewer.main_menu.pb_phero.setDisabled(True)
        self.viewer.phero.show()
    
    def phero_event_handle(self,signal):
        if signal == "close":
            self.viewer.main_menu.pb_phero.setDisabled(False)
            self.viewer.phero.hide()
            self.phero_screen.hide()
    
    def phero_bg_setting_message_handle(self,message):
        if message == "OK":
            # update phero background param
            self.phero_bg_info_paras['pos_text_color'] = self.viewer.phero_bg_setting.pos_text_color
            self.phero_bg_info_paras['pos_line_color'] = self.viewer.phero_bg_setting.pos_line_color
            self.phero_bg_info_paras['pos_marker_color'] = self.viewer.phero_bg_setting.pos_marker_color
            self.phero_bg_info_paras['arena_border_color'] = self.viewer.phero_bg_setting.arena_border_color
            self.phero_bg_info_paras['pos_text_width'] = self.viewer.phero_bg_setting.sb_pos_text_width.value()
            self.phero_bg_info_paras['pos_line_width'] = self.viewer.phero_bg_setting.sb_pos_line_width.value()
            self.phero_bg_info_paras['pos_marker_width'] = self.viewer.phero_bg_setting.sb_pos_marker_width.value()
            self.phero_bg_info_paras['arena_border_width'] = self.viewer.phero_bg_setting.sb_arena_border_width.value()
            self.phero_bg_info_paras['arena_border_margin'] = self.viewer.phero_bg_setting.sb_arena_border_margin.value()
            
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
        
        self.phero_mode = self.viewer.phero.comboBox_led_mode.currentText()
        
        self.phero_model.dt = 1.0/self.viewer.phero.spinBox_frame_rate.value()
        
        self.arena_length = self.viewer.phero.spinBox_arena_l.value()
        self.arena_width = self.viewer.phero.spinBox_arena_w.value()
        
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
                                                      dtype=np.unit8)*self.phero_bg_sc
            elif self.viewer.phero.radioButton_image.isChecked():
                self.phero_background_image = self.phero_bg_loaded_image.copy()
            elif self.viewer.phero.radioButton_draw.isChecked():
                self.phero_background_image = self.phero_bg_drawn_image.copy()
            elif self.viewer.phero.radioButton_info.isChecked():
                # dynamically updated in phero_render()
                pass
        
        elif self.phero_mode == 'Loc-Calibration':
            self.phero_image = self.phero_model.generate_calibration_pattern(self.arena_length)
            self.viewer.phero.show_label_image(self.viewer.phero.label_image, 
                                                           self.phero_image)
        if (self.viewer.phero.pb_show.text() == "Hide") and (self.phero_image is not None):
            self.phero_screen.show_label_img(self.phero_screen_start_pos[0],
                                            self.phero_screen_start_pos[1],
                                            self.phero_img_width,
                                            self.phero_img_height,
                                            self.phero_image)
        
    def phero_load_picture(self):
        self.phero_loaded_image_path = self.load_picture(self.viewer.phero)
    
    def phero_load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self.viewer.phero, 
                                                  'Load Video', 
                                                  './', 'Picture File(*.mp4;)')
        if filename:
            print(filename)
            self.phero_loaded_video_path = filename
            self.phero_video_capture = cv2.VideoCapture(filename)
            self.phero_video_t_frame = int(self.phero_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def phero_load_bg_img(self):
        self.phero_bg_loaded_image_path = self.load_picture(self.viewer.phero)

    def phero_show_pheromone(self):
        # self.phero_refresh_parameter()
        if (self.viewer.phero.pb_show.text() == "Show") and (self.phero_image is not None):
            self.phero_screen.show_label_img(self.phero_screen_start_pos[0],
                                             self.phero_screen_start_pos[1], 
                                             self.phero_img_width,
                                             self.phero_img_height,
                                             self.phero_image)
            self.phero_screen.show()
            self.viewer.phero.pb_show.setText("Hide")
        else:
            self.phero_screen.hide()
            self.viewer.phero.pb_show.setText("Show")
    
    def phero_render(self):
        self.phero_mode = self.viewer.phero.comboBox_led_mode.currentText()
        if self.phero_mode == "Static":
            #* static: just show the loaded picture
            if self.phero_image is None:
                if self.phero_loaded_image_path is None:
                    QMessageBox.warning(self.viewer.phero,
                                        'Error',
                                        'Please load image first.')
                    self.phero_timer.stop()
                else:
                    image = cv2.imread(self.vscene_loaded_image_path)
                    if image is None:
                        QMessageBox.warning(self.viewer.vscene, 
                                            'Error', 
                                            'Cannot load image:{}'.format(self.vscene_loaded_image_path))
                        self.phero_timer.stop()
                    else:
                        gray = self.viewer.phero.cb_img_gray.isChecked()
                        self.phero_image = self.phero_model.transform_loaded_image(image,
                                                                                   self.phero_img_width,
                                                                                   self.phero_img_height,
                                                                                   gray=gray)
        elif self.phero_mode == "Video":
            #* video: play the video
            if self.phero_video_capture is None:
                QMessageBox.warning(self.viewer.phero,
                                    'Error',
                                    'Please load video first.')
                self.phero_timer.stop()
            else:
                self.phero_video_capture.set(1, self.phero_frame_num)
                ret, frame = self.phero_video_capture.read()
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
                    self.viewer.phero.label_video_filename.setText("{}: {}/{}".format(f, 
                                                                                      self.phero_frame_num, 
                                                                                      self.phero_video_t_frame))
        
        elif self.phero_mode == "Dy-Localization":
            # * dynamically interact with the localization system
            if not self.loc_data_thread.stop:
                # the latest frame data
                pos = self.loc_data_thread.loc_data_model.get_last_pos()
                # if got the positions of the robots
                if pos:
                    if self.viewer.phero.radioButton_info.isChecked():
                        # use information
                        image = np.zeros([self.phero_model.pixel_height,
                                          self.phero_model.pixel_width, 3],
                                         dtype=np.uint8)
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
                        for k,v in pos.items():
                            if (v[0] >= 0) and (v[0] <= self.arena_length) and \
                                (v[1]>=0) and (v[1] <= self.arena_width):
                                y = int(v[0]/self.arena_length*self.phero_model.pixel_width)
                                x = int(v[1]/self.arena_width*self.phero_model.pixel_height)
                                # pos-text
                                if self.viewer.phero_bg_setting.groupBox_pos_text.isChecked():
                                    image = cv2.putText(image, 
                                                        "{}:({},{})".format(k,v[0],v[1]),
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
                                                        self.phero_bg_info_paras['pos_marker_radius'],
                                                        self.phero_bg_info_paras['pos_marker_color'],
                                                        self.phero_bg_info_paras['pos_marker_width'])
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
                        try:
                            phero_image = self.phero_model.render_pheromone(pos, self.phero_channel,
                                                                        self.arena_length, self.arena_width)
                        except:
                            QMessageBox.warning(self.viewer.phero, 'error', 'Please try to refresh parameters.')
                    if self.phero_background_image is None:
                        QMessageBox.warning(self.viewer.phero,'Error','No valide background image!')
                        self.phero_timer.stop()
                    else:
                        # merge the background and pheromone, pheromone z-index is lower
                        img_temp = (phero_image/np.max(phero_image)*255).astype(np.uint8)
                        mask = cv2.bitwise_and(cv2.cvtColor(img_temp,cv2.COLOR_RGB2GRAY),
                                               cv2.cvtColor(self.phero_background_image,cv2.COLOR_RGB2GRAY))
                        if np.sum(mask) > 0:
                           phero_image[np.where(mask>0)] = self.phero_background_image[np.where(mask>0)]
                        phero_image[np.where(mask<=0)] = self.phero_background_image[np.where(mask<=0)] + phero_image[np.where(mask<=0)]
                        self.phero_image = phero_image.copy()
            else:
                QMessageBox.warning(self.viewer.phero,'Error','Please read the localization data first!')
                self.phero_timer.stop()
                    
        elif self.phero_mode == "Dy-Customized":
            #* user defined
            pass
        
        self.phero_frame_num += 1

        if (self.viewer.phero.pb_show.text() == "Hide") and (self.phero_image is not None):
            self.phero_screen.show_label_img(self.phero_screen_start_pos[0],
                                             self.phero_screen_start_pos[1],
                                             self.phero_img_width,
                                             self.phero_img_height,
                                             self.phero_image)
        
    def phero_start_render(self):
        self.phero_frame_rate = self.viewer.phero.spinBox_frame_rate.value()
        self.phero_model.dt = 1.0/self.phero_frame_rate
        self.phero_timer.start(int(1000/self.phero_frame_rate))
    
    def phero_stop_render(self):
        self.phero_frame_num = 0
        self.phero_timer.stop()
    
    def phero_pause_render(self):
        self.phero_timer.stop()
    
    def phero_save_config(self):
        filename, _ = QFileDialog.getSaveFileName(self.viewer.phero, 
                                                 'save pheromone config',
                                                 './', 'ini(*.ini)')
        if len(filename) != 0:
            cfgfile = open(filename, 'w')
            Config = configparser.ConfigParser()
            Config.add_section('Pheromone')
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
                Config.set('Pheromone', k, v)
                
            Config.write(cfgfile)
            cfgfile.close()
            self.viewer.system_logger('Save config {} successfully'.format(filename))
        else:
            QMessageBox.warning(self.viewer.phero, 'Error',
                                'Not a valid filename!')
        
    def phero_load_config(self):
        filename, _ = QFileDialog.getOpenFileName(self.viewer.phero, 
                                                  'Load config File', './')
        if filename:
            Config = configparser.ConfigParser()
            Config.read(filename)
            # set values
            options = Config.options('Pheromone')
            if 'mode' in options:
                mode = Config.getint('Pheromone', 'Mode')
                self.viewer.phero.comboBox_led_mode.setCurrentIndex(mode)
            for p in ['d_kernel_s','diffusion','evaporation','injection','radius']:
                for c in ['r','g','b']:
                    if p + '_' + c in options:
                        if (p == "radius") or (p == "d_kernel_s"):
                            value = Config.getint('Pheromone', p + '_' + c)
                        else:
                            value = Config.getfloat('Pheromone', p + '_' + c)
                        eval('self.viewer.phero.sp_'+ p + '_' + c +'.setValue(value)')
            for name in ['sled_w','sled_h','sled_r','sled_c',
                         'sp_x','sp_y','arena_l','arena_w',
                         'frame_rate']:
                if name in options:
                    if name == 'arena_l' or name == 'arena_w':
                        value = Config.getfloat('Pheromone', name)
                    else:
                        value = Config.getint('Pheromone', name)
                    eval('self.viewer.phero.spinBox_{}.setValue(value)'.format(name))
            if 'diversity_string' in options:
                self.viewer.phero.te_diversity.clear()
                self.viewer.phero.te_diversity.insertPlainText(Config.get('Pheromone',
                                                                          'diversity_string'))
            self.viewer.system_logger('Successful loaded config file:{}'.format(filename))
            
    def loc_show_window(self):
        self.viewer.main_menu.pb_loc.setDisabled(True)
        self.viewer.loc.show()
    
    def loc_event_handle(self, signal):
        if signal == "close":
            self.viewer.main_menu.pb_loc.setDisabled(False)
            self.viewer.loc.hide()
    
    def loc_tcp_connect(self):
        if not self.loc_tcp_is_connected:
            ip = self.viewer.loc.te_socket_ip.toPlainText()
            port = int(self.viewer.loc.te_socket_port.toPlainText())
            # connect to socket
            try:
                self.loc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.loc_data_thread = SocketDataReceiver(self.loc_socket)
                self.loc_socket.connect((ip, port))
            except:
                self.viewer.system_logger('Cannot connect to {} at {}'.format(ip, port),
                                          log_type="error")
                print('Cannot connect')
                return 0
            # successfully connect
            self.loc_tcp_is_connected = True
            self.loc_data_thread.socket_is_connect = True
            self.viewer.loc.pb_connect_to_loc.setText('^.^')
            self.viewer.loc.pb_connect_to_loc.setDisabled(True)
            self.viewer.loc.pb_disconnect_to_loc.setDisabled(False)
            self.viewer.loc.pb_read_data.setDisabled(False)
    
    def loc_socket_disconnect(self):
        if self.loc_tcp_is_connected:
            self.loc_tcp_is_connected = False
            self.loc_data_thread.socket_is_connect = False
            self.viewer.loc.pb_connect_to_loc.setText('Connect')
            self.viewer.loc.pb_connect_to_loc.setDisabled(False)
            self.viewer.loc.pb_read_data.setDisabled(True)
            self.loc_socket.shutdown(2)
            self.loc_socket.close()
        else:
            print('no connection')
    
    def loc_toggle_read_data(self):
        if self.viewer.loc.sender().text() == "ReadData":
            if self.loc_tcp_is_connected:
                if self.loc_data_thread.stop:
                    self.loc_data_thread.stop = False
                    self.loc_data_thread.start()
                    self.viewer.loc.pb_disconnect_to_loc.setDisabled(True)
                    self.viewer.loc.pb_read_data.setText('Stop')
                    self.loc_data_display_timer.start(100)
            else:
                print('no valid TCP connection')
        elif self.viewer.loc.sender().text() == "Stop":
            self.loc_data_thread.stop = True
            self.viewer.loc.pb_disconnect_to_loc.setDisabled(False)
            self.viewer.loc.pb_read_data.setText('ReadData')
        else:
            print(self.viewer.loc.sender().text())
    
    def loc_data_display(self):
        if self.loc_data_thread.loc_data_now:
            # print(self.loc_data_now)
            self.loc_display_img = np.zeros((540, 960, 3), np.uint8)
            # the latest frame data
            pos = self.loc_data_thread.loc_data_model.get_last_pos()
            if pos:
                for k, v in pos.items():
                    self.loc_display_img = cv2.circle(self.loc_display_img,
                                                      (int(v[0]/self.arena_length*960),
                                                       int(v[1]/self.arena_width*540)),
                                                      5, (255, 0, 0))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    dir = ' L' if v[0] < self.arena_length/2 else ' R'
                    self.loc_display_img = cv2.putText(self.loc_display_img, str(k) + dir,
                                                       (int(v[0] / self.arena_length * 960),
                                                        int(v[1] / self.arena_width * 540)),
                                                       font, 1, (0, 0, 255), 2)

            # for i, id in enumerate(self.loc_robot_ids[-1]):
            #     self.loc_display_img = cv2.circle(self.loc_display_img,
            #                                       (int(self.loc_robot_pos_xs[-1][i]/1.5*960),
            #                                        int(self.loc_robot_pos_ys[-1][i]/0.8*540)),
            #                                       5, (255, 0, 0))
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     self.loc_display_img = cv2.putText(self.loc_display_img, str(id),
            #                                        (int(self.loc_robot_pos_xs[-1][i]/1.5*960),
            #                                         int(self.loc_robot_pos_ys[-1][i]/0.8*540)),
            #                                        font, 1, (0, 0, 255), 2)
            # add to the view
            img = cv2.cvtColor(self.loc_display_img, cv2.COLOR_BGR2RGB)
            self.viewer.loc.update_localization_dislay(img)
        
if __name__ == "__main__":

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    App = QApplication(sys.argv)
    controller = Controller()
    controller.viewer.login.show()
    sys.exit(App.exec_())
