import sys
import os
import socket
from multiprocessing import Process
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
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
        self.phero_screen = LEDScreen()
        self.phero_image = None
        self.phero_timer = QTimer()
        self.phero_timer.timeout.connect(self.phero_render)
        self.phero_loaded_image_path = None
        self.phero_loaded_video_path = None
        self.phero_video_capture = None
        self.phero_frame_num = 0
        self.arena_length = 1.41
        self.arena_width = 0.8
        
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
        self.vscene_frame_rate = self.viewer.vscene.spinBox_led_v_frame_rate.value()
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
                roll = (self.viewer.vscene.spinBox_led_v_r_speed.value()/self.vscene_frame_rate) * self.vscene_frame_num
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
                    self.vscene_image = self.vscene_model.generate_video(self.vscene_video_image,
                                                                      self.vscene_img_width,
                                                                      self.vscene_img_height,
                                                                      self.vscene_fold_row,
                                                                      self.vscene_fold_column,
                                                                      animation={'roll':int(roll)})
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
        self.vscene_frame_rate = self.viewer.vscene.spinBox_led_v_frame_rate.value()
        self.vscene_timer.start(int(1000/self.vscene_frame_rate))
    
    def vscene_stop_video(self):
        self.vscene_frame_num = 0
        self.vscene_timer.stop()
    
    def vscene_pause_video(self):
        self.vscene_timer.stop()
        
    def phero_show_window(self):
        self.viewer.main_menu.pb_phero.setDisabled(True)
        self.viewer.phero.show()
    
    def phero_event_handle(self,signal):
        if signal == "close":
            self.viewer.main_menu.pb_phero.setDisabled(False)
            self.viewer.phero.hide()
            self.phero_screen.hide()
    
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
        
        self.phero_model.dt = 1.0/self.viewer.phero.spinBox_led_v_frame_rate.value()
        
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
            self.arena_length = self.viewer.phero.sp_arena_l.value()
            self.arena_width = self.viewer.phero.sp_arena_w.value()
            self.phero_model.pixel_width = self.phero_img_width
            self.phero_model.pixel_height = self.phero_img_height
            self.phero_frame_rate = self.viewer.phero.spinBox_led_v_frame_rate.value()
            # self.phero_model.dt =  1000/self.phero_frame_rate
            print(self.phero_frame_rate)
            # pheromone parameters
            for p in ['diffusion','evaporation','injection','radius']:
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
                # print(self.loc_data_now)
                self.loc_display_img = np.zeros((540, 960, 3), np.uint8)
                # the latest frame data
                pos = self.loc_data_thread.loc_data_model.get_last_pos()
                if pos:                    
                    # if got the positions of the robots
                    self.phero_image = self.phero_model.render_pheromone(pos, self.phero_channel,
                                                                         self.arena_length, self.arena_width)
            else:
                QMessageBox.warning(self.viewer.phero,'Error','Please read the localization data first!')
                self.phero_timer.stop()
            # pos = {'0':[0.2,0.2],'5':[0.3,0.4],'3':[0.3,0.1],'4':[0.4,0.3]}
            # self.phero_image = self.phero_model.render_pheromone(pos, self.phero_channel,
            #                                                      self.arena_length, self.arena_width)
                    
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
        self.phero_frame_rate = self.viewer.phero.spinBox_led_v_frame_rate.value()
        self.phero_model.dt = 1.0/self.viewer.phero.spinBox_led_v_frame_rate.value()
        self.phero_timer.start(int(1000/self.phero_frame_rate))
    
    def phero_stop_render(self):
        self.phero_frame_num = 0
        self.phero_timer.stop()
    
    def phero_pause_render(self):
        self.phero_timer.stop()
    
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
