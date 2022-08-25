import datetime

import numpy as np
import cv2
import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, pyqtSignal, QRect
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import QMainWindow, QLabel, QColorDialog, QFontDialog, QWidget

from viewers.Ui_login import Ui_Login
from viewers.Ui_main_menu import Ui_main_menu
from viewers.Ui_vscene import Ui_vscene
from viewers.Ui_pheromone import Ui_phero
from viewers.Ui_localization import Ui_localization
from viewers.Ui_localization_embedded import Ui_localization_embedded
from viewers.Ui_phero_bg_info_setting import Ui_phero_bg_info_setting
from viewers.Ui_communication import Ui_com
from viewers.Ui_visualization_plots import Ui_VisualizationPlot
from viewers.Ui_message_box import Ui_message

class WinLogin(QMainWindow, Ui_Login):
    def __init__(self):
        super(WinLogin, self).__init__()
        self.setupUi(self)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.pushButton_close.clicked.connect(self.close)


class MainMenu(QMainWindow, Ui_main_menu):
    def __init__(self):
        super(MainMenu, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
    

class VScene(QMainWindow, Ui_vscene):
    signal = pyqtSignal(str)
    
    def __init__(self):
        super(VScene, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
    
    def closeEvent(self, event):
        print('closing the visual scene window')
        self.signal.emit('close')

class Pheromone(QMainWindow, Ui_phero):
    signal = pyqtSignal(str)
    
    def __init__(self):
        super(Pheromone, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.sp_d_kernel_s_r.valueChanged.connect(self.d_kernel_size_change)
        self.sp_d_kernel_s_g.valueChanged.connect(self.d_kernel_size_change)
        self.sp_d_kernel_s_b.valueChanged.connect(self.d_kernel_size_change)
    
    def d_kernel_size_change(self):
        # force kernel size to be odd
        v = self.sender().value()
        if v % 2 == 0:
            self.sender().setValue(v + 1)
        
    def closeEvent(self, event):
        print('closing the pheromone window')
        self.signal.emit('close')
    
    def show_label_image(self, label, img):

        if len(img.shape) == 3:
            _image = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
            # image = QPixmap(_image).scaled(label.width(), label.height())
            image = QPixmap(_image)
        else:
            _image = QImage(img[:], img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
            image = QPixmap(_image)

            # image = QPixmap(_image).scaled(label.width(), label.height())
        
        label.setPixmap(image)
        

class Localization(QMainWindow, Ui_localization):
    signal = pyqtSignal(str)
    
    def __init__(self):
        super(Localization, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.label_localization_dislay.setScaledContents(True)
    
    def update_localization_dislay(self, img):
        _image = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        # image = QPixmap(_image).scaled(self.label_localization_dislay.width(), 
        #                                self.loc_label_view.height())
        self.label_localization_dislay.setPixmap(QPixmap(_image))
    
    def closeEvent(self, event):
        print('closing the localization window')
        self.signal.emit('close')


class LocalizationEmbedded(QMainWindow, Ui_localization_embedded):
    signal = pyqtSignal(str)
    def __init__(self):
        super(LocalizationEmbedded, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.label_localization_display.setScaledContents(True)
    
    def update_localization_display(self, img):
        _image = QImage(img[:], img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
        self.label_localization_display.setPixmap(QPixmap(_image))
    
    def closeEvent(self, event):
        print('closing the localization window')
        self.signal.emit('close')
    
class LEDScreen(QMainWindow):
    def __init__(self):
        super(LEDScreen, self).__init__()
        self.label = QLabel(self)
        # no frame
        self.setWindowFlags(Qt.FramelessWindowHint)
        # always on top
        # self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.label.setScaledContents(True)

    def show_label_img(self, x, y, width, height, img):
        # set the window size
        self.setGeometry(QRect(x, y, width, height))
        self.label.setGeometry(QRect(0, 0, width, height))
        # color
        if len(img.shape) == 3:
            _image = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
            image = QPixmap(_image)
            # image = QPixmap(_image).scaled(self.label.width(), self.label.height())
        # gray
        else:
            _image = QImage(img[:], img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
            # image = QPixmap(_image).scaled(self.label.width(), self.label.height())
            image = QPixmap(_image)
        self.label.setPixmap(image)


class PheroBgInfoSetting(QMainWindow, Ui_phero_bg_info_setting):
    signal = pyqtSignal(str)
    
    def __init__(self):
        super(PheroBgInfoSetting, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        
        self.lv_pos_text.setScaledContents(True)
        self.pb_pos_text_color.clicked.connect(self.pos_text_color_pick)
        self.sb_pos_text_width.valueChanged.connect(self.pos_text_image_update)
        self.pos_text_color = (255,255,255)
        
        self.lv_pos_line.setScaledContents(True)
        self.pb_pos_line_color.clicked.connect(self.pos_line_color_pick)
        self.sb_pos_line_width.valueChanged.connect(self.pos_line_image_update)
        self.pos_line_color = (125,125,125)
        
        self.lv_pos_marker.setScaledContents(True)
        self.pb_pos_marker_color.clicked.connect(self.pos_marker_color_pick)
        self.sb_pos_marker_width.valueChanged.connect(self.pos_marker_image_update)
        self.pos_marker_color = (0,255,255)
        
        self.lv_arena_border.setScaledContents(True)
        self.pb_arena_border_color.clicked.connect(self.arena_border_color_pick)
        self.sb_arena_border_width.valueChanged.connect(self.arena_border_image_update)
        self.sb_arena_border_margin.valueChanged.connect(self.arena_border_image_update)
        self.arena_border_color = (255,255,255)
        
        self.pb_ok.clicked.connect(self.click_ok)
        self.pb_cancel.clicked.connect(self.click_cancel)
        
        self.pos_text_image_update()
        self.pos_marker_image_update()
        self.pos_line_image_update()
        self.arena_border_image_update()
    
    def pos_text_color_pick(self):
        col = QColorDialog.getColor()
        self.pos_text_color = (col.red(),col.green(),col.blue())
        self.pos_text_image_update()
    
    def pos_line_color_pick(self):
        col = QColorDialog.getColor()
        self.pos_line_color = (col.red(),col.green(),col.blue())
        self.pos_line_image_update()
        
    def pos_marker_color_pick(self):
        col = QColorDialog.getColor()
        self.pos_marker_color = (col.red(),col.green(),col.blue())
        self.pos_marker_image_update()
    
    def arena_border_color_pick(self):
        col = QColorDialog.getColor()
        self.arena_border_color = (col.red(),col.green(),col.blue())
        self.arena_border_image_update()
    
    def pos_text_image_update(self):
        img = np.zeros([100, 100, 3],
                       dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        width = int(self.sb_pos_text_width.value())
        img = cv2.putText(img, '5', 
                          (30, 70), font, 2, self.pos_text_color, width)
        _image = QImage(img[:], img.shape[1], img.shape[0], 
                        img.shape[1] * 3, QImage.Format_RGB888)
        image = QPixmap(_image)
        self.lv_pos_text.setPixmap(image)
    
    def pos_line_image_update(self):
        img = np.zeros([100, 100, 3],
                       dtype=np.uint8)
        width = int(self.sb_pos_line_width.value())
        img = cv2.line(img, (0,50),(100,50), self.pos_line_color, width)
        img = cv2.line(img, (50,0),(50,100), self.pos_line_color, width)
        _image = QImage(img[:], img.shape[1], img.shape[0], 
                        img.shape[1] * 3, QImage.Format_RGB888)
        image = QPixmap(_image)
        self.lv_pos_line.setPixmap(image)
    
    def pos_marker_image_update(self):
        img = np.zeros([100, 100, 3],
                       dtype=np.uint8)
        width = int(self.sb_pos_marker_width.value())
        img = cv2.circle(img, (50,50), 20, self.pos_marker_color, width)
        _image = QImage(img[:], img.shape[1], img.shape[0], 
                        img.shape[1] * 3, QImage.Format_RGB888)
        image = QPixmap(_image)
        self.lv_pos_marker.setPixmap(image)
    
    def arena_border_image_update(self):
        img = np.zeros([100, 100, 3],
                       dtype=np.uint8)
        width = int(self.sb_arena_border_width.value())
        margin = int(self.sb_arena_border_margin.value())
        img = cv2.rectangle(img, (margin,margin), 
                            (100-margin,100-margin), 
                            self.arena_border_color, width)
        _image = QImage(img[:], img.shape[1], img.shape[0], 
                        img.shape[1] * 3, QImage.Format_RGB888)
        image = QPixmap(_image)
        self.lv_arena_border.setPixmap(image)
        
    def click_ok(self):
        self.signal.emit('OK')
    
    def click_cancel(self):
        self.signal.emit('Cancel')


class Communication(Ui_com, QMainWindow):
    def __init__(self):
        super(Communication, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())


class VisualizationPlot(Ui_VisualizationPlot, QMainWindow):
    close_signal = pyqtSignal(str)
    def __init__(self, index, type='plot') -> None:
        super(VisualizationPlot, self).__init__()
        self.setupUi(self)
        self._types = ['plot', 'map', 'bar']
        self.type = type
        self.plot_index = index
        self.generate_figure()
    
    def generate_figure(self):
        self.figure = pg.PlotWidget()
        self.figure.setBackground('k')
        self.vl_figure.addWidget(self.figure)
        # if self.type == 'plot':

        # elif self.type == 'map':
        #     self.figure = pg.ScatterPlotWidget()
        #     self.figure.setBackground('k')
        # elif self.type == 'bar':
        #     self.figure = pg.BarGraphItem()
        #     self.figure.setBackground('k')
        # else:
        #     pass
    
    def closeEvent(self, event) -> None:
        super().closeEvent(event)
        self.close_signal.emit('close')


class MessageBox(QMainWindow, Ui_message):
    def __init__(self):
        super(MessageBox, self).__init__()
        self.setupUi(self)

class Viewer:
    def __init__(self):
        self.login = WinLogin()
        self.main_menu = MainMenu()
        self.vscene = VScene()
        self.phero = Pheromone()
        self.loc = Localization()
        self.loc_embedded = LocalizationEmbedded()
        self.com = Communication()
        self.message_box = MessageBox()
        self.plots = []
        
        self.phero_bg_setting = PheroBgInfoSetting()
        
        self.logger_str_header = {'error': '--Err: ', 'info': '-Info: ', 'warning': '-Warn: '}
        self.logger_str_color = {'error': 'red', 'info': 'green', 'warning': 'orange'}
    
    def add_visualization_figure(self, name='plot'):
        self.plots.append(VisualizationPlot(len(self.plots),name))
        self.plots[-1].show()
        self.plots[-1].close_signal.connect(lambda: self.plots.remove(self.plots[-1]))
        print(self.plots)
    
    def remove_visualization_figure(self, all=False):
        if all:
            pass
        else:
            pass
    
    def system_logger(self, log, log_type='info', out='system'):
        time_e = datetime.datetime.now()
        time_e_s = datetime.datetime.strftime(time_e, '%Y-%m-%d %H:%M:%S')[-8:]

        if log_type in self.logger_str_header.keys():
            header = self.logger_str_header[log_type]
            color = self.logger_str_color[log_type]
        else:
            header = '^-^'
            color = 'black'

        log_str = time_e_s + header + log
        s = "<font color=%s>%s</font><br>" % (color, log_str)
        if out == 'system':
            cursor = self.main_menu.text_edit_sys_info.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.main_menu.text_edit_sys_info.setTextCursor(cursor)
            self.main_menu.text_edit_sys_info.insertHtml(s)
        elif out == 'exp':
            cursor = self.text_edit_exp_info.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.text_edit_exp_info.setTextCursor(cursor)
            self.text_edit_exp_info.insertHtml(s)
    
    def show_message_box(self, msg, msg_type='info'):
        html_str = "<p>{}</p>"
        self.message_box.textEdit.setHtml(html_str.format(msg))
        self.message_box.show()
        

if __name__ == '__main__':
    App = QApplication(sys.argv)
    view = PheroBgInfoSetting()
    view.show()
    sys.exit(App.exec_())

