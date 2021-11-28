import datetime

from PyQt5.QtCore import Qt, pyqtSignal, QRect
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import QMainWindow, QLabel

from viewers.Ui_login import Ui_Login
from viewers.Ui_main_menu import Ui_main_menu
from viewers.Ui_vscene import Ui_vscene
from viewers.Ui_pheromone import Ui_phero
from viewers.Ui_localization import Ui_localization

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
    
        

class LEDScreen(QMainWindow):
    def __init__(self):
        super(LEDScreen, self).__init__()
        self.label = QLabel(self)
        # no frame
        self.setWindowFlags(Qt.FramelessWindowHint)
        # always on top
        # self.setWindowFlags(Qt.WindowStaysOnTopHint)

    def show_label_img(self, x, y, width, height, img):
        # set the window size
        self.setGeometry(QRect(x, y, width, height))
        self.label.setGeometry(QRect(0, 0, width, height))
        # color
        if len(img.shape) == 3:
            _image = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
            image = QPixmap(_image).scaled(self.label.width(), self.label.height())
        # gray
        else:
            _image = QImage(img[:], img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
            image = QPixmap(_image).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(image)


class Viewer:
    def __init__(self):
        self.login = WinLogin()
        self.main_menu = MainMenu()
        self.vscene = VScene()
        self.phero = Pheromone()
        self.loc = Localization()
        
        self.logger_str_header = {'error': '--Err: ', 'info': '-Info: ', 'warning': '-Warn: '}
        self.logger_str_color = {'error': 'red', 'info': 'green', 'warning': 'orange'}
    
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
    
