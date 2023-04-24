import datetime

import numpy as np
import cv2
import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QCheckBox
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
from viewers.Ui_data_save_setting import Ui_DataSavingSetting
from viewers.Ui_loc_pattern import Ui_loc_pattern


class WinLocPattern(QMainWindow, Ui_loc_pattern):
    def __init__(self):
        super(WinLocPattern, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())

        self.update_id_combox()
        self.sp_pattern_num_c.valueChanged.connect(self.update_id_combox)
        self.sp_pattern_num_r.valueChanged.connect(self.update_id_combox)
        # self.show_id_text(['0 0.258 0.194\n', '1 0.258 0.355\n', '2 0.258 0.516\n', '3 0.258 0.677\n', '4 0.484 0.194\n', '5 0.484 0.355\n', '6 0.484 0.516\n', '7 0.484 0.677\n', '8 0.71 0.194\n', '9 0.71 0.355\n', '10 0.71 0.516\n', '11 0.71 0.677\n', '12 0.935 0.194\n', '13 0.935 0.355\n', '14 0.935 0.516\n', '15 0.935 0.677\n'],h_id=5)
        
    def update_id_combox(self):
        for i in range(self.sp_pattern_num_c.value()*self.sp_pattern_num_r.value()):
            self.cbox_id_prew.addItem(str(i))
            
    def show_id_text(self, ids, h_id=None):
        self.te_preview_id.clear()
        html = '<p style="font-size:16px">'
        for i_, i in enumerate(ids):
            if i_ == h_id:
                html += ('<font color="yellow">' + i[:-1] + "</font>")
                html += '<br>'
            else:
                html += i[:-1] 
                html += '<br>'

        html += '</p>'
        # html = """
        # <p>{}<p>
        # """.format(id_s)
        
        self.te_preview_id.setHtml(html)

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
        _image = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
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
        # self.label.setScaledContents(True)

    def show_label_img(self, img):
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
    
    def set_window_position(self, x, y, width, height,):
        # set the window size
        self.setGeometry(QRect(x, y, width, height))
        self.label.setGeometry(QRect(0, 0, width, height))


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
        img = cv2.arrowedLine(img, (50,50), (50, 60), self.pos_marker_color, width)
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
    raw_data_display = pyqtSignal(str)
    def __init__(self):
        super(Communication, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.raw_data_display.connect(self._insert_raw_data_display)
    
    def raw_data_insert_text(self, data):
        self.raw_data_display.emit(data)
    
    def _insert_raw_data_display(self, msg):
        self.text_edit_recv_raw.insertPlainText(msg)


class VisualizationPlot(Ui_VisualizationPlot, QMainWindow):
    signal = pyqtSignal(str)
    def __init__(self, index, type='plot') -> None:
        super(VisualizationPlot, self).__init__()
        self.setupUi(self)
        self._types = ['plot', 'map', 'bar']
        self.type = type
        self.plot_index = index
        
        self.generate_figure()
        
        self.COLORS = ['r','g','b','c','m','y','w']
        self.color_in_use = {}
        self.pb_add.clicked.connect(lambda:self.signal.emit('add_'+str(self.plot_index)))
        self.pb_remove.clicked.connect(lambda:self.signal.emit('remove_'+str(self.plot_index)))
        
    def generate_figure(self):
        if self.type == "plot":
            self.figure = pg.PlotWidget()
            self.figure.setBackground('k')
            self.vl_figure.addWidget(self.figure)
            self.legend = self.figure.getPlotItem().addLegend(brush=pg.mkBrush((255,255,255,50)))
            self.lines = {}
        elif self.type == "map":
            self.figure = pg.plot()
            self.scatter_plot = pg.ScatterPlotItem(symbol='d',
                                                   size=20,
                                                   brush=pg.mkBrush('y'))
            self.figure.addItem(self.scatter_plot)
            self.texts = {}
        elif self.type == "distribution":
            self.figure = pg.plot()
            self.legend = pg.LegendItem(brush=pg.mkBrush((255,255,255,50)))
            self.figure.addItem(self.legend)
            self.bars = {}
        
        self.vl_figure.addWidget(self.figure)
            
    def add_plots(self, data_key):
        if self.type == 'plot':
            if not data_key in self.lines.keys():
                # select color
                color_used = [v for k, v in self.color_in_use.items()]
                color_available = set(self.COLORS) - set(color_used)
                color = list(color_available)[0]
                if self.type == 'plot':
                    plot = self.figure.getPlotItem().plot([0, 0], [0, 0], pen=pg.mkPen(color, width=1))
                    self.color_in_use.update({data_key:color})
                    self.lines.update({data_key:plot})
                    self.legend.addItem(plot, data_key)
                    print(self.lines)

        elif self.type == 'map':
            if not data_key in self.texts.keys(): 
                ti = pg.TextItem("Robot_{}".format(data_key),
                                anchor=(0, 0),
                                color='y')
                # self.scatter_plot.addPoints([0], [0])
                self.texts.update({data_key: ti})
                self.figure.addItem(ti)
                print(self.texts)
            
        elif self.type == "distribution":
            # select color
            color_used = [v for k, v in self.color_in_use.items()]
            color_available = set(self.COLORS) - set(color_used)
            color = list(color_available)[0]
            bar_plot = pg.BarGraphItem(brush=pg.mkBrush(color), pen='w', name=data_key)
            self.bars.update({data_key:bar_plot})
            self.figure.addItem(bar_plot)
            self.legend.addItem(bar_plot, data_key)

    def remove_plots(self, data_key):
        if self.type == 'plot':
            if data_key in self.lines.keys():
                self.figure.getPlotItem().removeItem(self.lines[data_key])
                self.legend.removeItem(data_key)
                self.lines.pop(data_key)
                self.color_in_use.pop(data_key)
        elif self.type == 'map':
            if data_key in self.texts.keys():
                self.figure.removeItem(self.texts[data_key])
                self.texts.pop(data_key)
                
        
    def closeEvent(self, event) -> None:
        super().closeEvent(event)
        self.signal.emit('close_' + str(self.plot_index))


class MessageBox(QMainWindow, Ui_message):
    response = pyqtSignal(str)
    def __init__(self):
        super(MessageBox, self).__init__()
        self.setupUi(self)
        self.pb_ok.clicked.connect(self.ok_callback)
        self.pb_cancel.clicked.connect(self.cancel_callback)
    
    def ok_callback(self):
        self.hide()
        self.response.emit("OK")
    
    def cancel_callback(self):
        self.hide()
        self.response.emit("Cancel")


class DialogSaveDataSetting(QMainWindow, Ui_DataSavingSetting):
    answer = pyqtSignal(str)

    def __init__(self, parent=None):
        super(DialogSaveDataSetting, self).__init__(parent)
        self.setupUi(self)
        self.gridLayout_data.setSpacing(8)
        self.gridLayout_data.setSpacing(3)
        self.data_check_box_list = {}
        self.id_check_box_list = {}
        self.pushButton_all_data.clicked.connect(self.check_all_data)
        self.pushButton_all_robot.clicked.connect(self.check_all_id)
        self.pushButton_ok.clicked.connect(self.dialog_answer)
        self.pushButton_cancel.clicked.connect(self.dialog_answer)

    def update_data_checkbox(self, data=None, robot_id=None):
        row_num = 12
        if data is not None:
            for d in data:
                if d not in self.data_check_box_list.keys():
                    i = len(self.data_check_box_list)
                    self.data_check_box_list[d] = QCheckBox()
                    self.data_check_box_list[d].setText(d)
                    self.data_check_box_list[d].setChecked(False)
                    self.data_check_box_list[d].setObjectName("checkBox_data_" + d)
                    self.gridLayout_data.addWidget(self.data_check_box_list[d], int(i % row_num), int(i // row_num))
        row_num = 5
        if robot_id is not None:
            for d in robot_id:
                if d not in self.id_check_box_list.keys():
                    i = len(self.id_check_box_list)
                    self.id_check_box_list[d] = QCheckBox()
                    self.id_check_box_list[d].setText(str(d))
                    self.id_check_box_list[d].setChecked(False)
                    self.id_check_box_list[d].setObjectName("checkBox_id_" + str(d))
                    self.gridLayout_id.addWidget(self.id_check_box_list[d], int(i % row_num), int(i // row_num))
    
    def check_all_data(self):
        if self.pushButton_all_data.text() == 'All DATA':
            for k, v in self.data_check_box_list.items():
                v.setChecked(True)
            self.pushButton_all_data.setText('Reset Data')
        elif self.pushButton_all_data.text() == 'Reset Data':
            for k, v in self.data_check_box_list.items():
                v.setChecked(False)
            self.pushButton_all_data.setText('All DATA')

    def check_all_id(self):
        if self.pushButton_all_robot.text() == 'All ROBOT':
            for k, v in self.id_check_box_list.items():
                v.setChecked(True)
            self.pushButton_all_robot.setText('Reset ROBOT')
        elif self.pushButton_all_robot.text() == 'Reset ROBOT':
            for k, v in self.id_check_box_list.items():
                v.setChecked(False)
            self.pushButton_all_robot.setText('All ROBOT')

    def dialog_answer(self):
        print('data_save_setting: %s' % self.sender().objectName()[11:])
        self.answer.emit(self.sender().objectName()[11:])
                    
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
        self.data_save_setting = DialogSaveDataSetting()
        self.plots = []
        
        self.phero_bg_setting = PheroBgInfoSetting()
        self.loc_pattern = WinLocPattern()
        
        self.logger_str_header = {'error': '--Err: ', 'info': '-Info: ', 'warning': '-Warn: '}
        self.logger_str_color = {'error': 'red', 'info': 'green', 'warning': 'orange'}
    
    def add_visualization_figure(self, data_str, ids, name='plot'):
        self.plots.append(VisualizationPlot(len(self.plots), name))
        if name == 'map':
            self.plots[-1].cbox_data.addItems(ids)
        elif name == 'plot':
            for i in ids:
                self.plots[-1].cbox_data.addItems(["Robot_{}/{}".format(i, d) for d in data_str])
        elif name == 'distribution':
            self.plots[-1].cbox_data.addItems(data_str)
        self.plots[-1].show()
        print(self.plots, self.plots[-1].type, self.plots[-1].plot_index)
    
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
            cursor = self.main_menu.text_edit_exp_info.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.main_menu.text_edit_exp_info.setTextCursor(cursor)
            self.main_menu.text_edit_exp_info.insertHtml(s)
        elif out == "com":
            cursor = self.com.text_edit_com_info.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.com.text_edit_com_info.setTextCursor(cursor)
            self.com.text_edit_com_info.insertHtml(s)
    
    def show_message_box(self, msg, msg_type='info'):
        html_str = "<p>{}</p>"
        self.message_box.textEdit.setHtml(html_str.format(msg))
        self.message_box.show()
    
    def show_label_image(self, label, img):
        if len(img.shape) == 3:
            _image = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
            image = QPixmap(_image)
        else:
            _image = QImage(img[:], img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
            image = QPixmap(_image)
        label.setPixmap(image)
        

if __name__ == '__main__':
    App = QApplication(sys.argv)
    # view = PheroBgInfoSetting()
    view = WinLocPattern()
    view.show()
    sys.exit(App.exec_())

