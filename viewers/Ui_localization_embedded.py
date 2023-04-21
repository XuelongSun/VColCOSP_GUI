# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\BaiduSyncdisk\Research\Swarm\VColCOS_GUI\viewers\localization_embedded.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_localization_embedded(object):
    def setupUi(self, localization_embedded):
        localization_embedded.setObjectName("localization_embedded")
        localization_embedded.resize(777, 563)
        localization_embedded.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        localization_embedded.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"color: rgb(19, 126, 124);")
        self.label_41 = QtWidgets.QLabel(localization_embedded)
        self.label_41.setGeometry(QtCore.QRect(154, 120, 41, 16))
        self.label_41.setObjectName("label_41")
        self.sp_arena_w = QtWidgets.QDoubleSpinBox(localization_embedded)
        self.sp_arena_w.setGeometry(QtCore.QRect(84, 136, 61, 22))
        self.sp_arena_w.setStyleSheet("border: 2px soild rgb(19, 126, 124);\n"
"color: rgb(255, 255, 0);\n"
"")
        self.sp_arena_w.setMaximum(10.0)
        self.sp_arena_w.setSingleStep(0.01)
        self.sp_arena_w.setProperty("value", 0.8)
        self.sp_arena_w.setObjectName("sp_arena_w")
        self.label_15 = QtWidgets.QLabel(localization_embedded)
        self.label_15.setGeometry(QtCore.QRect(37, 140, 45, 16))
        self.label_15.setObjectName("label_15")
        self.sp_cal_offset_x = QtWidgets.QDoubleSpinBox(localization_embedded)
        self.sp_cal_offset_x.setGeometry(QtCore.QRect(106, 380, 61, 22))
        self.sp_cal_offset_x.setStyleSheet("border: 2px soild rgb(19, 126, 124);\n"
"color: rgb(255, 255, 0);\n"
"")
        self.sp_cal_offset_x.setMaximum(10.0)
        self.sp_cal_offset_x.setSingleStep(0.01)
        self.sp_cal_offset_x.setProperty("value", 0.0)
        self.sp_cal_offset_x.setObjectName("sp_cal_offset_x")
        self.label_21 = QtWidgets.QLabel(localization_embedded)
        self.label_21.setGeometry(QtCore.QRect(37, 120, 41, 16))
        self.label_21.setObjectName("label_21")
        self.label_59 = QtWidgets.QLabel(localization_embedded)
        self.label_59.setGeometry(QtCore.QRect(22, 100, 181, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_59.setFont(font)
        self.label_59.setObjectName("label_59")
        self.label_42 = QtWidgets.QLabel(localization_embedded)
        self.label_42.setGeometry(QtCore.QRect(154, 140, 41, 16))
        self.label_42.setObjectName("label_42")
        self.label_8 = QtWidgets.QLabel(localization_embedded)
        self.label_8.setGeometry(QtCore.QRect(24, 40, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.frame = QtWidgets.QFrame(localization_embedded)
        self.frame.setGeometry(QtCore.QRect(240, 0, 367, 105))
        self.frame.setStyleSheet("border-image: url(:/resource/resources/localization.png);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.pb_start_calibration = QtWidgets.QPushButton(localization_embedded)
        self.pb_start_calibration.setGeometry(QtCore.QRect(26, 434, 95, 29))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_start_calibration.setFont(font)
        self.pb_start_calibration.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_start_calibration.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_start_calibration.setObjectName("pb_start_calibration")
        self.pb_generate_pattern = QtWidgets.QPushButton(localization_embedded)
        self.pb_generate_pattern.setGeometry(QtCore.QRect(26, 286, 199, 29))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_generate_pattern.setFont(font)
        self.pb_generate_pattern.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_generate_pattern.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_generate_pattern.setObjectName("pb_generate_pattern")
        self.label_60 = QtWidgets.QLabel(localization_embedded)
        self.label_60.setGeometry(QtCore.QRect(22, 160, 63, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_60.setFont(font)
        self.label_60.setObjectName("label_60")
        self.label_22 = QtWidgets.QLabel(localization_embedded)
        self.label_22.setGeometry(QtCore.QRect(37, 182, 41, 16))
        self.label_22.setObjectName("label_22")
        self.label_localization_display = QtWidgets.QLabel(localization_embedded)
        self.label_localization_display.setGeometry(QtCore.QRect(240, 246, 525, 311))
        self.label_localization_display.setWhatsThis("")
        self.label_localization_display.setStyleSheet("border:2px solid rgb(19, 126, 124);")
        self.label_localization_display.setText("")
        self.label_localization_display.setObjectName("label_localization_display")
        self.label_61 = QtWidgets.QLabel(localization_embedded)
        self.label_61.setGeometry(QtCore.QRect(22, 232, 133, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_61.setFont(font)
        self.label_61.setObjectName("label_61")
        self.sp_robot_num = QtWidgets.QSpinBox(localization_embedded)
        self.sp_robot_num.setGeometry(QtCore.QRect(84, 180, 45, 22))
        self.sp_robot_num.setStyleSheet("border: 2px soild rgb(19, 126, 124);\n"
"color: rgb(0,255, 0);\n"
"")
        self.sp_robot_num.setMinimum(1)
        self.sp_robot_num.setMaximum(1000)
        self.sp_robot_num.setSingleStep(1)
        self.sp_robot_num.setProperty("value", 5)
        self.sp_robot_num.setDisplayIntegerBase(10)
        self.sp_robot_num.setObjectName("sp_robot_num")
        self.sp_pattern_num_r = QtWidgets.QSpinBox(localization_embedded)
        self.sp_pattern_num_r.setGeometry(QtCore.QRect(83, 258, 33, 22))
        self.sp_pattern_num_r.setStyleSheet("border: 2px soild rgb(19, 126, 124);\n"
"color: rgb(0,255, 0);\n"
"")
        self.sp_pattern_num_r.setMinimum(1)
        self.sp_pattern_num_r.setMaximum(5)
        self.sp_pattern_num_r.setSingleStep(1)
        self.sp_pattern_num_r.setProperty("value", 4)
        self.sp_pattern_num_r.setDisplayIntegerBase(10)
        self.sp_pattern_num_r.setObjectName("sp_pattern_num_r")
        self.label_23 = QtWidgets.QLabel(localization_embedded)
        self.label_23.setGeometry(QtCore.QRect(36, 260, 41, 16))
        self.label_23.setObjectName("label_23")
        self.label_62 = QtWidgets.QLabel(localization_embedded)
        self.label_62.setGeometry(QtCore.QRect(22, 330, 121, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_62.setFont(font)
        self.label_62.setObjectName("label_62")
        self.label_63 = QtWidgets.QLabel(localization_embedded)
        self.label_63.setGeometry(QtCore.QRect(240, 105, 81, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_63.setFont(font)
        self.label_63.setObjectName("label_63")
        self.hS_brightness = QtWidgets.QSlider(localization_embedded)
        self.hS_brightness.setGeometry(QtCore.QRect(406, 166, 95, 22))
        self.hS_brightness.setStyleSheet("QSlider::groove:horizontal {\n"
"border: 2px solid rgb(19, 126, 124);\n"
"background: qlineargradient(x1: 0, x2: 1, stop: 0 #000, stop: 1 #fff);\n"
"height: 14px;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"background: rgb(19, 126, 124);\n"
"border: 1px solid rgb(19, 126, 124);\n"
"width: 13px;\n"
"margin-top: -4px;\n"
"margin-bottom: -4px;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"background: qlineargradient(x1:0, y1:0, x2:1, y2:1,\n"
"    stop:0 #fff, stop:1 #ddd);\n"
"border: 1px solid #444;\n"
"border-radius: 4px;\n"
"}")
        self.hS_brightness.setMaximum(100)
        self.hS_brightness.setProperty("value", 80)
        self.hS_brightness.setOrientation(QtCore.Qt.Horizontal)
        self.hS_brightness.setObjectName("hS_brightness")
        self.label_39 = QtWidgets.QLabel(localization_embedded)
        self.label_39.setGeometry(QtCore.QRect(356, 146, 41, 16))
        self.label_39.setObjectName("label_39")
        self.label_40 = QtWidgets.QLabel(localization_embedded)
        self.label_40.setGeometry(QtCore.QRect(358, 166, 41, 19))
        self.label_40.setObjectName("label_40")
        self.sp_camera_height = QtWidgets.QSpinBox(localization_embedded)
        self.sp_camera_height.setGeometry(QtCore.QRect(294, 168, 61, 19))
        self.sp_camera_height.setStyleSheet("border: 2px soild rgb(19, 126, 124);\n"
"color: rgb(255, 255, 127);\n"
"")
        self.sp_camera_height.setMaximum(2560)
        self.sp_camera_height.setProperty("value", 1080)
        self.sp_camera_height.setDisplayIntegerBase(10)
        self.sp_camera_height.setObjectName("sp_camera_height")
        self.label_11 = QtWidgets.QLabel(localization_embedded)
        self.label_11.setGeometry(QtCore.QRect(240, 146, 51, 16))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(localization_embedded)
        self.label_12.setGeometry(QtCore.QRect(240, 166, 51, 21))
        self.label_12.setObjectName("label_12")
        self.sp_camera_width = QtWidgets.QSpinBox(localization_embedded)
        self.sp_camera_width.setGeometry(QtCore.QRect(296, 144, 61, 19))
        self.sp_camera_width.setStyleSheet("border: 2px soild rgb(19, 126, 124);\n"
"color: rgb(255, 255, 127);\n"
"")
        self.sp_camera_width.setMaximum(3840)
        self.sp_camera_width.setProperty("value", 1920)
        self.sp_camera_width.setDisplayIntegerBase(10)
        self.sp_camera_width.setObjectName("sp_camera_width")
        self.line_3 = QtWidgets.QFrame(localization_embedded)
        self.line_3.setGeometry(QtCore.QRect(240, 121, 517, 20))
        self.line_3.setStyleSheet("color: rgb(19, 126, 124);")
        self.line_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_3.setLineWidth(2)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(localization_embedded)
        self.line_4.setGeometry(QtCore.QRect(90, 100, 141, 20))
        self.line_4.setStyleSheet("color: rgb(19, 126, 124);")
        self.line_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_4.setLineWidth(2)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(localization_embedded)
        self.line_5.setGeometry(QtCore.QRect(88, 160, 143, 20))
        self.line_5.setStyleSheet("color: rgb(19, 126, 124);")
        self.line_5.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_5.setLineWidth(2)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setObjectName("line_5")
        self.line_6 = QtWidgets.QFrame(localization_embedded)
        self.line_6.setGeometry(QtCore.QRect(156, 232, 77, 20))
        self.line_6.setStyleSheet("color: rgb(19, 126, 124);")
        self.line_6.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_6.setLineWidth(2)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setObjectName("line_6")
        self.line_7 = QtWidgets.QFrame(localization_embedded)
        self.line_7.setGeometry(QtCore.QRect(130, 330, 105, 20))
        self.line_7.setStyleSheet("color: rgb(19, 126, 124);")
        self.line_7.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_7.setLineWidth(2)
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setObjectName("line_7")
        self.label_13 = QtWidgets.QLabel(localization_embedded)
        self.label_13.setGeometry(QtCore.QRect(408, 144, 93, 16))
        self.label_13.setObjectName("label_13")
        self.hS_gain = QtWidgets.QSlider(localization_embedded)
        self.hS_gain.setGeometry(QtCore.QRect(646, 166, 109, 22))
        self.hS_gain.setStyleSheet("QSlider::groove:horizontal {\n"
"border: 2px solid rgb(19, 126, 124);\n"
"background: qlineargradient(x1: 0, x2: 1, stop: 0 #000, stop: 1 #fff);\n"
"height: 14px;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"background: rgb(19, 126, 124);\n"
"border: 1px solid rgb(19, 126, 124);\n"
"width: 13px;\n"
"margin-top: -4px;\n"
"margin-bottom: -4px;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"background: qlineargradient(x1:0, y1:0, x2:1, y2:1,\n"
"    stop:0 #fff, stop:1 #ddd);\n"
"border: 1px solid #444;\n"
"border-radius: 4px;\n"
"}")
        self.hS_gain.setMaximum(32)
        self.hS_gain.setSingleStep(1)
        self.hS_gain.setProperty("value", 12)
        self.hS_gain.setOrientation(QtCore.Qt.Horizontal)
        self.hS_gain.setObjectName("hS_gain")
        self.label_14 = QtWidgets.QLabel(localization_embedded)
        self.label_14.setGeometry(QtCore.QRect(646, 144, 109, 16))
        self.label_14.setObjectName("label_14")
        self.label_16 = QtWidgets.QLabel(localization_embedded)
        self.label_16.setGeometry(QtCore.QRect(518, 146, 111, 16))
        self.label_16.setObjectName("label_16")
        self.hS_exposure = QtWidgets.QSlider(localization_embedded)
        self.hS_exposure.setGeometry(QtCore.QRect(516, 166, 117, 22))
        self.hS_exposure.setStyleSheet("QSlider::groove:horizontal {\n"
"border: 2px solid rgb(19, 126, 124);\n"
"background: qlineargradient(x1: 0, x2: 1, stop: 0 #000, stop: 1 #fff);\n"
"height: 14px;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"background: rgb(19, 126, 124);\n"
"border: 1px solid rgb(19, 126, 124);\n"
"width: 13px;\n"
"margin-top: -4px;\n"
"margin-bottom: -4px;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"background: qlineargradient(x1:0, y1:0, x2:1, y2:1,\n"
"    stop:0 #fff, stop:1 #ddd);\n"
"border: 1px solid #444;\n"
"border-radius: 4px;\n"
"}")
        self.hS_exposure.setMaximum(1000)
        self.hS_exposure.setProperty("value", 50)
        self.hS_exposure.setOrientation(QtCore.Qt.Horizontal)
        self.hS_exposure.setObjectName("hS_exposure")
        self.pb_start = QtWidgets.QPushButton(localization_embedded)
        self.pb_start.setGeometry(QtCore.QRect(546, 220, 91, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_start.setFont(font)
        self.pb_start.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_start.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_start.setObjectName("pb_start")
        self.label_24 = QtWidgets.QLabel(localization_embedded)
        self.label_24.setGeometry(QtCore.QRect(36, 208, 171, 16))
        self.label_24.setObjectName("label_24")
        self.pb_load_id_txt = QtWidgets.QPushButton(localization_embedded)
        self.pb_load_id_txt.setGeometry(QtCore.QRect(140, 180, 73, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_load_id_txt.setFont(font)
        self.pb_load_id_txt.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_load_id_txt.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_load_id_txt.setObjectName("pb_load_id_txt")
        self.cb_show_id = QtWidgets.QCheckBox(localization_embedded)
        self.cb_show_id.setGeometry(QtCore.QRect(242, 222, 41, 19))
        self.cb_show_id.setObjectName("cb_show_id")
        self.cb_show_marker = QtWidgets.QCheckBox(localization_embedded)
        self.cb_show_marker.setGeometry(QtCore.QRect(290, 222, 61, 19))
        self.cb_show_marker.setObjectName("cb_show_marker")
        self.cb_cal_show_axes = QtWidgets.QCheckBox(localization_embedded)
        self.cb_cal_show_axes.setGeometry(QtCore.QRect(28, 506, 197, 23))
        self.cb_cal_show_axes.setObjectName("cb_cal_show_axes")
        self.cb_show_location = QtWidgets.QCheckBox(localization_embedded)
        self.cb_show_location.setGeometry(QtCore.QRect(356, 222, 89, 19))
        self.cb_show_location.setObjectName("cb_show_location")
        self.pb_save_as_img = QtWidgets.QPushButton(localization_embedded)
        self.pb_save_as_img.setGeometry(QtCore.QRect(642, 220, 117, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_save_as_img.setFont(font)
        self.pb_save_as_img.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_save_as_img.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_save_as_img.setObjectName("pb_save_as_img")
        self.pb_check_camera = QtWidgets.QPushButton(localization_embedded)
        self.pb_check_camera.setGeometry(QtCore.QRect(616, 64, 137, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_check_camera.setFont(font)
        self.pb_check_camera.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_check_camera.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_check_camera.setObjectName("pb_check_camera")
        self.sp_pattern_num_c = QtWidgets.QSpinBox(localization_embedded)
        self.sp_pattern_num_c.setGeometry(QtCore.QRect(168, 258, 61, 22))
        self.sp_pattern_num_c.setStyleSheet("border: 2px soild rgb(19, 126, 124);\n"
"color: rgb(0,255, 0);\n"
"")
        self.sp_pattern_num_c.setMinimum(1)
        self.sp_pattern_num_c.setMaximum(5)
        self.sp_pattern_num_c.setSingleStep(1)
        self.sp_pattern_num_c.setProperty("value", 4)
        self.sp_pattern_num_c.setDisplayIntegerBase(10)
        self.sp_pattern_num_c.setObjectName("sp_pattern_num_c")
        self.label_25 = QtWidgets.QLabel(localization_embedded)
        self.label_25.setGeometry(QtCore.QRect(120, 260, 41, 16))
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(localization_embedded)
        self.label_26.setGeometry(QtCore.QRect(26, 358, 81, 16))
        self.label_26.setObjectName("label_26")
        self.sp_chessboard_c = QtWidgets.QSpinBox(localization_embedded)
        self.sp_chessboard_c.setGeometry(QtCore.QRect(110, 354, 43, 22))
        self.sp_chessboard_c.setStyleSheet("border: 2px soild rgb(19, 126, 124);\n"
"color: rgb(0,255, 0);\n"
"")
        self.sp_chessboard_c.setMinimum(2)
        self.sp_chessboard_c.setMaximum(500)
        self.sp_chessboard_c.setSingleStep(1)
        self.sp_chessboard_c.setProperty("value", 15)
        self.sp_chessboard_c.setDisplayIntegerBase(10)
        self.sp_chessboard_c.setObjectName("sp_chessboard_c")
        self.sp_chessboard_r = QtWidgets.QSpinBox(localization_embedded)
        self.sp_chessboard_r.setGeometry(QtCore.QRect(180, 354, 47, 22))
        self.sp_chessboard_r.setStyleSheet("border: 2px soild rgb(19, 126, 124);\n"
"color: rgb(0,255, 0);\n"
"")
        self.sp_chessboard_r.setMinimum(2)
        self.sp_chessboard_r.setMaximum(500)
        self.sp_chessboard_r.setSingleStep(1)
        self.sp_chessboard_r.setProperty("value", 7)
        self.sp_chessboard_r.setDisplayIntegerBase(10)
        self.sp_chessboard_r.setObjectName("sp_chessboard_r")
        self.label_27 = QtWidgets.QLabel(localization_embedded)
        self.label_27.setGeometry(QtCore.QRect(158, 358, 17, 16))
        self.label_27.setObjectName("label_27")
        self.label_28 = QtWidgets.QLabel(localization_embedded)
        self.label_28.setGeometry(QtCore.QRect(26, 382, 57, 16))
        self.label_28.setObjectName("label_28")
        self.label_31 = QtWidgets.QLabel(localization_embedded)
        self.label_31.setGeometry(QtCore.QRect(26, 406, 55, 16))
        self.label_31.setObjectName("label_31")
        self.sp_cal_offset_y = QtWidgets.QDoubleSpinBox(localization_embedded)
        self.sp_cal_offset_y.setGeometry(QtCore.QRect(106, 404, 61, 22))
        self.sp_cal_offset_y.setStyleSheet("border: 2px soild rgb(19, 126, 124);\n"
"color: rgb(255, 255, 0);\n"
"")
        self.sp_cal_offset_y.setMaximum(10.0)
        self.sp_cal_offset_y.setSingleStep(0.01)
        self.sp_cal_offset_y.setProperty("value", 0.0)
        self.sp_cal_offset_y.setObjectName("sp_cal_offset_y")
        self.sp_arena_l = QtWidgets.QDoubleSpinBox(localization_embedded)
        self.sp_arena_l.setGeometry(QtCore.QRect(84, 114, 61, 22))
        self.sp_arena_l.setStyleSheet("border: 2px soild rgb(19, 126, 124);\n"
"color: rgb(255, 255, 0);\n"
"")
        self.sp_arena_l.setMaximum(10.0)
        self.sp_arena_l.setSingleStep(0.01)
        self.sp_arena_l.setProperty("value", 1.4)
        self.sp_arena_l.setObjectName("sp_arena_l")
        self.label_43 = QtWidgets.QLabel(localization_embedded)
        self.label_43.setGeometry(QtCore.QRect(172, 408, 41, 16))
        self.label_43.setObjectName("label_43")
        self.label_44 = QtWidgets.QLabel(localization_embedded)
        self.label_44.setGeometry(QtCore.QRect(172, 386, 41, 16))
        self.label_44.setObjectName("label_44")
        self.label_32 = QtWidgets.QLabel(localization_embedded)
        self.label_32.setGeometry(QtCore.QRect(28, 482, 55, 25))
        self.label_32.setObjectName("label_32")
        self.pb_save_cal_data = QtWidgets.QPushButton(localization_embedded)
        self.pb_save_cal_data.setGeometry(QtCore.QRect(128, 480, 93, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_save_cal_data.setFont(font)
        self.pb_save_cal_data.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_save_cal_data.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_save_cal_data.setObjectName("pb_save_cal_data")
        self.cb_cal_show_points = QtWidgets.QCheckBox(localization_embedded)
        self.cb_cal_show_points.setGeometry(QtCore.QRect(28, 526, 203, 23))
        self.cb_cal_show_points.setToolTip("")
        self.cb_cal_show_points.setObjectName("cb_cal_show_points")
        self.label_64 = QtWidgets.QLabel(localization_embedded)
        self.label_64.setGeometry(QtCore.QRect(240, 190, 231, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_64.setFont(font)
        self.label_64.setObjectName("label_64")
        self.line_8 = QtWidgets.QFrame(localization_embedded)
        self.line_8.setGeometry(QtCore.QRect(238, 210, 517, 7))
        self.line_8.setStyleSheet("color: rgb(19, 126, 124);")
        self.line_8.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_8.setLineWidth(2)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setObjectName("line_8")
        self.cb_show_trajectory = QtWidgets.QCheckBox(localization_embedded)
        self.cb_show_trajectory.setGeometry(QtCore.QRect(452, 222, 85, 19))
        self.cb_show_trajectory.setObjectName("cb_show_trajectory")
        self.pb_start_capture = QtWidgets.QPushButton(localization_embedded)
        self.pb_start_capture.setGeometry(QtCore.QRect(616, 8, 137, 43))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_start_capture.setFont(font)
        self.pb_start_capture.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_start_capture.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_start_capture.setObjectName("pb_start_capture")
        self.cb_cal_show_corner = QtWidgets.QCheckBox(localization_embedded)
        self.cb_cal_show_corner.setGeometry(QtCore.QRect(132, 436, 97, 23))
        self.cb_cal_show_corner.setObjectName("cb_cal_show_corner")

        self.retranslateUi(localization_embedded)
        QtCore.QMetaObject.connectSlotsByName(localization_embedded)

    def retranslateUi(self, localization_embedded):
        _translate = QtCore.QCoreApplication.translate
        localization_embedded.setWindowTitle(_translate("localization_embedded", "VColCOSP-LocalizationPanel"))
        self.label_41.setText(_translate("localization_embedded", "m"))
        self.label_15.setText(_translate("localization_embedded", "Width"))
        self.label_21.setText(_translate("localization_embedded", "Length"))
        self.label_59.setText(_translate("localization_embedded", "| Arena"))
        self.label_42.setText(_translate("localization_embedded", "m"))
        self.label_8.setText(_translate("localization_embedded", "VColCOSP - Localization"))
        self.pb_start_calibration.setText(_translate("localization_embedded", "Run"))
        self.pb_generate_pattern.setText(_translate("localization_embedded", "Generate Patthern"))
        self.label_60.setText(_translate("localization_embedded", "| Robot"))
        self.label_22.setText(_translate("localization_embedded", "number"))
        self.label_localization_display.setToolTip(_translate("localization_embedded", "1"))
        self.label_61.setText(_translate("localization_embedded", "| Visual Pattern"))
        self.label_23.setText(_translate("localization_embedded", "row"))
        self.label_62.setText(_translate("localization_embedded", "| Calibration"))
        self.label_63.setText(_translate("localization_embedded", "| Camera"))
        self.label_39.setText(_translate("localization_embedded", "pixel"))
        self.label_40.setText(_translate("localization_embedded", "pixel"))
        self.label_11.setText(_translate("localization_embedded", "Width"))
        self.label_12.setText(_translate("localization_embedded", "Height"))
        self.label_13.setText(_translate("localization_embedded", "Brightness:"))
        self.label_14.setText(_translate("localization_embedded", "Gain:"))
        self.label_16.setText(_translate("localization_embedded", "Exposure Time:"))
        self.pb_start.setText(_translate("localization_embedded", "Start"))
        self.label_24.setText(_translate("localization_embedded", "ID File:"))
        self.pb_load_id_txt.setText(_translate("localization_embedded", "Load ID "))
        self.cb_show_id.setText(_translate("localization_embedded", "ID"))
        self.cb_show_marker.setText(_translate("localization_embedded", "Marker"))
        self.cb_cal_show_axes.setText(_translate("localization_embedded", "Show World (Arena) Axes"))
        self.cb_show_location.setText(_translate("localization_embedded", "location"))
        self.pb_save_as_img.setText(_translate("localization_embedded", "Save as Picture"))
        self.pb_check_camera.setText(_translate("localization_embedded", "Start Capture"))
        self.label_25.setText(_translate("localization_embedded", "column"))
        self.label_26.setText(_translate("localization_embedded", "Chess Board:"))
        self.label_27.setText(_translate("localization_embedded", "X"))
        self.label_28.setText(_translate("localization_embedded", "Offset X:"))
        self.label_31.setText(_translate("localization_embedded", "Offset Y:"))
        self.label_43.setText(_translate("localization_embedded", "m"))
        self.label_44.setText(_translate("localization_embedded", "m"))
        self.label_32.setText(_translate("localization_embedded", "Results:"))
        self.pb_save_cal_data.setText(_translate("localization_embedded", "Save"))
        self.cb_cal_show_points.setWhatsThis(_translate("localization_embedded", "1"))
        self.cb_cal_show_points.setText(_translate("localization_embedded", "Show Random Testing Points"))
        self.label_64.setText(_translate("localization_embedded", "| Localization Results"))
        self.cb_show_trajectory.setText(_translate("localization_embedded", "Trajectory"))
        self.pb_start_capture.setText(_translate("localization_embedded", "Check Camera"))
        self.cb_cal_show_corner.setText(_translate("localization_embedded", "Show Corners"))
import resources_rc
