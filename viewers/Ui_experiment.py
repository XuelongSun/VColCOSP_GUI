# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\VColCOSP_GUI\viewers\experiment.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(657, 374)
        Form.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"color: rgb(19, 126, 124);")
        self.pb_save_config = QtWidgets.QPushButton(Form)
        self.pb_save_config.setGeometry(QtCore.QRect(142, 186, 117, 29))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_save_config.setFont(font)
        self.pb_save_config.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_save_config.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_save_config.setObjectName("pb_save_config")
        self.label_28 = QtWidgets.QLabel(Form)
        self.label_28.setGeometry(QtCore.QRect(350, 50, 141, 16))
        font = QtGui.QFont()
        font.setFamily("SimSun")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_28.setFont(font)
        self.label_28.setObjectName("label_28")
        self.label_54 = QtWidgets.QLabel(Form)
        self.label_54.setGeometry(QtCore.QRect(26, 48, 67, 20))
        font = QtGui.QFont()
        font.setFamily("SimSun")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_54.setFont(font)
        self.label_54.setObjectName("label_54")
        self.line_13 = QtWidgets.QFrame(Form)
        self.line_13.setGeometry(QtCore.QRect(386, 12, 251, 25))
        self.line_13.setStyleSheet("color: rgb(19, 126, 124);")
        self.line_13.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_13.setLineWidth(4)
        self.line_13.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_13.setObjectName("line_13")
        self.text_edit_exp_info = QtWidgets.QTextEdit(Form)
        self.text_edit_exp_info.setGeometry(QtCore.QRect(22, 244, 611, 111))
        self.text_edit_exp_info.setReadOnly(True)
        self.text_edit_exp_info.setObjectName("text_edit_exp_info")
        self.label_27 = QtWidgets.QLabel(Form)
        self.label_27.setGeometry(QtCore.QRect(352, 134, 141, 16))
        font = QtGui.QFont()
        font.setFamily("SimSun")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_27.setFont(font)
        self.label_27.setObjectName("label_27")
        self.line_14 = QtWidgets.QFrame(Form)
        self.line_14.setGeometry(QtCore.QRect(14, 12, 247, 25))
        self.line_14.setStyleSheet("color: rgb(19, 126, 124);")
        self.line_14.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_14.setLineWidth(4)
        self.line_14.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_14.setObjectName("line_14")
        self.et_exp_timer = QtWidgets.QLabel(Form)
        self.et_exp_timer.setGeometry(QtCore.QRect(24, 220, 607, 21))
        font = QtGui.QFont()
        font.setFamily("SimSun")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.et_exp_timer.setFont(font)
        self.et_exp_timer.setStyleSheet("color: rgb(19, 126, 124);\n"
"border-top-color: rgb(0, 0, 0);")
        self.et_exp_timer.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.et_exp_timer.setObjectName("et_exp_timer")
        self.pb_save_data = QtWidgets.QPushButton(Form)
        self.pb_save_data.setGeometry(QtCore.QRect(510, 94, 115, 33))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_save_data.setFont(font)
        self.pb_save_data.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_save_data.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_save_data.setObjectName("pb_save_data")
        self.label_56 = QtWidgets.QLabel(Form)
        self.label_56.setGeometry(QtCore.QRect(238, 78, 21, 31))
        font = QtGui.QFont()
        font.setFamily("SimSun")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_56.setFont(font)
        self.label_56.setObjectName("label_56")
        self.line_11 = QtWidgets.QFrame(Form)
        self.line_11.setGeometry(QtCore.QRect(500, 134, 133, 16))
        self.line_11.setStyleSheet("color: rgb(19, 126, 124);")
        self.line_11.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_11.setLineWidth(2)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setObjectName("line_11")
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setGeometry(QtCore.QRect(262, 12, 125, 25))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.pb_add_plot = QtWidgets.QPushButton(Form)
        self.pb_add_plot.setGeometry(QtCore.QRect(352, 156, 79, 29))
        self.pb_add_plot.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_add_plot.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:4px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:4px;\n"
"}\n"
"")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/resource/resources/plot.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pb_add_plot.setIcon(icon)
        self.pb_add_plot.setIconSize(QtCore.QSize(20, 20))
        self.pb_add_plot.setObjectName("pb_add_plot")
        self.pb_destroy_figure = QtWidgets.QPushButton(Form)
        self.pb_destroy_figure.setGeometry(QtCore.QRect(540, 156, 81, 63))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_destroy_figure.setFont(font)
        self.pb_destroy_figure.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_destroy_figure.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:4px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:4px;\n"
"}\n"
"")
        self.pb_destroy_figure.setObjectName("pb_destroy_figure")
        self.lineEdit_exp_name = QtWidgets.QLineEdit(Form)
        self.lineEdit_exp_name.setGeometry(QtCore.QRect(92, 48, 151, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.lineEdit_exp_name.setFont(font)
        self.lineEdit_exp_name.setObjectName("lineEdit_exp_name")
        self.pb_start_exp = QtWidgets.QPushButton(Form)
        self.pb_start_exp.setGeometry(QtCore.QRect(22, 144, 113, 73))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_start_exp.setFont(font)
        self.pb_start_exp.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_start_exp.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_start_exp.setObjectName("pb_start_exp")
        self.pb_add_map = QtWidgets.QPushButton(Form)
        self.pb_add_map.setGeometry(QtCore.QRect(442, 156, 81, 29))
        self.pb_add_map.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_add_map.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:4px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:4px;\n"
"}\n"
"")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/resource/resources/map.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pb_add_map.setIcon(icon1)
        self.pb_add_map.setIconSize(QtCore.QSize(20, 20))
        self.pb_add_map.setObjectName("pb_add_map")
        self.pb_load_config = QtWidgets.QPushButton(Form)
        self.pb_load_config.setGeometry(QtCore.QRect(144, 144, 113, 29))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_load_config.setFont(font)
        self.pb_load_config.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_load_config.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_load_config.setObjectName("pb_load_config")
        self.label_55 = QtWidgets.QLabel(Form)
        self.label_55.setGeometry(QtCore.QRect(26, 78, 147, 31))
        font = QtGui.QFont()
        font.setFamily("SimSun")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_55.setFont(font)
        self.label_55.setObjectName("label_55")
        self.cb_auto_save = QtWidgets.QCheckBox(Form)
        self.cb_auto_save.setGeometry(QtCore.QRect(512, 70, 109, 19))
        self.cb_auto_save.setObjectName("cb_auto_save")
        self.pb_save_data_setting = QtWidgets.QPushButton(Form)
        self.pb_save_data_setting.setGeometry(QtCore.QRect(360, 76, 91, 49))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pb_save_data_setting.setFont(font)
        self.pb_save_data_setting.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_save_data_setting.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_save_data_setting.setObjectName("pb_save_data_setting")
        self.sp_exp_task_interval = QtWidgets.QDoubleSpinBox(Form)
        self.sp_exp_task_interval.setGeometry(QtCore.QRect(188, 78, 51, 31))
        self.sp_exp_task_interval.setStyleSheet("border: 2px soild rgb(19, 126, 124);\n"
"color: rgb(0,255, 0);\n"
"")
        self.sp_exp_task_interval.setProperty("value", 1.0)
        self.sp_exp_task_interval.setObjectName("sp_exp_task_interval")
        self.cb_exp_data_plot = QtWidgets.QCheckBox(Form)
        self.cb_exp_data_plot.setGeometry(QtCore.QRect(28, 118, 231, 19))
        self.cb_exp_data_plot.setObjectName("cb_exp_data_plot")
        self.pb_add_distribution = QtWidgets.QPushButton(Form)
        self.pb_add_distribution.setGeometry(QtCore.QRect(352, 192, 173, 29))
        self.pb_add_distribution.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_add_distribution.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:4px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:4px;\n"
"}\n"
"")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/resource/resources/distribution.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pb_add_distribution.setIcon(icon2)
        self.pb_add_distribution.setIconSize(QtCore.QSize(20, 20))
        self.pb_add_distribution.setObjectName("pb_add_distribution")
        self.line_12 = QtWidgets.QFrame(Form)
        self.line_12.setGeometry(QtCore.QRect(450, 50, 183, 16))
        self.line_12.setStyleSheet("color: rgb(19, 126, 124);")
        self.line_12.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_12.setLineWidth(2)
        self.line_12.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_12.setObjectName("line_12")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pb_save_config.setText(_translate("Form", "Save Config"))
        self.label_28.setText(_translate("Form", "| Data Saving"))
        self.label_54.setText(_translate("Form", "Name:"))
        self.label_27.setText(_translate("Form", "| Data Visualization"))
        self.et_exp_timer.setText(_translate("Form", "Timer"))
        self.pb_save_data.setText(_translate("Form", "Save Data"))
        self.label_56.setText(_translate("Form", "s"))
        self.label_11.setText(_translate("Form", "Experiment"))
        self.pb_add_plot.setText(_translate("Form", "+ Plot"))
        self.pb_destroy_figure.setText(_translate("Form", "Destroy All"))
        self.lineEdit_exp_name.setText(_translate("Form", "sample"))
        self.pb_start_exp.setText(_translate("Form", "Start \n"
" Experiment"))
        self.pb_add_map.setText(_translate("Form", "+ Map"))
        self.pb_load_config.setText(_translate("Form", "Load Config"))
        self.label_55.setText(_translate("Form", "Task Time Interval"))
        self.cb_auto_save.setText(_translate("Form", "Auto Save"))
        self.pb_save_data_setting.setText(_translate("Form", "Settings"))
        self.cb_exp_data_plot.setText(_translate("Form", "experiment data visulization"))
        self.pb_add_distribution.setText(_translate("Form", "+ Distribution"))
import resources_rc