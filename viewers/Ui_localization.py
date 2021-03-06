# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\BaiduNetdiskWorkspace\Research\Swarm\VColCOS_GUI\viewers\localization.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_localization(object):
    def setupUi(self, localization):
        localization.setObjectName("localization")
        localization.resize(630, 630)
        localization.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"color: rgb(19, 126, 124);")
        self.label_8 = QtWidgets.QLabel(localization)
        self.label_8.setGeometry(QtCore.QRect(20, 30, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_15 = QtWidgets.QLabel(localization)
        self.label_15.setGeometry(QtCore.QRect(40, 133, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.te_socket_ip = QtWidgets.QTextEdit(localization)
        self.te_socket_ip.setGeometry(QtCore.QRect(80, 130, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.te_socket_ip.setFont(font)
        self.te_socket_ip.setStyleSheet("border:1px solid rgb(19, 126, 124);\n"
"color:rgb(255, 255, 127)")
        self.te_socket_ip.setObjectName("te_socket_ip")
        self.label_22 = QtWidgets.QLabel(localization)
        self.label_22.setGeometry(QtCore.QRect(40, 173, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.te_socket_port = QtWidgets.QTextEdit(localization)
        self.te_socket_port.setGeometry(QtCore.QRect(80, 170, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.te_socket_port.setFont(font)
        self.te_socket_port.setStyleSheet("border:1px solid rgb(19, 126, 124);\n"
"color:rgb(255, 255, 127)")
        self.te_socket_port.setObjectName("te_socket_port")
        self.pb_read_data = QtWidgets.QPushButton(localization)
        self.pb_read_data.setGeometry(QtCore.QRect(520, 130, 91, 71))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pb_read_data.setFont(font)
        self.pb_read_data.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_read_data.setObjectName("pb_read_data")
        self.label_21 = QtWidgets.QLabel(localization)
        self.label_21.setGeometry(QtCore.QRect(20, 110, 391, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.pb_disconnect_to_loc = QtWidgets.QPushButton(localization)
        self.pb_disconnect_to_loc.setGeometry(QtCore.QRect(390, 170, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pb_disconnect_to_loc.setFont(font)
        self.pb_disconnect_to_loc.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_disconnect_to_loc.setObjectName("pb_disconnect_to_loc")
        self.pb_connect_to_loc = QtWidgets.QPushButton(localization)
        self.pb_connect_to_loc.setGeometry(QtCore.QRect(390, 130, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pb_connect_to_loc.setFont(font)
        self.pb_connect_to_loc.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:10px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.pb_connect_to_loc.setObjectName("pb_connect_to_loc")
        self.label_localization_dislay = QtWidgets.QLabel(localization)
        self.label_localization_dislay.setGeometry(QtCore.QRect(23, 242, 591, 361))
        self.label_localization_dislay.setStyleSheet("border:2px solid rgb(19, 126, 124);")
        self.label_localization_dislay.setText("")
        self.label_localization_dislay.setObjectName("label_localization_dislay")
        self.frame = QtWidgets.QFrame(localization)
        self.frame.setGeometry(QtCore.QRect(240, 10, 351, 101))
        self.frame.setStyleSheet("border-image: url(:/resource/resources/localization.png);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        self.retranslateUi(localization)
        QtCore.QMetaObject.connectSlotsByName(localization)

    def retranslateUi(self, localization):
        _translate = QtCore.QCoreApplication.translate
        localization.setWindowTitle(_translate("localization", "VColCOSP-VirtualPheromonePanel"))
        self.label_8.setText(_translate("localization", "VColCOSP - Localization"))
        self.label_15.setText(_translate("localization", "IP"))
        self.te_socket_ip.setToolTip(_translate("localization", "<html><head/><body><p>Define robot <span style=\" font-weight:600;\">ID</span> with <span style=\" font-weight:600;\">Pheromone Type</span> (Color Channel) <span style=\" font-style:italic;\">e.g.: 0:red;1,2:green;</span></p></body></html>"))
        self.te_socket_ip.setHtml(_translate("localization", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">192.168.123.111</span></p></body></html>"))
        self.label_22.setText(_translate("localization", "PORT"))
        self.te_socket_port.setToolTip(_translate("localization", "<html><head/><body><p>Define robot <span style=\" font-weight:600;\">ID</span> with <span style=\" font-weight:600;\">Pheromone Type</span> (Color Channel) <span style=\" font-style:italic;\">e.g.: 0:red;1,2:green;</span></p></body></html>"))
        self.te_socket_port.setHtml(_translate("localization", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">6666</span></p></body></html>"))
        self.pb_read_data.setText(_translate("localization", "ReadData"))
        self.label_21.setText(_translate("localization", "| TCP Connect to LOCALIZATION"))
        self.pb_disconnect_to_loc.setText(_translate("localization", "Disconnect"))
        self.pb_connect_to_loc.setText(_translate("localization", "Connect"))
import resources_rc
