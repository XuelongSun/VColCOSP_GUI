# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\BaiduNetdiskWorkspace\Research\Swarm\VColCOS_GUI\viewers\visualization_plots.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_VisualizationPlot(object):
    def setupUi(self, VisualizationPlot):
        VisualizationPlot.setObjectName("VisualizationPlot")
        VisualizationPlot.resize(487, 373)
        VisualizationPlot.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"color: rgb(19, 126, 124);")
        self.cbox_data = QtWidgets.QComboBox(VisualizationPlot)
        self.cbox_data.setGeometry(QtCore.QRect(12, 24, 371, 25))
        self.cbox_data.setObjectName("cbox_data")
        self.cbox_data.addItem("")
        self.pb_add = QtWidgets.QPushButton(VisualizationPlot)
        self.pb_add.setGeometry(QtCore.QRect(394, 22, 35, 29))
        self.pb_add.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_add.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:4px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:4px;\n"
"}\n"
"")
        self.pb_add.setIconSize(QtCore.QSize(20, 20))
        self.pb_add.setObjectName("pb_add")
        self.pb_remove = QtWidgets.QPushButton(VisualizationPlot)
        self.pb_remove.setGeometry(QtCore.QRect(438, 22, 35, 29))
        self.pb_remove.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pb_remove.setStyleSheet("QPushButton::disabled {\n"
"    border: 2px solid rgb(149, 149, 149);\n"
"    color:rgb(149, 149, 149);\n"
"    border-radius:4px;\n"
"}\n"
"QPushButton{\n"
"border: 2px solid rgb(19, 126, 124);\n"
"border-radius:4px;\n"
"}\n"
"")
        self.pb_remove.setIconSize(QtCore.QSize(20, 20))
        self.pb_remove.setObjectName("pb_remove")
        self.verticalLayoutWidget = QtWidgets.QWidget(VisualizationPlot)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(14, 64, 457, 297))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.vl_figure = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.vl_figure.setContentsMargins(0, 0, 0, 0)
        self.vl_figure.setObjectName("vl_figure")

        self.retranslateUi(VisualizationPlot)
        QtCore.QMetaObject.connectSlotsByName(VisualizationPlot)

    def retranslateUi(self, VisualizationPlot):
        _translate = QtCore.QCoreApplication.translate
        VisualizationPlot.setWindowTitle(_translate("VisualizationPlot", "ColCOSP-"))
        self.cbox_data.setItemText(0, _translate("VisualizationPlot", "test-sin(t)"))
        self.pb_add.setText(_translate("VisualizationPlot", "+"))
        self.pb_remove.setText(_translate("VisualizationPlot", "-"))
import resources_rc
