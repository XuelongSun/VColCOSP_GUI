# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\VColCOSP_GUI\viewers\exp_results.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_exp_result_win(object):
    def setupUi(self, exp_result_win):
        exp_result_win.setObjectName("exp_result_win")
        exp_result_win.resize(510, 600)
        exp_result_win.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"color: rgb(19, 126, 124);")
        self.verticalLayoutWidget = QtWidgets.QWidget(exp_result_win)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(12, 8, 485, 581))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.et_title = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.et_title.setFont(font)
        self.et_title.setStyleSheet("color: rgb(19, 126, 124);\n"
"border-top-color: rgb(0, 0, 0);")
        self.et_title.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.et_title.setAlignment(QtCore.Qt.AlignCenter)
        self.et_title.setObjectName("et_title")
        self.verticalLayout.addWidget(self.et_title)
        self.info_text = QtWidgets.QTextEdit(self.verticalLayoutWidget)
        self.info_text.setFrameShadow(QtWidgets.QFrame.Plain)
        self.info_text.setReadOnly(True)
        self.info_text.setObjectName("info_text")
        self.verticalLayout.addWidget(self.info_text)

        self.retranslateUi(exp_result_win)
        QtCore.QMetaObject.connectSlotsByName(exp_result_win)

    def retranslateUi(self, exp_result_win):
        _translate = QtCore.QCoreApplication.translate
        exp_result_win.setWindowTitle(_translate("exp_result_win", "Form"))
        self.et_title.setText(_translate("exp_result_win", "Experiment-Sample"))
