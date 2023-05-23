from multiprocessing import Process, Queue, Array

import cv2
import numpy as np
import struct as st
import serial

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QColorDialog
from PyQt5.QtCore import Qt, QTimer

from model import LedImageProcess, PheromoneModel, LocDataModel, SerialDataModel, LocalizationModel,\
    distance
from viewer import LEDScreen, Viewer
from camera import HighFpsCamera


def serial_data_capture(q_paramters:Queue, q_data:Queue):
    p = q_paramters.get()
    serial_port = serial.Serial(baudrate=115200,
                                port=p['port'],
                                parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE,
                                bytesize=serial.EIGHTBITS)
    try:
        serial_port.open()
    except:
        q_data.put({"code":"cannot open port"})
        return
    while p['is_running']:
        if p['serial_port_is_open']:
            d_display = b''
            d = serial_port.read(1)
            if d:
                d_display += d
                # check frame head and end
                if d == b'\x01':
                    d = serial_port.read(1)
                    d_display += d
                    if d == b'\xfe':
                        # got frame head, then read {self.serial_package_len} bytes
                        data = serial_port.read(int(p['recv_pack_len']))
                        d_display += data
                        # check frame end
                        d = serial_port.read(2)
                        if d == b'\xfe\x01':
                            # got frame end
                            # update robot data
                            d_display += d
                            q_data.put({"d_display":d_display})