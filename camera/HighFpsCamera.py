#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('./camera')
from ImageConvert import *
from MVSDK import *
import struct
import time
import datetime
import numpy
import cv2 as cv
import gc

g_cameraStatusUserInfo = b"statusInfo"

class Camera:
	def __init__(self):
		# self.connectCallBackFuncEx = connectCallBackEx(self.deviceLinkNotify)
		self.connectCallBackFuncEx = connectCallBackEx(self.deviceLinkNotify)
		self.frameCallbackFuncEx = callbackFuncEx(self.onGetFrameEx)
		self.usr_image = c_buffer(b'\0', 1920*1200)
		self.g_isStop = False

	def onGetFrameEx(self, frame, userInfo):

		if (self.g_isStop == True):
			return

		nRet = frame.contents.valid(frame)
		if ( nRet != 0):
			print("frame is invalid!")
			# 释放驱动图像缓存资源
			frame.contents.release(frame)
			return

		# 此处客户应用程序应将图像拷贝出使用
		# 给转码所需的参数赋值
		imageParams = IMGCNV_SOpenParam()
		imageParams.dataSize	= frame.contents.getImageSize(frame)
		imageParams.height	  = frame.contents.getImageHeight(frame)
		imageParams.width	   = frame.contents.getImageWidth(frame)
		imageParams.paddingX	= frame.contents.getImagePaddingX(frame)
		imageParams.paddingY	= frame.contents.getImagePaddingY(frame)
		imageParams.pixelForamt = frame.contents.getImagePixelFormat(frame)

		# 将裸数据图像拷出
		imageBuff = frame.contents.getImage(frame)
		# userBuff = c_buffer(b'\0', imageParams.dataSize)
		# usr_image = c_buffer(b'\0', imageParams.dataSize)
		memmove(self.usr_image, c_char_p(imageBuff), imageParams.dataSize)
		
		
		# 释放驱动图像缓存资源
		frame.contents.release(frame)

		# 如果图像格式是 Mono8 直接使用
		# if imageParams.pixelForamt == EPixelType.gvspPixelMono8:
		# 	grayByteArray = bytearray(userBuff)
		# 	cvImage = numpy.array(grayByteArray).reshape(imageParams.height, imageParams.width)
		# else:
		# 	# 转码 => BGR24
		# 	pass
			# rgbSize = c_int()
			# rgbBuff = c_buffer(b'\0', imageParams.height * imageParams.width * 3)

			# nRet = IMGCNV_ConvertToBGR24(cast(userBuff, c_void_p), \
			# 							 byref(imageParams), \
			# 							 cast(rgbBuff, c_void_p), \
			# 							 byref(rgbSize))

			# colorByteArray = bytearray(rgbBuff)
			# print("IMGCNV_ConvertToBGR24 time = %4f" %(time.time()-t))
			# cvImage = numpy.array(colorByteArray).reshape(imageParams.height, imageParams.width, 3)
	#########################################################################################################
		# grayByteArray = bytearray(userBuff)
		# cvImage = numpy.array(grayByteArray).reshape(imageParams.height, imageParams.width)
		# cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BAYER_BG2GRAY)
		# show_Image = cv2.resize(cvImage, (960, 600))

	# --- end if ---

		# cv2.imshow('myWindow', show_Image)
		# gc.collect()
		# cv2.waitKey(1)
		# print("IMGCNV_ConvertToBGR24 time = %4f" %(time.time()-t))


	# 相机连接状态回调函数
	def deviceLinkNotify(self,connectArg, linkInfo):
		if ( EVType.offLine == connectArg.contents.m_event ):
			print("camera has off line, userInfo [%s]" %(c_char_p(linkInfo).value))
		elif ( EVType.onLine == connectArg.contents.m_event ):
			print("camera has on line, userInfo [%s]" %(c_char_p(linkInfo).value))
		self.connectCallBackFuncEx = connectCallBackEx(self.deviceLinkNotify)
		self.frameCallbackFuncEx = callbackFuncEx(self.onGetFrameEx)

	

	# 注册相机连接状态回调
	def subscribeCameraStatus(self,camera):
		# 注册上下线通知
		eventSubscribe = pointer(GENICAM_EventSubscribe())
		eventSubscribeInfo = GENICAM_EventSubscribeInfo()
		eventSubscribeInfo.pCamera = pointer(camera)
		nRet = GENICAM_createEventSubscribe(byref(eventSubscribeInfo), byref(eventSubscribe))
		if ( nRet != 0):
			print("create eventSubscribe fail!")
			return -1
		
		nRet = eventSubscribe.contents.subscribeConnectArgsEx(eventSubscribe, self.connectCallBackFuncEx, g_cameraStatusUserInfo)
		if ( nRet != 0 ):
			print("subscribeConnectArgsEx fail!")
			# 释放相关资源
			eventSubscribe.contents.release(eventSubscribe)
			return -1  
		
		# 不再使用时，需释放相关资源
		eventSubscribe.contents.release(eventSubscribe) 
		return 0

	# 反注册相机连接状态回调
	def unsubscribeCameraStatus(self,camera):
		# 反注册上下线通知
		eventSubscribe = pointer(GENICAM_EventSubscribe())
		eventSubscribeInfo = GENICAM_EventSubscribeInfo()
		eventSubscribeInfo.pCamera = pointer(camera)
		nRet = GENICAM_createEventSubscribe(byref(eventSubscribeInfo), byref(eventSubscribe))
		if ( nRet != 0):
			print("create eventSubscribe fail!")
			return -1
			
		nRet = eventSubscribe.contents.unsubscribeConnectArgsEx(eventSubscribe, self.connectCallBackFuncEx, g_cameraStatusUserInfo)
		if ( nRet != 0 ):
			print("unsubscribeConnectArgsEx fail!")
			# 释放相关资源
			eventSubscribe.contents.release(eventSubscribe)
			return -1
		
		# 不再使用时，需释放相关资源
		eventSubscribe.contents.release(eventSubscribe)
		return 0   

	# 设置软触发
	def setSoftTriggerConf(self,camera):
		# 创建control节点
		acqCtrlInfo = GENICAM_AcquisitionControlInfo()
		acqCtrlInfo.pCamera = pointer(camera)
		acqCtrl = pointer(GENICAM_AcquisitionControl())
		nRet = GENICAM_createAcquisitionControl(pointer(acqCtrlInfo), byref(acqCtrl))
		if ( nRet != 0 ):
			print("create AcquisitionControl fail!")
			return -1
		
		# 设置触发源为软触发
		trigSourceEnumNode = acqCtrl.contents.triggerSource(acqCtrl)
		nRet = trigSourceEnumNode.setValueBySymbol(byref(trigSourceEnumNode), b"Software")
		if ( nRet != 0 ):
			print("set TriggerSource value [Software] fail!")
			# 释放相关资源
			trigSourceEnumNode.release(byref(trigSourceEnumNode))
			acqCtrl.contents.release(acqCtrl)
			return -1
		
		# 需要释放Node资源
		trigSourceEnumNode.release(byref(trigSourceEnumNode))
		
		# 设置触发方式
		trigSelectorEnumNode = acqCtrl.contents.triggerSelector(acqCtrl)
		nRet = trigSelectorEnumNode.setValueBySymbol(byref(trigSelectorEnumNode), b"FrameStart")
		if ( nRet != 0 ):
			print("set TriggerSelector value [FrameStart] fail!")
			# 释放相关资源
			trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
			acqCtrl.contents.release(acqCtrl) 
			return -1
		
		# 需要释放Node资源	
		trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
		
		# 打开触发模式
		trigModeEnumNode = acqCtrl.contents.triggerMode(acqCtrl)
		nRet = trigModeEnumNode.setValueBySymbol(byref(trigModeEnumNode), b"On")
		if ( nRet != 0 ):
			print("set TriggerMode value [On] fail!")
			# 释放相关资源
			trigModeEnumNode.release(byref(trigModeEnumNode))
			acqCtrl.contents.release(acqCtrl)
			return -1
		
		# 需要释放相关资源	
		trigModeEnumNode.release(byref(trigModeEnumNode))
		acqCtrl.contents.release(acqCtrl)
		
		return 0	 

	# 设置外触发
	def setLineTriggerConf(self,camera):
		# 创建control节点
		acqCtrlInfo = GENICAM_AcquisitionControlInfo()
		acqCtrlInfo.pCamera = pointer(camera)
		acqCtrl = pointer(GENICAM_AcquisitionControl())
		nRet = GENICAM_createAcquisitionControl(pointer(acqCtrlInfo), byref(acqCtrl))
		if ( nRet != 0 ):
			print("create AcquisitionControl fail!")
			return -1
		
		# 设置触发源为软触发
		trigSourceEnumNode = acqCtrl.contents.triggerSource(acqCtrl)
		nRet = trigSourceEnumNode.setValueBySymbol(byref(trigSourceEnumNode), b"Line1")
		if ( nRet != 0 ):
			print("set TriggerSource value [Line1] fail!")
			# 释放相关资源
			trigSourceEnumNode.release(byref(trigSourceEnumNode))
			acqCtrl.contents.release(acqCtrl)
			return -1
		
		# 需要释放Node资源
		trigSourceEnumNode.release(byref(trigSourceEnumNode))
		
		# 设置触发方式
		trigSelectorEnumNode = acqCtrl.contents.triggerSelector(acqCtrl)
		nRet = trigSelectorEnumNode.setValueBySymbol(byref(trigSelectorEnumNode), b"FrameStart")
		if ( nRet != 0 ):
			print("set TriggerSelector value [FrameStart] fail!")
			# 释放相关资源
			trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
			acqCtrl.contents.release(acqCtrl) 
			return -1
		
		# 需要释放Node资源	
		trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
		
		# 打开触发模式
		trigModeEnumNode = acqCtrl.contents.triggerMode(acqCtrl)
		nRet = trigModeEnumNode.setValueBySymbol(byref(trigModeEnumNode), b"On")
		if ( nRet != 0 ):
			print("set TriggerMode value [On] fail!")
			# 释放相关资源
			trigModeEnumNode.release(byref(trigModeEnumNode))
			acqCtrl.contents.release(acqCtrl)
			return -1
		
		# 需要释放Node资源	
		trigModeEnumNode.release(byref(trigModeEnumNode))
		
		# 设置触发沿
		trigActivationEnumNode = acqCtrl.contents.triggerActivation(acqCtrl)
		nRet = trigActivationEnumNode.setValueBySymbol(byref(trigActivationEnumNode), b"RisingEdge")
		if ( nRet != 0 ):
			print("set TriggerActivation value [RisingEdge] fail!")
			# 释放相关资源
			trigActivationEnumNode.release(byref(trigActivationEnumNode))
			acqCtrl.contents.release(acqCtrl)
			return -1
		
		# 需要释放Node资源	
		trigActivationEnumNode.release(byref(trigActivationEnumNode))
		acqCtrl.contents.release(acqCtrl)
		return 0 

	# 打开相机
	def openCamera(self,camera):
		# 连接相机
		nRet = camera.connect(camera, c_int(GENICAM_ECameraAccessPermission.accessPermissionControl))
		if ( nRet != 0 ):
			print("camera connect fail!")
			return -1
		else:
			print("camera connect success.")
	
		# 注册相机连接状态回调
		nRet = self.subscribeCameraStatus(camera)
		if ( nRet != 0 ):
			print("subscribeCameraStatus fail!")
			return -1

		return 0

	# 关闭相机
	def closeCamera(self,camera):
		# 反注册相机连接状态回调
		nRet = self.unsubscribeCameraStatus(camera)
		if ( nRet != 0 ):
			print("unsubscribeCameraStatus fail!")
			return -1
	
		# 断开相机
		nRet = camera.disConnect(byref(camera))
		if ( nRet != 0 ):
			print("disConnect camera fail!")
			return -1
		
		return 0	

	# 设置曝光
	def setExposureTime(self,camera, dVal):
		# 通用属性设置:设置曝光 --根据属性类型，直接构造属性节点。如曝光是 double类型，构造doubleNode节点
		exposureTimeNode = pointer(GENICAM_DoubleNode())
		exposureTimeNodeInfo = GENICAM_DoubleNodeInfo() 
		exposureTimeNodeInfo.pCamera = pointer(camera)
		exposureTimeNodeInfo.attrName = b"ExposureTime"
		nRet = GENICAM_createDoubleNode(byref(exposureTimeNodeInfo), byref(exposureTimeNode))
		if ( nRet != 0 ):
			print("create ExposureTime Node fail!")
			return -1
		
		# 设置曝光时间
		nRet = exposureTimeNode.contents.setValue(exposureTimeNode, c_double(dVal))  
		if ( nRet != 0 ):
			print("set ExposureTime value [%f]us fail!"  % (dVal))
			# 释放相关资源
			exposureTimeNode.contents.release(exposureTimeNode)
			return -1
		else:
			print("set ExposureTime value [%f]us success." % (dVal))
				
		# 释放节点资源	 
		exposureTimeNode.contents.release(exposureTimeNode)	
		return 0
		
	# 枚举相机
	def enumCameras(self):
		# 获取系统单例
		system = pointer(GENICAM_System())
		nRet = GENICAM_getSystemInstance(byref(system))
		if ( nRet != 0 ):
			print("getSystemInstance fail!")
			return None, None

		# 发现相机 
		cameraList = pointer(GENICAM_Camera()) 
		cameraCnt = c_uint()
		nRet = system.contents.discovery(system, byref(cameraList), byref(cameraCnt), c_int(GENICAM_EProtocolType.typeAll));
		if ( nRet != 0 ):
			print("discovery fail!")
			return None, None
		elif cameraCnt.value < 1:
			print("discovery no camera!")
			return None, None
		else:
			print("cameraCnt: " + str(cameraCnt.value))
			return cameraCnt.value, cameraList

	def grabOne(self,camera):
		# 创建流对象
		streamSourceInfo = GENICAM_StreamSourceInfo()
		streamSourceInfo.channelId = 0
		streamSourceInfo.pCamera = pointer(camera)
		
		streamSource = pointer(GENICAM_StreamSource())
		nRet = GENICAM_createStreamSource(pointer(streamSourceInfo), byref(streamSource))
		if ( nRet != 0 ):
			print("create StreamSource fail!")	 
			return -1
		
		# 创建control节点
		acqCtrlInfo = GENICAM_AcquisitionControlInfo()
		acqCtrlInfo.pCamera = pointer(camera)
		acqCtrl = pointer(GENICAM_AcquisitionControl())
		nRet = GENICAM_createAcquisitionControl(pointer(acqCtrlInfo), byref(acqCtrl))
		if ( nRet != 0 ):
			print("create AcquisitionControl fail!")
			# 释放相关资源
			streamSource.contents.release(streamSource)  
			return -1
			
		# 执行一次软触发
		trigSoftwareCmdNode = acqCtrl.contents.triggerSoftware(acqCtrl)
		nRet = trigSoftwareCmdNode.execute(byref(trigSoftwareCmdNode))
		if( nRet != 0 ):
			print("Execute triggerSoftware fail!")
			# 释放相关资源
			trigSoftwareCmdNode.release(byref(trigSoftwareCmdNode))
			acqCtrl.contents.release(acqCtrl)
			streamSource.contents.release(streamSource)   
			return -1   

		# 释放相关资源
		trigSoftwareCmdNode.release(byref(trigSoftwareCmdNode))
		acqCtrl.contents.release(acqCtrl)
		streamSource.contents.release(streamSource) 
		
		return 0  

	# 设置感兴趣区域  --- 感兴趣区域的宽高 和 xy方向的偏移量  入参值应符合对应相机的递增规则
	def setROI(self,camera, OffsetX, OffsetY, nWidth, nHeight):
		#获取原始的宽度
		widthMaxNode = pointer(GENICAM_IntNode())
		widthMaxNodeInfo = GENICAM_IntNodeInfo() 
		widthMaxNodeInfo.pCamera = pointer(camera)
		widthMaxNodeInfo.attrName = b"WidthMax"
		nRet = GENICAM_createIntNode(byref(widthMaxNodeInfo), byref(widthMaxNode))
		if ( nRet != 0 ):
			print("create WidthMax Node fail!")
			return -1
		
		oriWidth = c_longlong()
		nRet = widthMaxNode.contents.getValue(widthMaxNode, byref(oriWidth))
		if ( nRet != 0 ):
			print("widthMaxNode getValue fail!")
			# 释放相关资源
			widthMaxNode.contents.release(widthMaxNode)
			return -1  
		
		# 释放相关资源
		widthMaxNode.contents.release(widthMaxNode)
		
		# 获取原始的高度
		heightMaxNode = pointer(GENICAM_IntNode())
		heightMaxNodeInfo = GENICAM_IntNodeInfo() 
		heightMaxNodeInfo.pCamera = pointer(camera)
		heightMaxNodeInfo.attrName = b"HeightMax"
		nRet = GENICAM_createIntNode(byref(heightMaxNodeInfo), byref(heightMaxNode))
		if ( nRet != 0 ):
			print("create HeightMax Node fail!")
			return -1
		
		oriHeight = c_longlong()
		nRet = heightMaxNode.contents.getValue(heightMaxNode, byref(oriHeight))
		if ( nRet != 0 ):
			print("heightMaxNode getValue fail!")
			# 释放相关资源
			heightMaxNode.contents.release(heightMaxNode)
			return -1
		
		# 释放相关资源
		heightMaxNode.contents.release(heightMaxNode)
			
		# 检验参数
		if ( ( oriWidth.value < (OffsetX + nWidth)) or ( oriHeight.value < (OffsetY + nHeight)) ):
			print("please check input param!")
			return -1
		
		# 设置宽度
		widthNode = pointer(GENICAM_IntNode())
		widthNodeInfo = GENICAM_IntNodeInfo() 
		widthNodeInfo.pCamera = pointer(camera)
		widthNodeInfo.attrName = b"Width"
		nRet = GENICAM_createIntNode(byref(widthNodeInfo), byref(widthNode))
		if ( nRet != 0 ):
			print("create Width Node fail!") 
			return -1
		
		nRet = widthNode.contents.setValue(widthNode, c_longlong(nWidth))
		if ( nRet != 0 ):
			print("widthNode setValue [%d] fail!" % (nWidth))
			# 释放相关资源
			widthNode.contents.release(widthNode)
			return -1  
		
		# 释放相关资源
		widthNode.contents.release(widthNode)
		
		# 设置高度
		heightNode = pointer(GENICAM_IntNode())
		heightNodeInfo = GENICAM_IntNodeInfo() 
		heightNodeInfo.pCamera = pointer(camera)
		heightNodeInfo.attrName = b"Height"
		nRet = GENICAM_createIntNode(byref(heightNodeInfo), byref(heightNode))
		if ( nRet != 0 ):
			print("create Height Node fail!")
			return -1
		
		nRet = heightNode.contents.setValue(heightNode, c_longlong(nHeight))
		if ( nRet != 0 ):
			print("heightNode setValue [%d] fail!" % (nHeight))
			# 释放相关资源
			heightNode.contents.release(heightNode)
			return -1	
		
		# 释放相关资源
		heightNode.contents.release(heightNode)	
		
		# 设置OffsetX
		OffsetXNode = pointer(GENICAM_IntNode())
		OffsetXNodeInfo = GENICAM_IntNodeInfo() 
		OffsetXNodeInfo.pCamera = pointer(camera)
		OffsetXNodeInfo.attrName = b"OffsetX"
		nRet = GENICAM_createIntNode(byref(OffsetXNodeInfo), byref(OffsetXNode))
		if ( nRet != 0 ):
			print("create OffsetX Node fail!")
			return -1
		
		nRet = OffsetXNode.contents.setValue(OffsetXNode, c_longlong(OffsetX))
		if ( nRet != 0 ):
			print("OffsetX setValue [%d] fail!" % (OffsetX))
			# 释放相关资源
			OffsetXNode.contents.release(OffsetXNode)
			return -1	
		
		# 释放相关资源
		OffsetXNode.contents.release(OffsetXNode)  
		
		# 设置OffsetY
		OffsetYNode = pointer(GENICAM_IntNode())
		OffsetYNodeInfo = GENICAM_IntNodeInfo() 
		OffsetYNodeInfo.pCamera = pointer(camera)
		OffsetYNodeInfo.attrName = b"OffsetY"
		nRet = GENICAM_createIntNode(byref(OffsetYNodeInfo), byref(OffsetYNode))
		if ( nRet != 0 ):
			print("create OffsetY Node fail!")
			return -1
		
		nRet = OffsetYNode.contents.setValue(OffsetYNode, c_longlong(OffsetY))
		if ( nRet != 0 ):
			print("OffsetY setValue [%d] fail!" % (OffsetY))
			# 释放相关资源
			OffsetYNode.contents.release(OffsetYNode)
			return -1	
		
		# 释放相关资源
		OffsetYNode.contents.release(OffsetYNode)   
		return 0

	def start_grab_img(self):
		cameraCnt, cameraList = self.enumCameras()
		if cameraCnt is None:
			return -1
		
		# 显示相机信息
		for index in range(0, cameraCnt):
			camera = cameraList[index]
			print("\nCamera Id = " + str(index))
			print("Key		   = " + str(camera.getKey(camera)))
			print("vendor name   = " + str(camera.getVendorName(camera)))
			print("Model  name   = " + str(camera.getModelName(camera)))
			print("Serial number = " + str(camera.getSerialNumber(camera)))
			
		self.camera = cameraList[0]

		# 打开相机
		nRet = self.openCamera(self.camera)
		if ( nRet != 0 ):
			print("openCamera fail.")
			return -1

		# 设置曝光
		nRet = self.setExposureTime(self.camera, 8000)
		if ( nRet != 0 ):
			print("set ExposureTime fail")
			return -1

		# 创建流对象
		self.streamSourceInfo = GENICAM_StreamSourceInfo()
		self.streamSourceInfo.channelId = 0
		self.streamSourceInfo.pCamera = pointer(self.camera)
		
		self.streamSource = pointer(GENICAM_StreamSource())
		nRet = GENICAM_createStreamSource(pointer(self.streamSourceInfo), byref(self.streamSource))
		if ( nRet != 0 ):
			print("create StreamSource fail!")
			return -1
		trigModeEnumNode = pointer(GENICAM_EnumNode())
		trigModeEnumNodeInfo = GENICAM_EnumNodeInfo() 
		trigModeEnumNodeInfo.pCamera = pointer(self.camera)
		trigModeEnumNodeInfo.attrName = b"TriggerMode"
		nRet = GENICAM_createEnumNode(byref(trigModeEnumNodeInfo), byref(trigModeEnumNode))
		if ( nRet != 0 ):
			print("create TriggerMode Node fail!")
			# 释放相关资源
			self.streamSource.contents.release(self.streamSource) 
			return -1
		
		nRet = trigModeEnumNode.contents.setValueBySymbol(trigModeEnumNode, b"Off")
		if ( nRet != 0 ):
			print("set TriggerMode value [Off] fail!")
			# 释放相关资源
			trigModeEnumNode.contents.release(trigModeEnumNode)
			self.streamSource.contents.release(self.streamSource) 
			return -1
		# 需要释放Node资源	
		trigModeEnumNode.contents.release(trigModeEnumNode)

		# 注册拉流回调函数
		userInfo = b"test"
		nRet = self.streamSource.contents.attachGrabbingEx(self.streamSource, self.frameCallbackFuncEx, userInfo)	
		if ( nRet != 0 ):
			print("attachGrabbingEx fail!")
			# 释放相关资源
			self.streamSource.contents.release(self.streamSource)  
			return -1
			
		# 开始拉流
		nRet = self.streamSource.contents.startGrabbing(self.streamSource, c_ulonglong(0), \
												c_int(GENICAM_EGrabStrategy.grabStrartegySequential))
		if( nRet != 0):
			print("startGrabbing fail!")
			# 释放相关资源
			self.streamSource.contents.release(self.streamSource)   
			return -1
		
		return 0
	def get_gray_image(self):

		grayByteArray = bytearray(self.usr_image)
		Image = numpy.array(grayByteArray).reshape(1200, 1920)
		Image = cv.cvtColor(Image, cv.COLOR_BAYER_BG2GRAY)
		return Image

	def get_BGR_image(self):

		colorByteArray = bytearray(self.usr_image)
		Image = numpy.array(colorByteArray).reshape(1200, 1920)
		Image = cv.cvtColor(Image, cv.COLOR_BAYER_BG2BGR)
		return Image

	def stop_grab_img(self):
		self.g_isStop = True
		# 反注册回调函数
		userInfo = b"test"
		nRet = self.streamSource.contents.detachGrabbingEx(self.streamSource, self.frameCallbackFuncEx, userInfo) 
		if ( nRet != 0 ):
			print("detachGrabbingEx fail!")
			# 释放相关资源
			self.streamSource.contents.release(self.streamSource)  
			return -1

		# 停止拉流
		nRet = self.streamSource.contents.stopGrabbing(self.streamSource)
		if ( nRet != 0 ):
			print("stopGrabbing fail!")
			# 释放相关资源
			self.streamSource.contents.release(self.streamSource)  
			return -1
		nRet = self.closeCamera(self.camera)
		if ( nRet != 0 ):
			print("closeCamera fail")
			# 释放相关资源
			self.streamSource.contents.release(self.streamSource)   
			return -1
		
		# 释放相关资源
		self.streamSource.contents.release(self.streamSource)	
		
		return 0


	def demo(self):	
		# 发现相机
		cameraCnt, cameraList = self.enumCameras()
		if cameraCnt is None:
			return -1
		
		# 显示相机信息
		for index in range(0, cameraCnt):
			camera = cameraList[index]
			print("\nCamera Id = " + str(index))
			print("Key		   = " + str(camera.getKey(camera)))
			print("vendor name   = " + str(camera.getVendorName(camera)))
			print("Model  name   = " + str(camera.getModelName(camera)))
			print("Serial number = " + str(camera.getSerialNumber(camera)))
			
		camera = cameraList[0]

		# 打开相机
		nRet = self.openCamera(camera)
		if ( nRet != 0 ):
			print("openCamera fail.")
			return -1

		# 设置曝光
		nRet = self.setExposureTime(camera, 8000)
		if ( nRet != 0 ):
			print("set ExposureTime fail")
			# 释放相关资源
			streamSource.contents.release(streamSource)
			return -1
		
		# 创建流对象
		streamSourceInfo = GENICAM_StreamSourceInfo()
		streamSourceInfo.channelId = 0
		streamSourceInfo.pCamera = pointer(camera)
		
		streamSource = pointer(GENICAM_StreamSource())
		nRet = GENICAM_createStreamSource(pointer(streamSourceInfo), byref(streamSource))
		if ( nRet != 0 ):
			print("create StreamSource fail!")
			return -1
		
		# 通用属性设置:设置触发模式为off --根据属性类型，直接构造属性节点。如触发模式是 enumNode，构造enumNode节点
		# 自由拉流：TriggerMode 需为 off
		trigModeEnumNode = pointer(GENICAM_EnumNode())
		trigModeEnumNodeInfo = GENICAM_EnumNodeInfo() 
		trigModeEnumNodeInfo.pCamera = pointer(camera)
		trigModeEnumNodeInfo.attrName = b"TriggerMode"
		nRet = GENICAM_createEnumNode(byref(trigModeEnumNodeInfo), byref(trigModeEnumNode))
		if ( nRet != 0 ):
			print("create TriggerMode Node fail!")
			# 释放相关资源
			streamSource.contents.release(streamSource) 
			return -1
		
		nRet = trigModeEnumNode.contents.setValueBySymbol(trigModeEnumNode, b"Off")
		if ( nRet != 0 ):
			print("set TriggerMode value [Off] fail!")
			# 释放相关资源
			trigModeEnumNode.contents.release(trigModeEnumNode)
			streamSource.contents.release(streamSource) 
			return -1
		
		# 需要释放Node资源	
		trigModeEnumNode.contents.release(trigModeEnumNode) 

		# 开始拉流
		nRet = streamSource.contents.startGrabbing(streamSource, c_ulonglong(0), \
												c_int(GENICAM_EGrabStrategy.grabStrartegySequential))
		if( nRet != 0):
			print("startGrabbing fail!")
			# 释放相关资源
			streamSource.contents.release(streamSource)
			return -1

		isGrab = True

		frame = pointer(GENICAM_Frame())
		imageParams = IMGCNV_SOpenParam()
		t= time.time()
		while isGrab :
			
			# 主动取图
			# frame = pointer(GENICAM_Frame())
			nRet = streamSource.contents.getFrame(streamSource, byref(frame), c_uint(1000))
			if ( nRet != 0 ):
				print("getFrame fail! Timeout:[1000]ms")
				# 释放相关资源
				streamSource.contents.release(streamSource)   
				return -1 
			else:
				print("getFrame success BlockId = [" + str(frame.contents.getBlockId(frame)) + "], get frame time: " + str(time.time() - t))
				t= time.time()
			nRet = frame.contents.valid(frame)
			if ( nRet != 0 ):
				print("frame is invalid!")
				# 释放驱动图像缓存资源
				frame.contents.release(frame)
				# 释放相关资源
				streamSource.contents.release(streamSource)
				return -1 

			# 给转码所需的参数赋值
			# imageParams = IMGCNV_SOpenParam()
			imageParams.dataSize	= frame.contents.getImageSize(frame)
			imageParams.height	  = frame.contents.getImageHeight(frame)
			imageParams.width	   = frame.contents.getImageWidth(frame)
			imageParams.paddingX	= frame.contents.getImagePaddingX(frame)
			imageParams.paddingY	= frame.contents.getImagePaddingY(frame)
			imageParams.pixelForamt = frame.contents.getImagePixelFormat(frame)

			# 将裸数据图像拷出
			imageBuff = frame.contents.getImage(frame)
			userBuff = c_buffer(b'\0', imageParams.dataSize)
			memmove(userBuff, c_char_p(imageBuff), imageParams.dataSize)

			# 释放驱动图像缓存
			frame.contents.release(frame)

			# 如果图像格式是 Mono8 直接使用
			if imageParams.pixelForamt == EPixelType.gvspPixelMono8:
				grayByteArray = bytearray(userBuff)
				cvImage = numpy.array(grayByteArray).reshape(imageParams.height, imageParams.width)
			else:
				grayByteArray = bytearray(userBuff)
				cvImage = numpy.array(grayByteArray).reshape(imageParams.height, imageParams.width)
				cvImage = cv.cvtColor(cvImage, cv.COLOR_BAYER_BG2GRAY)
				show_Image = cv.resize(cvImage, (960, 600))
	
			cv.imshow('myWindow', show_Image)
			
			gc.collect()
			

			if (cv.waitKey(1) >= 0):
				isGrab = False
				break
		# --- end while ---

		cv.destroyAllWindows()

		# 停止拉流
		nRet = streamSource.contents.stopGrabbing(streamSource)
		if ( nRet != 0 ):
			print("stopGrabbing fail!")
			# 释放相关资源
			streamSource.contents.release(streamSource)  
			return -1

		# 关闭相机
		nRet = self.closeCamera(camera)
		if ( nRet != 0 ):
			print("closeCamera fail")
			# 释放相关资源
			streamSource.contents.release(streamSource)   
			return -1
		
		# 释放相关资源
		streamSource.contents.release(streamSource)	
		
		return 0
	
if __name__=="__main__": 

	cam = Camera()
	nRet = cam.demo()
	if nRet != 0:
		print("Some Error happend")
	print("--------- Demo end ---------")
	# 3s exit
	time.sleep(0.5)
