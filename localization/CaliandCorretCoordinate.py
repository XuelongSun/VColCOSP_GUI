# from curses import KEY_SAVE
# from multiprocessing.connection import wait
from time import sleep
import numpy as np
import cv2 as cv
import time
# import glob
import sys

sys.path.append('./camera')

import camera.HighFpsCamera as HighFpsCamera

cheese_width = 15
cheese_height = 7

# 终止条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 准备对象点， 如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cheese_width*cheese_height,3), np.float32)
objp[:,:2] = np.mgrid[0:cheese_width,0:cheese_height].T.reshape(-1,2)
# draw hess board 8cm/grid
objp[:,:2] = objp[:,:2]*0.08+0.16
# objp[:,2] = objp[:,2] + 1.9
# print(objp)
# 用于存储所有图像的对象点和图像点的数组。
axis = np.float32([[0,0,0], [1.4,0,0], [0,0.8,0], [0,0,-0.15]]).reshape(-1,3)
objpoints = [] # 真实世界中的3d点
imgpoints = [] # 图像中的2d点


def draw_chess_board(width,height,cheese_cell):
	# width = 15
	# height = 7
	qipan_cell = cheese_cell
	
	# width_pix = (width + 1) * qipan_cell + qipan_cell  # add extra  qipan_cell  for reserve blank
	# height_pix = (height + 1) * qipan_cell + qipan_cell
	
	#white = (255,255,255)
	#black =  (0,0,0)
	
	image = np.zeros((1080, 1920, 3), dtype=np.uint8)
	image.fill(255)
	
	
	color = (255,255,255)
	
	offsetx = 17+3*qipan_cell
	offsety = 3*qipan_cell
	# y0 = 0
	fill_color = 0
	for j in range(0,height + 1):
		y = j * qipan_cell
		for i in range(0,width+1):
			#rint(i)
			x0 = i *qipan_cell+offsetx
			y0 = y+offsety
			rect_start = (x0,y0)
	
			x1 = x0 + qipan_cell
			y1 = y0 + qipan_cell
			rect_end = (x1,y1)
			# print(x0,y0,x1,y1, fill_color)
			cv.rectangle(image, rect_start, rect_end,color, 1, 0)
			#print(fill_color)
			image[y0:y1,x0:x1] = fill_color
			if width % 2: 
				if i != width:
					fill_color = (0 if ( fill_color == 255) else 255)
			else:
				if i != width + 1:
					fill_color = (0 if ( fill_color == 255) else 255)

	# cv.imwrite('chess_board2.jpg',image)
	return image

def draw(img, corners, imgpts):
	corner = tuple(corners[0].ravel().astype(int))
	img = cv.line(img, tuple(imgpts[0].ravel().astype(int)), tuple(imgpts[1].ravel().astype(int)), (255,0,0), 5)
	img = cv.line(img, tuple(imgpts[0].ravel().astype(int)), tuple(imgpts[2].ravel().astype(int)), (0,255,0), 5)
	img = cv.line(img, tuple(imgpts[0].ravel().astype(int)), tuple(imgpts[3].ravel().astype(int)), (0,0,255), 5)
	return img

def draw_four_correct_pattern():
	arena_size = (1.4,0.8)
	screen_image_size = (1920,1080)
	image_size = (1920,1080)
	offset = 17
	pattern_size = 0.04
	correct_img_pattern_r = int((image_size[1]/arena_size[1]*pattern_size)/2)
	correct_img_pattern = np.zeros((correct_img_pattern_r*2,correct_img_pattern_r*2),np.uint8)
	cv.circle(correct_img_pattern,(correct_img_pattern_r,correct_img_pattern_r),correct_img_pattern_r,(255,255,255),-1)
	cv.circle(correct_img_pattern,(correct_img_pattern_r,correct_img_pattern_r),int(correct_img_pattern_r*0.75),(0,0,0),-1)
	cv.circle(correct_img_pattern,(correct_img_pattern_r,correct_img_pattern_r),int(correct_img_pattern_r*0.25),(255,255,255),-1)
	# cv.blur(correct_img_pattern,(5,5))
	correct_img = np.zeros((1080,1920),dtype=np.uint8)
	correct_img_pattern_pos = [[offset,0],[screen_image_size[0] - 2*correct_img_pattern_r-offset,0],[offset,screen_image_size[1] - 2*correct_img_pattern_r],[screen_image_size[0] - 2*correct_img_pattern_r-offset,screen_image_size[1] - 2*correct_img_pattern_r]]
	
	for pos in correct_img_pattern_pos:
		correct_img[pos[1]:pos[1]+2*correct_img_pattern_r,pos[0]:pos[0]+2*correct_img_pattern_r] = correct_img_pattern
	# cv.circle(correct_img,(pattern_pixel_r,pattern_pixel_r),pattern_pixel_r,(255,255,255),-1)
	# cv.circle(correct_img,(pattern_pixel_r,pattern_pixel_r),int(pattern_pixel_r*0.75),(0,0,0),-1)
	# cv.circle(correct_img,(pattern_pixel_r,pattern_pixel_r),int(pattern_pixel_r*0.25),(255,255,255),-1)
	# cv.circle(correct_img,(screen_image_size[0] - pattern_pixel_r,pattern_pixel_r),pattern_pixel_r,(255,255,255),-1)
	# cv.circle(correct_img,(screen_image_size[0] - pattern_pixel_r,pattern_pixel_r),int(pattern_pixel_r*0.75),(0,0,0),-1)
	# cv.circle(correct_img,(screen_image_size[0] - pattern_pixel_r,pattern_pixel_r),int(pattern_pixel_r*0.25),(255,255,255),-1)
	# cv.circle(correct_img,(screen_image_size[0] - pattern_pixel_r,screen_image_size[1] - pattern_pixel_r),pattern_pixel_r,(255,255,255),-1)
	# cv.circle(correct_img,(screen_image_size[0] - pattern_pixel_r,screen_image_size[1] - pattern_pixel_r),int(pattern_pixel_r*0.75),(0,0,0),-1)
	# cv.circle(correct_img,(screen_image_size[0] - pattern_pixel_r,screen_image_size[1] - pattern_pixel_r),int(pattern_pixel_r*0.25),(255,255,255),-1)
	# cv.circle(correct_img,(pattern_pixel_r,screen_image_size[1] - pattern_pixel_r),pattern_pixel_r,(255,255,255),-1)
	# cv.circle(correct_img,(pattern_pixel_r,screen_image_size[1] - pattern_pixel_r),int(pattern_pixel_r*0.75),(0,0,0),-1)
	# cv.circle(correct_img,(pattern_pixel_r,screen_image_size[1] - pattern_pixel_r),int(pattern_pixel_r*0.25),(255,255,255),-1)
	# cv.circle(correct_img,(600,350),10,(255,255,255),-1)

	# cv.namedWindow('correct_img', cv.WINDOW_NORMAL)
	# cv.setWindowProperty('correct_img', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
	# cv.moveWindow('correct_img', 15, 0)
	return correct_img

# draw hess board 8cm/grid
# image = draw_chess_board(cheese_width,cheese_height,int(1080/20))
image = cv.imread('chess_board1.jpg')

#create a window for display
win_name = "chess board"

# show the image in the window
# cv.namedWindow(win_name,cv.WINDOW_AUTOSIZE)

# show the image as full screen
cv.namedWindow(win_name,cv.WINDOW_NORMAL)
cv.setWindowProperty(win_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

cv.imshow(win_name, image)
# cv.moveWindow(win_name, 0, 0)
cv.waitKey(10000)

# read image from camera######################
cam = HighFpsCamera.Camera()
nRet = cam.start_grab_img()
if nRet == 0:
	print('start grab image success')
else:
	print("Some Error happend")
	exit()
	# ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT,1080)
	# print(ret)
# wait for the camera to warm up
time.sleep(1)

	
img = cam.get_BGR_image()


image_show = cv.resize(img, (1280,720))
cv.imshow('camera', image_show)
k = cv.waitKey(int(1000/33)) & 0xff
# if k == ord('c'):
# 	# break
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# find chess board corners

ret, corners = cv.findChessboardCorners(gray, (cheese_width,cheese_height), None)
if ret == True:
	# print("found")
	objpoints.append(objp)
	corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
	imgpoints.append(corners)
	cv.drawChessboardCorners(img, (cheese_width,cheese_height), corners2, ret)
	image_show = cv.resize(img, (1280,720))
	cv.imshow('camera', image_show)
	cv.waitKey(1)
	
	
else:
	pass
	print("Can't find chess board corners")
	# if k == ord('q'):
	# 	break
	# if k == ord('s'):
	# 	image = cv.imread('chess_board2.jpg')
	# 	cv.imshow(win_name, image)
	# 	cv.waitKey(1)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print(mtx,dist,rvecs,tvecs)
# ret, corners = cv.findChessboardCorners(gray, (cheese_width,cheese_height), None)
# if ret == True:

# objp1 = np.zeros((cheese_width*cheese_height,3), np.float32)
# objp1[:,:2] = np.mgrid[0:cheese_width,0:cheese_height].T.reshape(-1,2)
# # draw hess board 8cm/grid
# objp1[:,:2] = objp[:,:2]*0.08+0.16
# objp1[:,2] = objp[:,2] + 1.9
# objp2 = np.zeros((cheese_width*cheese_height,3), np.float32)
# objp2[:,:2] = np.mgrid[0:cheese_width,0:cheese_height].T.reshape(-1,2)
# # draw hess board 4cm/grid
# objp2[:,:2] = objp[:,:2]*0.08+0.16
# objp2[:,2] = objp[:,2] + 1.9



# print(objp)
ret,rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)
# 将3D点投影到图像平面
imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
print(imgpts)
# img = draw(img,corners2[0],imgpts)
img = draw(img,corners[0],imgpts)
image_show = cv.resize(img, (1280,720))
cv.imshow('img',image_show)
k = cv.waitKey(500) & 0xFF
# if k == ord('s'):
# 	cv.imwrite('1.png', img)



########second phase: draw four pattern for decrease the localition error#######
# pattern_image = draw_four_correct_pattern()
# cv.imshow(win_name,pattern_image)
# cv.waitKey(0)
# ret, img = cap.read()

zoom_x = 1
zoom_y = 1
shift_x = 0
shift_y = 0


def trans_corrdi_p2w(p2w_M, p_pos):
	p_pos[0].append(1)
	p_pos = p_pos[0]
	print(p_pos)
	# p_pos = np.matrix(p_pos)
	w_pos = np.matmul(p2w_M,p_pos)
	w_pos = (w_pos[0]/w_pos[0,3]).tolist()
	w_pos[0][0] = w_pos[0][0] * zoom_x
	w_pos[0][1] = w_pos[0][1] * zoom_y
	w_pos[0][0] = w_pos[0][0] + shift_x
	w_pos[0][1] = w_pos[0][1] + shift_y
	return w_pos[0][:2]

temp_array = np.array([0,0,0])
mtx1 = np.insert(mtx,3,temp_array,axis=1)
rotM = cv.Rodrigues(rvecs)[0]
rot_trans_M = np.insert(np.insert(rotM,3,tvecs.T,axis=1),3,np.array([0,0,0,1]),axis=0)

w2p_M = np.matrix(np.matmul(mtx1,rot_trans_M))
p2w_M = w2p_M.I

pts1 = []
pts2 = []
pts1.append(trans_corrdi_p2w(p2w_M, imgpoints[0][0].tolist()))
pts1.append(trans_corrdi_p2w(p2w_M, imgpoints[0][cheese_width-1].tolist()))
pts1.append(trans_corrdi_p2w(p2w_M, imgpoints[0][-cheese_width].tolist()))
pts1.append(trans_corrdi_p2w(p2w_M, imgpoints[0][-1].tolist()))

pts2.append(objp[0][:2].tolist())
pts2.append(objp[cheese_width-1][:2].tolist())
pts2.append(objp[-cheese_width][:2].tolist())
pts2.append(objp[-1][:2].tolist())
pts1 = np.array(pts1).astype(np.float32)
pts2 = np.array(pts2).astype(np.float32)
# print('W_pos:', pts1)
print('P_pos:', pts2)

PresM = cv.getPerspectiveTransform(pts1,pts2)

# print(PresM)

# point1 = []
point2 = []
point1 = np.insert(pts1,2,[1,1,1,1],axis=1)
for point in point1:
	point2.append(np.matmul(PresM,point).tolist())
	# point2 = np.append(point2,np.matmul(PresM,point))
for point in point2:
	point[0] = point[0]/point[2]
	point[1] = point[1]/point[2]
	point[2] = 1
point2 = np.array(point2).astype(np.float32)
# print('p:',point1)
print('p2:',point2)
print(PresM)
# print('objp:', objp[0],objp[cheese_width-1], objp[-cheese_width],objp[-1])

# temp_img = np.zeros((1080,1920,3),np.uint8)
for point in point2:
	print(point)
	phero_pos = (int(point[0]/1.4*1890+17),int(point[1]/0.8*1080))
	cv.circle(image, phero_pos ,5,(0,200,200),-1)
cv.imshow(win_name, image)
cv.waitKey(0)

np.savez_compressed('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, PresM= PresM)

cam.stop_grab_img()
cv.destroyAllWindows()




