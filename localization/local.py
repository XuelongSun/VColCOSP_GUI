
from copy import deepcopy
import time
import sys
import os
from threading import Thread

import cv2 as cv
import numpy as np

from socket_test import DataServer

sys.path.append('./camera')

import camera.HighFpsCamera as HighFpsCamera
import localize.localize as localize

cam = HighFpsCamera.Camera()
localSys = localize.Localize_obj()

phero_win_name = "phero_win"
# cv.namedWindow(phero_win_name,cv.WINDOW_AUTOSIZE)
cv.namedWindow(phero_win_name,cv.WINDOW_NORMAL)
cv.setWindowProperty(phero_win_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

phero_map = np.zeros((1080,1920,3),np.uint8)

nRet = cam.start_grab_img()
if nRet == 0:
	print('start grab image success')
else:
	print("Some Error happend")
	exit()
# wait for camera to be ready
time.sleep(1)
t = time.time()

global data_str

data_str = '123'
data_server = DataServer(data_str)
data_server_process = Thread(target=data_server.run)
data_server_process.start()

while True:
	img_ori = cam.get_gray_image()
	# save_img = deepcopy(img_ori)
	
	t = time.time()
	# img_ori = cv.undistort(img_ori, localSys.mtx, localSys.dist, None, localSys.newcameramtx)
	world_pos = localSys.search_pattern(img_ori)
	world_pos = sorted(world_pos,key=(lambda x:x[0]))
	data_str = 'Detected {} of {} at 1603184275917588 \n'.format(len(world_pos), len(world_pos))
	for d in world_pos:
		data_str += 'Robot {} {:.3f} {:.3f} {:3.3f} {} \n'.format(str(int(d[0])).zfill(3),
																d[1],
																d[2],
																101.0,
																1603184275917588,
																)
	data_server.data = data_str
	# print(data_str)
	# print('t:', time.time()-t)
	if len(world_pos) > 0:
		phero_map = np.zeros((1080,1920,3),np.uint8)
		for i in range(len(world_pos)):
			phero_pos = (int(world_pos[i][1]/1.4*1890)+17,int(world_pos[i][2]/0.8*1080))
			cv.circle(phero_map, phero_pos, 20, (0,255,255), -1)
			cv.putText(phero_map, str(int(world_pos[i][0])), (phero_pos[0],phero_pos[1]-10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
		cv.imshow(phero_win_name, phero_map)
		cv.waitKey(1)

			# cv.circle(img_ori, (id_pos[1],id_pos[2]), 10, (0,255,255), -1)
			# cv.putText(img_ori, str(id_pos[0]), (id_pos[1],id_pos[2]), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

	show_Image = cv.resize(img_ori, (960, 600))
	cv.imshow('myWindow', show_Image)
	# gc.collect()
	k = cv.waitKey(1)
	if k == 27:
		break
	
	# cv.imwrite('img/test.jpg', save_img)
localSys.r_output.close()
cam.stop_grab_img()



print("--------- EX end ---------")
cv.destroyAllWindows()

time.sleep(0.5)