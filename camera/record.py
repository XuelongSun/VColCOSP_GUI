import time

import HighFpsCamera

import cv2 as cv

cam = HighFpsCamera.Camera()
#define the code and create a video writer obj
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('test_on_negtive.mp4',fourcc,60.0,(1920,1200),False)

# Experiment lasting time(s) 
ex_time=20

nRet = cam.start_grab_img()
if nRet == 0:
	print('start grab image success')
else:
	print("Some Error happend")
	exit()
# wait for camera to be ready
time.sleep(1)
t_start = t_last=time.monotonic()

while True:
	t_last = time.monotonic()
	img_ori = cam.get_gray_image()

	show_Image = cv.resize(img_ori, (960, 600))
	out.write(img_ori)
	cv.imshow('myWindow', show_Image)
	# gc.collect()
	t_now = time.monotonic()
	if (t_now - t_start) > ex_time:
		break
	k = cv.waitKey(1)
	if k == 27:
		break
	t_frame = time.monotonic() -t_last
	print(t_frame)
cam.stop_grab_img()
out.release()


print("--------- EX end ---------t = {}".format((time.monotonic()-t_start)))

cv.destroyAllWindows()


time.sleep(0.5)