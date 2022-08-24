from distutils.ccompiler import show_compilers
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import time
import copy
import sys
from copy import deepcopy

class Localize_obj(object):
	def __init__(self):
		is_corrected = True
		if is_corrected == False:
			print('The camera and coordination is not calibrated, please calibrate it first')
			sys.exit()
		else:
			with np.load('./localize/calibration_data.npz') as X:
				self.mtx, self.dist, rvecs, tvecs, self.PresM = [X[i] for i in ('mtx','dist','rvecs','tvecs','PresM')]
		temp_array = np.array([0,0,0])
		mtx1 = np.insert(self.mtx,3,temp_array,axis=1)
		rotM = cv.Rodrigues(rvecs)[0]
		rot_trans_M = np.insert(np.insert(rotM,3,tvecs.T,axis=1),3,np.array([0,0,0,1]),axis=0)

		self.w2p_M = np.matrix(np.matmul(mtx1,rot_trans_M))
		self.p2w_M = self.w2p_M.I
		self.newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (1920,1200), 0, (1920,1200))
		
		IDTable = np.loadtxt('./localize/ID.txt')

		self.r_output=open("./img/data.txt",'w+') 

		self.num_of_pattern = 16
		self.arena_size = (1.4,0.8)
		self.image_size = (1920,1200)
		self.zoom_ratio = 2
		self.pattern_size = 0.04
		self.tolerence = 0.4
		self.r_tolerence = 0.004
		self.IDTable = IDTable

		# (inner_min_minor_r, inner_max_major_r, outer_minor_r, outer_major_r, pattern_r)
		# self.pat_ratio = (37,92,130,160,200)
		self.pat_ratio = (60,145,280,340,400)
		
		# 0.9 is the roughly rate of arena area in the image
		self.pat_img_r = int((self.image_size[0]/self.arena_size[0]*self.pattern_size*0.9)/2)

		outer_r0 = self.pat_img_r*self.pat_ratio[2]/self.pat_ratio[4]
		outer_r1 = self.pat_img_r*self.pat_ratio[3]/self.pat_ratio[4]
		self.outer_area = np.pi * outer_r0 * outer_r1
		self.outer_r_ratio = outer_r0/outer_r1


		self.search_area_num = 0

		self.pattern = []
		for [id, r1, r0] in IDTable:
			self.pattern.append(Pattern(int(id), r1, r0, IDTable))


		# for dedug
		self.cnt = 0
	
	def __del__(self):
		self.r_output.close()


	def get_possible_posi(self,img):
		dim = (int(img.shape[1] /self.zoom_ratio), int(img.shape[0] /self.zoom_ratio))
		img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
		# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		img_gray = img
		cnt = 0
		try_thresh = 255
		possib_pos = []
		# try threshold with 255/2 = 128, 255/4 = 64, 255/8 = 32
		for try_i in range(0,1):
			try_thresh = int(try_thresh/2)
			ret, thresh = cv.threshold(img_gray,try_thresh,255,cv.THRESH_BINARY)
			# kernel = np.ones((3,3),np.uint8)
			# thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
			# thresh = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 21,20)
			contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
			# cv.drawContours(img, contours, -1, 128, 1)
			# cv.imwrite('./img/thresh.jpg', thresh)
			# cv.imwrite('./img/thresh_raw.jpg', img)
			if len(contours) == 0:
				continue

			
			outer_area = self.outer_area/(self.zoom_ratio**2)
			outer_r_ratio = self.outer_r_ratio
			area_tolerence = outer_area*0.4
			ratio_tolerence = outer_r_ratio*0.2
			

			for i,hier in enumerate(hierarchy[0]):
				# maxdis = 1
				# iid=-2

				if hier[2] == -1 and hier[3] != -1:
					if contours[hierarchy[0][i][3]].shape[0]>5:
						# save_img = cv.drawContours(img, contours, i, (0,255,0), 2)
						# save_img = cv.drawContours(img, contours, hierarchy[0][i][3], (255,0,0), 2)
						# inner_ellipse = cv.fitEllipse(contours[i])
						outer_ellipse = cv.fitEllipse(contours[hierarchy[0][i][3]])
						outerArea = cv.contourArea(contours[hierarchy[0][i][3]])
						# elli_ratio1 = inner_ellipse[1][0]/inner_ellipse[1][1]
						elli_ratio2 = outer_ellipse[1][0]/outer_ellipse[1][1]
						# Area tolerence = 
						# print('1:',outerArea, elli_ratio2)
						if abs(outerArea - outer_area) < area_tolerence and abs(elli_ratio2 - outer_r_ratio) < ratio_tolerence:
							temp = ((np.array(outer_ellipse[0]))* self.zoom_ratio).astype(int)
							temp1 = []
							cnt_i = 0
							if cnt > 0:
								for cnt_ii in range(0,cnt):
									if np.linalg.norm(temp-possib_pos[cnt_i])<self.pat_img_r:
										break
									cnt_i += 1
							
							if cnt_i != cnt:
								continue

							# print('area:',outerArea)
							# cv.ellipse(img, inner_ellipse, (0,255,255), 2)
							# cv.ellipse(img, outer_ellipse, (255,255,0), 1)
							# print(outer_ellipse, elli_ratio2)
							# print(outerArea, elli_ratio2)


							cnt+=1
							if cnt < 16:
								possib_pos.append(temp)
							elif int(cnt) == 16:
								possib_pos.append(temp)
								return possib_pos
							elif int(cnt) > 16:
								return possib_pos
		
		# print(cnt)
		# cv.imwrite('./img/possible_posi.jpg', img)
		# cv.imwrite('./img/possible_posi_thresh.jpg', thresh)
		
		return possib_pos
	
	def segment(self, img, pos, ii):
		search_size = self.pat_img_r*(1+0)
		search_start_y, search_start_x = max(int(pos[1] - search_size),0), max(int(pos[0] - search_size),0)
		search_end_y, search_end_x = min(int(pos[1] + search_size),img.shape[0]), min(int(pos[0] + search_size),img.shape[1])
		
		seg_img = copy.deepcopy(img[search_start_y:search_end_y,search_start_x:search_end_x])
		
		

		# img_gray = cv.cvtColor(seg_img, cv.COLOR_BGR2GRAY)
		img_gray = seg_img
		# ret, thresh = cv.threshold(img_gray,100,255,cv.THRESH_BINARY)
		# thr_size = int(search_size/2)*2 - 1
		# thresh = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, thr_size,10)

		# calculate the threshold as the mean of the gray level
		thre = img_gray.mean()
		ret, thresh = cv.threshold(img_gray,thre,255,cv.THRESH_BINARY)
		# filename = './img/seg_img_thresh'+str(ii)+'.jpg'
		# cv.imwrite(filename, thresh)
		# kernel = np.ones((5,5),np.uint8)
		# thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
		contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		# cv.drawContours(seg_img, contours, -1, (255,0,0), 2)

		

		if len(contours) == 0:
			return None

		outer_area = self.outer_area
		outer_r_ratio = self.outer_r_ratio
		area_tolerence = outer_area*0.4
		ratio_tolerence = outer_r_ratio*0.2
		maxdis = 1
		iid=-2

		for i,hier in enumerate(hierarchy[0]):
			# maxdis = 1
			# iid=-2

			if hier[2] == -1 and hier[3] != -1:
				if contours[i].shape[0] > 5 and contours[hierarchy[0][i][3]].shape[0]>5:
					# save_img = cv.drawContours(img, contours, i, (0,255,0), 2)
					# save_img = cv.drawContours(img, contours, hierarchy[0][i][3], (255,0,0), 2)
					inner_ellipse = cv.fitEllipse(contours[i])
					outer_ellipse = cv.fitEllipse(contours[hierarchy[0][i][3]])
					# innerArea = cv.contourArea(contours[i])
					outerArea = cv.contourArea(contours[hierarchy[0][i][3]])
					elli_ratio2 = outer_ellipse[1][0]/outer_ellipse[1][1]
					# Area tolerence = 
					# print(outerArea, elli_ratio2)
					if abs(outerArea - outer_area) < area_tolerence and abs(elli_ratio2 - outer_r_ratio) < ratio_tolerence:
						# cnt+=1
						# print('area:',outerArea)
						# cv.ellipse(seg_img, inner_ellipse, (0,255,255), 1)
						# cv.ellipse(seg_img, outer_ellipse, (255,255,0), 1)
						# cv.circle(seg_img, (int(outer_ellipse[0][0]),int( outer_ellipse[0][1])), 5, (0,0,255), -1)
						offset = 2
						if abs(outer_ellipse[1][0]/outer_ellipse[1][1] -1)< 0.1:
							r0 = (inner_ellipse[1][0] - offset)/(outer_ellipse[1][0] + offset)
							r1 = (inner_ellipse[1][1] - offset)/(outer_ellipse[1][1] + offset)
						elif abs(inner_ellipse[2] - outer_ellipse[2]) <= 45 or abs(inner_ellipse[2] - outer_ellipse[2]) >= 135:
							r0 = (inner_ellipse[1][0] - offset)/(outer_ellipse[1][0] + offset)
							r1 = (inner_ellipse[1][1] - offset)/(outer_ellipse[1][1] + offset)
						elif abs(abs(inner_ellipse[2] - outer_ellipse[2]) - 90)< 45:
							r0 = (inner_ellipse[1][1] - offset)/(outer_ellipse[1][0] + offset)
							r1 = (inner_ellipse[1][0] - offset)/(outer_ellipse[1][1] + offset)

						
						for i,[id, ideal_r1, ideal_r0] in enumerate(self.IDTable):
							dis = (ideal_r0 - r0)**2 + (ideal_r1 - r1)**2
							if dis < maxdis:
								maxdis = dis
								iid = id
						# filename = './img/seg_img'+str(ii)+'.jpg'
						# text = str(int(iid)) + '\t'+ str(round(r1,3))+'\t'+str(round(r0,3)) + '\t' +str(round(search_start_x,3)) + '\t' +str(round(search_start_y,3))
						# if iid == 13 or iid == 14:
						# 	text = str(int(iid)) + '\t' + str(round(r1,3))+'\t'+str(round(r0,3))+ '\t' +str(round(outer_ellipse[1][0],3)) + '\t' +str(round(outer_ellipse[1][1],3))+'\t' +str(round(inner_ellipse[1][0],3)) + '\t' +str(round(inner_ellipse[1][1],3))
						# 	print(text,file=self.r_output)
						# # print(text)
						# cv.putText(seg_img, text, (0,10), cv.FONT_HERSHEY_PLAIN, 0.55, (255, 0, 255), 1)
						# # filename = './img/seg_img'+str(self.cnt)+'.jpg'
						# cv.imwrite(filename, seg_img)
						# # cv.imwrite(filename, thresh)
						# print('r1,r0:',int(iid),round(maxdis,5),round(r1,3),round(r0,3),inner_ellipse[1:], outer_ellipse[1:], abs(inner_ellipse[2] - outer_ellipse[2]))
						return np.array([iid, search_start_x + outer_ellipse[0][0], search_start_y + outer_ellipse[0][1]]).astype(int)
						





						
						
						
						# dis = (self.ideal_r0 - self.r0)**2 + (self.ideal_r1 - self.r1)**2
						# if dis < self.r_tolerence:


						# print(outer_ellipse, elli_ratio2)
						
						# temp = ((np.array(outer_ellipse[0]))* self.zoom_ratio).astype(int)
						# possib_pos.append(temp)
						
	def trans_corrdi_p2w(self, p2w_M, PresM, p_pos):
		id = p_pos[0]
		p_pos[:-1] = p_pos[1:]
		p_pos[-1] = 1
		# p_pos.append(1)
		# p_pos = np.matrix(p_pos)
		w_pos = np.matmul(p2w_M,p_pos)
		w_pos = (w_pos[0]/w_pos[0,3]).tolist()
		w_pos[0].pop(2)
		w_pos = np.matmul(PresM,w_pos[0])
		w_pos = w_pos/w_pos[2]
		w_pos[1:] = w_pos[:-1]
		w_pos[0] = int(id)
		return w_pos

	def search_pattern(self, img_ori):
		possib_pos = self.get_possible_posi(img_ori)
		self.id_cam_pos = []
		self.world_pos = []
		
		
		for i,pos in enumerate(possib_pos):
			id_cam_pos = self.segment(img_ori, pos, i)
			if id_cam_pos is not None:
				self.id_cam_pos.append(deepcopy(id_cam_pos))
				self.world_pos.append(self.trans_corrdi_p2w(self.p2w_M, self.PresM, id_cam_pos))
		
		
		
		
		return self.world_pos
					




		

		
		
		



class Display_obj(object):
	def __init__(self):
		self.screen_image_size = (1920,1080)
		self.display_size = (1.4,0.8)
		self.real_image = (1890,1080)


class Pattern(object):
	def __init__(self, id, ideal_r1, ideal_r0, IDTable):
		self.id = id
		self.thresh = 127
		self.search_img= []
		self.innersize = 0
		self.outersize = 0
		self.inner = []
		self.outer = []
		self.inner_ellipse = []
		self.outer_ellipse = []
		self.ideal_r0 = ideal_r0
		self.ideal_r1 = ideal_r1
		self.r_tolerence = 0.004
		self.r0 = 0
		self.r1 = 0
		self.IDTable = IDTable
		self.lost_frames = 61

		self.cam_pos = [[0,0]]
		self.previous_search = [False]
		self.angle = 0
		self.world_pos = [[0,0]]
	
	# def find_pattern(self, img):



# IDTable = np.loadtxt('ID.txt')
# local = Localize_obj(IDTable)




# cap = cv.VideoCapture(0)
# if not cap.isOpened():
# 	print("Cannot open camera")
# 	exit()
# else:
# 	print("Camera is opened")
# 	ret = cap.set(cv.CAP_PROP_FRAME_WIDTH,3840)
# 	print(ret)

# t1 = time.time()

# while cap.isOpened():
# 	local.cnt +=1
	
# 	# img = cv.imread('./img/pic 4k pattern.jpg')
# 	ret, img_ori = cap.read()
	
# 	# if read the frame，ret is True
# 	if not ret:
# 		print("Can't receive frame (stream end?). Exiting ...")
# 		break

# 	# img_ori = cv.imread('./img/WIN_20220421_13_59_31_Pro.jpg')
# 	t = time.time()
# 	possib_pos = local.test_get_possible_posi(img_ori)
	

# 	for i,pos in enumerate(possib_pos):
# 		id_pos = local.segment(img_ori, pos, i)
# 		print(id_pos)
# 		if id_pos is None:
# 			pass
# 		else:
# 			cv.circle(img_ori, (id_pos[1],id_pos[2]), 20, (0,255,255), -1)
# 	print('t:', time.time()-t)
# 	show_img = cv.resize(img_ori, (1280,720))
# 	cv.imshow('img', show_img)
# 	k = cv.waitKey(500)
# 	if t - t1 > 30:
# 		break

# # print(possib_pos)

# cap.release()






# sys.exit()

