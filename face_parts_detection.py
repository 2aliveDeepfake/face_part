
# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from matplotlib import pyplot as plt
from collections import OrderedDict
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

# 이미지 경로
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
# 얼굴 영역 전체 분할
ap.add_argument("-a", "--face_part_all")
# 원하는 얼굴 영역 입력(여러 영역일 경우 ,로 구분)
ap.add_argument("-part", "--face_part")
# 원하는 포인트대로 분할(한 영역은 , 다른 영역은 /로 구분)
ap.add_argument("-pts", "--face_part_point")

args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

test_dataset = os.listdir(args["image"])
num_images = len(test_dataset)

for i, img_name in enumerate(test_dataset):

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(args["image"]+img_name)
	#image = imutils.resize(image, width=800)

	h, w, c = image.shape
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# show the original input image and detect faces in the grayscale image
	# cv2.imshow("Input", image)
	rects = detector(gray, 2)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)

		# 선택된 영역이 이미지 넓이를 벗어난 경우 예외처리
		left = rect.left()
		top = rect.top()
		if left<0 :
			left = 0
		elif top<0 :
			top = 0


		shape = face_utils.shape_to_np(shape)

		#---------------------------------------------------------------------------------------
		# 추가 포인트 찍기
		# 눈썹 6개 왼쪽 - 69, 70, 71 / 오른쪽 - 72,73,74
		# 광대 2개 - 75, 76
		# 광개 옆 2개 - 77, 78
		# 입 주변 5개 - 79, 80, 81, 82, 83
		# 미간 1개 - 84
		# 광대위 2개 - 85,86
		# 눈썹옆 2개 - 87,88
		# 턱선 밑 9개 - 89,90,91,92,93,94,95,96,97
		# 귀 4개 - 98,99,100,101
		# 입꼬리 옆 2개 - 102,103
		# 눈썹 위 9개 - 104,105,106,107,108,109,110,111,112

		# shape 좌표 접근 shape[랜드마크번호][x:0,y:1]

		# 눈썹 6개 왼쪽 - 69, 70, 71 / 오른쪽 - 72,73,74
		# 69번 : 17, 19번의 선에서 18이 만나는 곳의 길이만큼 17번에서 올리기
		# y = (y2 - y1) / (x2 - x1) * (x - x1) + y1


		part1_m = (shape[17][1]-shape[19][1])/(shape[17][0]-shape[19][0])
		part1_n = shape[17][1]-(part1_m*shape[17][0])
		part1_y = part1_m*shape[18][0] + part1_n
		part1_d = abs(shape[18][1]-part1_y)
		#print('part1_m',part1_m,'part1_n',part1_n,'part1_y',part1_y,'part1_d',part1_d)

		x_69 = shape[17][0]
		y_69 = int(shape[17][1]-part1_d)
		x_70 = shape[19][0]
		y_70 = int(shape[19][1]-part1_d)
		x_71 = shape[21][0]
		y_71 = int(shape[21][1]-part1_d)

		part5_m = (shape[26][1]-shape[24][1])/(shape[26][0]-shape[24][0])
		part5_n = shape[26][1]-(part5_m*shape[26][0])
		part5_y = part5_m*shape[25][0] + part5_n
		part5_d = abs(shape[25][1]-part5_y)
		#print('part5_m',part5_m,'part5_n',part5_n,'part5_y',part5_y,'part5_d',part5_d)

		x_72 = shape[22][0]
		y_72 = int(shape[22][1]-part5_d)
		x_73 = shape[24][0]
		y_73 = int(shape[24][1]-part5_d)
		x_74 = shape[26][0]
		y_74 = int(shape[26][1]-part5_d)

		#미간 1개 - 84
		# x좌표 : 21번과 22번 중간
		x_84 = int(round((shape[21][0]+shape[22][0])/2))
		y_84 = shape[21][1]

		#광대 2개 - 75, 76
		# 75 : 3과 29의 중간 , 76 : 13과 29의 중간
		x_75 = int(round((shape[3][0] + shape[29][0]) / 2))
		y_75 = int(round((shape[3][1] + shape[29][1]) / 2))
		x_76 = int(round((shape[13][0] + shape[29][0]) / 2))
		y_76 = int(round((shape[13][1] + shape[29][1]) / 2))

		# 광개 옆 2개 - 77, 78
		# 77 : 2와 41 중간, 78 : 46과 14 중간
		x_77 = int(round((shape[2][0] + shape[41][0]) / 2))
		y_77 = int(round((shape[2][1] + shape[41][1]) / 2))
		x_78 = int(round((shape[14][0] + shape[46][0]) / 2))
		y_78 = int(round((shape[14][1] + shape[46][1]) / 2))

		# 입 주변 5개 - 79, 80, 81, 82, 83
		# 79 : 5, 59 / 80 :  6, 58 / 81 : 8, 57 / 82 : 10, 58 / 83 : 11, 55
		x_79 = int(round((shape[5][0] + shape[59][0]) / 2))
		y_79 = int(round((shape[5][1] + shape[59][1]) / 2))
		x_80 = int(round((shape[6][0] + shape[58][0]) / 2))
		y_80 = int(round((shape[6][1] + shape[58][1]) / 2))
		x_81 = int(round((shape[8][0] + shape[57][0]) / 2))
		y_81 = int(round((shape[8][1] + shape[57][1]) / 2))
		x_82 = int(round((shape[10][0] + shape[58][0]) / 2))
		y_82 = int(round((shape[10][1] + shape[58][1]) / 2))
		x_83 = int(round((shape[11][0] + shape[55][0]) / 2))
		y_83 = int(round((shape[11][1] + shape[55][1]) / 2))

		# 광대위 2개 - 85,86
		# 85 : 39, 75 / 86 : 42, 76
		x_85 = int(round((shape[39][0] + x_75) / 2))
		y_85 = int(round((shape[39][1] + y_75) / 2))
		x_86 = int(round((shape[42][0] + x_76) / 2))
		y_86 = int(round((shape[42][1] + y_76) / 2))

		# 눈썹옆 2개 - 87,88
		# 87 : x_0,y_69 / 88 : x_16, y_74
		x_87 = shape[0][0]
		y_87 = y_69
		x_88 = shape[16][0]
		y_88 = y_74

		# 턱선 밑 9개 - 89,90,91,92,93,94,95,96,97
		# x2,y2가 x1,y1와 x3,y3의 중간일 때 x1 = 2*x2-x3 y1 = 2*y2-y3
		x_89 = int(2 * shape[3][0] - (x_75+shape[3][0])/2)
		y_89 = int(2 * shape[3][1] - (y_75+shape[3][1])/2)
		x_90 = int(2 * shape[4][0] - (shape[4][0]+shape[48][0])/2)
		y_90 = int(2 * shape[4][1] - (shape[4][1]+shape[48][1])/2)
		x_91 = 2 * shape[5][0] - x_79
		y_91 = 2 * shape[5][1] - y_79
		x_92 = 2 * shape[6][0] - x_80
		y_92 = 2 * shape[6][1] - y_80
		x_93 = 2 * shape[8][0] - x_81
		y_93 = 2 * shape[8][1] - y_81
		x_94 = 2 * shape[10][0] - x_82
		y_94 = 2 * shape[10][1] - y_82
		x_95 = 2 * shape[11][0] - x_83
		y_95 = 2 * shape[11][1] - y_83
		x_96 = int(2 * shape[12][0] - (shape[12][0]+shape[54][0])/2)
		y_96 = int(2 * shape[12][1] - (shape[12][1]+shape[54][1])/2)
		x_97 = int(2 * shape[13][0] - (x_76+shape[13][0])/2)
		y_97 = int(2 * shape[13][1] - (y_76+shape[13][1])/2)

		# 귀 4개 - 98,99,100,101
		x_98 = 0
		y_98 = y_70
		x_99 = 0
		y_99 = shape[2][1]
		x_100 = w
		y_100 = y_73
		x_101 = w
		y_101 = shape[14][1]

		# 입꼬리 옆 2개 - 102,103
		#102 - 48,4 / 103 - 54, 12
		x_102 = int(round((shape[48][0]+shape[4][0])/2))
		y_102 = int(round((shape[48][1]+shape[4][1])/2))
		x_103 = int(round((shape[54][0] + shape[12][0]) / 2))
		y_103 = int(round((shape[54][1] + shape[12][1]) / 2))

		# 눈썹 위 9개 - 104,105,106,107,108,109,110,111,112
		#  19, 37 사이 y 간격만큼 더해서 위로 올리기
		# 19, 70 사이 y 간격만큼 더해서 위로 올리기
		brow_d = shape[37][1] - shape[19][1]
		x_104 = x_87
		y_104 = y_87 - brow_d
		x_105 = x_69
		y_105 = y_69 - brow_d
		x_106 = shape[19][0]
		y_106 = shape[19][1] - brow_d
		x_107 = shape[21][0]
		y_107 = shape[21][1] - brow_d
		x_108 = x_84
		y_108 = y_84 - brow_d
		x_109 = shape[22][0]
		y_109 = shape[22][1] - brow_d
		x_110 = shape[24][0]
		y_110 = shape[24][1] - brow_d
		x_111 = shape[26][0]
		y_111 = shape[26][1] - brow_d
		x_112 = x_88
		y_112 = y_88 - brow_d


		# ---------------------------------------------------------------------------------------
		# 추가 점 배열에 담기
		additional_point = OrderedDict([(69, (x_69,y_69)),(70, (x_70,y_70)), (71, (x_71,y_71)), (72, (x_72, y_72)), (73, (x_73, y_73)), (74, (x_74, y_74)),
										 (75, (x_75, y_75)) , (76, (x_76, y_76)), (77, (x_77, y_77)), (78, (x_78, y_78)), (79, (x_79, y_79)),
										(80, (x_80, y_80)), (81, (x_81, y_81)), (82, (x_82, y_82)), (83, (x_83, y_83)), (84, (x_84, y_84)),
										(85, (x_85, y_85)), (86, (x_86, y_86)), (87, (x_87, y_87)), (88, (x_88, y_88)) , (89, (x_89, y_89)),
										 (90, (x_90, y_90)) , (91, (x_91, y_91)) , (92, (x_92, y_92)) , (93, (x_93, y_93)) , (94, (x_94, y_94)),
										(95, (x_95, y_95)) , (96, (x_96, y_96)) , (97, (x_97, y_97)), (98, (x_98, y_98)) , (99, (x_99, y_99)),
										 (100, (x_100, y_100)), (101, (x_101, y_101)), (102, (x_102, y_102)), (103, (x_103, y_103)),
										(104, (x_104, y_104)), (105, (x_105, y_105)), (106, (x_106, y_106)), (107, (x_107, y_107)),
										(108, (x_108, y_108)), (109, (x_109, y_109)), (110, (x_110, y_110)), (111, (x_111, y_111)) , (112, (x_112, y_112))
										])

		# ---------------------------------------------------------------------------------------
		#영역별로 나눠서 배열에 담기
		facepart_index = OrderedDict([("part_1", (69,17,19,70)),("part_2", (70,19,21,71)),("part_3", (71,21,27,84)),
									  ("part_4", (84, 27, 22, 72)),("part_5", (22,24,73,72)),("part_6", (24,26,74,73)),
									  ("part_7", (0, 17, 36, 1)),("part_8", (17, 36, 37, 19)),("part_9", (19,38,39,21)),
									  ("part_10", (21, 27, 28, 39)), ("part_11", (27, 22, 42, 28)), ("part_12", (22, 24, 43, 42)),
									  ("part_13", (24, 44, 45, 26)), ("part_14", (45, 26, 16, 15)),("part_15", (2, 3, 75, 77)),
									  ("part_16", (77, 75, 39, 40, 41)), ("part_17", (28, 29, 75, 39)), ("part_18", (28, 29, 76, 42)),
									  ("part_19", (42, 47, 46, 78, 76)), ("part_20", (14, 13, 76, 78)), ("part_21", (3, 4, 48, 75)),
									  ("part_22", (48, 33, 29, 75)), ("part_23", (29,33,54,76)), ("part_24", (13,12,54,76)),
									  ("part_25", (48, 4, 5, 79)), ("part_26", (48, 33, 59, 79)),("part_27", (33,54,83,55)),
									  ("part_28", (54, 12, 11, 83)), ("part_29", (59, 58, 80, 79)),("part_30", (58,57,81,80)),
									  ("part_31", (57, 56, 82, 81)), ("part_32", (55, 56, 82, 83)),	  ("part_33", (5, 6, 80, 79)),
									  ("part_34", (6, 8, 81, 80)), ("part_35", (8, 10, 82, 81)), ("part_36", (10, 11, 83, 82)),
									  ("part_37", (33, 59, 58, 57, 56, 55)),("part_38", (1, 36, 41,2)),("part_39", (15, 14, 46, 45)),
									  ("part_40", (19,37,38)), ("part_41", (43, 44, 24)), ("part_42", (36, 37, 38, 39,40,41)),
									   ("part_43", (42, 43, 44, 45, 46, 47)), ("part_44", (29,75,33)), ("part_45", (29,33,76)),
									 ("part_46", (75, 48, 33)), ("part_47", (33,54,76)),("part_48", (33,48,49,50,51,52,53,54)),
									  ("part_49", (48,79,59)),("part_50", (54,55,83)), ("part_51", (17,36,41,77)),
									 ("part_52", (41,40,39,85,77)), ("part_53", (39,28,85)), ("part_54", (28,42,86)),
									  ("part_55", (42,47,46,78,86)), ("part_56", (26,78,46,45)), ("part_57", (0, 1, 2, 77, 17)),
									("part_58", (77, 75, 85)), ("part_59", (86, 76, 78)), ("part_60", (78, 14, 15, 16, 26)),
									  ("part_61", (0, 87, 69, 17)), ("part_62", (74, 26, 16, 88))
							 ])


		# ---------------------------------------------------------------------------------------
		#영역별로 모두 표시
		if args['face_part_all']:

			face_part_all_img = image.copy()

			for (_, name) in enumerate(facepart_index.keys()):
				pts = np.zeros((len(facepart_index[name]), 2), np.int32)
			#	print('facepart_index[name]', facepart_index[name], 'len(facepart_index[name]', len(facepart_index[name]))
				for i, j in enumerate(facepart_index[name]):
			#		print('facepart_index[name]',facepart_index[name], 'i',i,'j',j)
					if j <= 68 :
						pts[i] = [shape[j][0], shape[j][1]]
						#print ('pts', pts)
					else:
						pts[i] = [additional_point[j][0], additional_point[j][1]]
						#print ('pts', pts)
			#	print('facepart_index[name]', facepart_index[name], 'pts', pts)

				cv2.polylines(face_part_all_img, [pts], True, (0, 255, 0), thickness=1)

			cv2.imshow('face_part_all',face_part_all_img)
			cv2.waitKey(0)

		# ---------------------------------------------------------------------------------------
		# 지정한 영역만 표시
		if args['face_part']:
			face_part_img = image.copy()

			face_part = args['face_part'].split(',')
			face_part_num = len(face_part)
			# print(face_part, face_part_num)

			for (_, pts_name) in enumerate(face_part):
				index_name = 'part_'+pts_name
		#		print('_', _, 'name', pts_name, 'index_name', index_name, 'facepart_index[index_name]', facepart_index[index_name][0])
				pts = np.zeros((len(facepart_index[index_name]), 2), np.int32)
				for i, j in enumerate(facepart_index[index_name]):
					if j <= 68 :
						pts[i] = [shape[j][0], shape[j][1]]
					else:
						pts[i] = [additional_point[j][0], additional_point[j][1]]
				#	print('facepart_index[name]',facepart_index[index_name], 'i',i,'j',j, '_',_)
				cv2.polylines(face_part_img, [pts], True, (0, 255, 0), thickness=1)

			cv2.imshow('face_part',face_part_img)
			cv2.waitKey(0)

		# ---------------------------------------------------------------------------------------
		# 지정한 포인트대로 자르기
		if args['face_part_point']:

			face_part_pts_img = image.copy()
			# 영역별로 x,y좌표를 저장하는 배열
			face_part_list = []

			#영역이 여러개일 경우
			if '/' in args['face_part_point'] :
				face_part = args['face_part_point'].split('/')
				# print('face_part', face_part)

				for i,j in enumerate(face_part) :
					face_part_pts_crop = image.copy()
				#	print('i', i,'j',j)
					face_part_point = face_part[i].split(',')
					face_part_point_num = len(face_part_point)
					pts = np.zeros((len(face_part_point), 2), np.int32)

					for (_, pts_name) in enumerate(face_part_point):
						#	print('_', _, 'pts_name', pts_name, 'face_part_point',face_part_point)

						if int(pts_name) <= 68:
							pts[_] = [shape[int(pts_name)][0], shape[int(pts_name)][1]]
						else:
							pts[_] = [additional_point[int(pts_name)][0], additional_point[int(pts_name)][1]]

					face_part_list.append(pts)
				#	print('face_part_list', face_part_list)

					# 선택한 영역만 표시
					cv2.polylines(face_part_pts_img, [pts], True, (0, 255, 0), thickness=1)

					# 선택한 영역만 자르기

					# 선택한 영역 크롭하기
					rect = cv2.boundingRect(pts)
					x, y, w, h = rect
					# 선택된 영역이 이미지 넓이를 벗어난 경우 예외처리
					if x < 0:
						x = 0
					elif y < 0:
						y = 0
					croped = face_part_pts_crop[y:y + h, x:x + w].copy()

					# 마스크로 영역 지정
					pts = pts - pts.min(axis=0)
					mask = np.zeros(croped.shape[:2], np.uint8)
					cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

					# 검은색 배경으로 자르기
					black_background = cv2.bitwise_and(croped, croped, mask=mask)

					# 흰색 배경으로 자르기
					bg = np.ones_like(croped, np.uint8) * 255
					cv2.bitwise_not(bg, bg, mask=mask)
					white_backgroud = bg + black_background

					output_dir = args["image"] + 'output/'
					if not os.path.exists(output_dir):
						os.mkdir(output_dir)
					cv2.imwrite(output_dir + 'p_' + img_name, face_part_pts_img)
					cv2.imwrite(output_dir + 'c_' + img_name, croped)
					cv2.imwrite(output_dir + 'b_' + img_name, black_background)


					#cv2.imshow('face_part_crop', face_part_pts_img)
					#cv2.imshow("crop", croped)
					#cv2.imshow("mask", mask)
					#cv2.imshow("black_background", black_background)
					#cv2.imshow("white_backgroud", white_backgroud)
					#cv2.waitKey(0)
				    #cv2.destroyAllWindows()

			# 영역이 한개일 경우
			else:
				face_part_pts_crop = image.copy()
				face_part_point = args['face_part_point'].split(',')
				face_part_point_num = len(face_part_point)
				pts = np.zeros((len(face_part_point), 2), np.int32)

				for (_, pts_name) in enumerate(face_part_point):
				#	print('_', _, 'pts_name', pts_name, 'face_part_point',face_part_point)

					if int(pts_name) <= 68 :
						pts[_] = [shape[int(pts_name)][0], shape[int(pts_name)][1]]
					else:
						pts[_] = [additional_point[int(pts_name)][0], additional_point[int(pts_name)][1]]

				# print('pts', pts)

				# 선택한 영역만 표시
				cv2.polylines(face_part_pts_img, [pts], True, (0, 255, 0), thickness=1)

				# 선택한 영역만 자르기

				#선택한 영역 크롭하기
				rect = cv2.boundingRect(pts)
				x, y, w, h = rect
				#print('crop',x,y,w,h)
				#print('crop_point',y, y+h, x, x+w)
				# 선택된 영역이 이미지 넓이를 벗어난 경우 예외처리
				if x<0 :
					x = 0
				elif y<0 :
					y = 0
				croped = face_part_pts_crop[y:y + h, x:x + w].copy()

				# 마스크로 영역 지정
				pts = pts - pts.min(axis=0)
				mask = np.zeros(croped.shape[:2], np.uint8)
				cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

				# 검은색 배경으로 자르기
				black_background = cv2.bitwise_and(croped, croped, mask=mask)

				# 흰색 배경으로 자르기
				bg = np.ones_like(croped, np.uint8) * 255
				cv2.bitwise_not(bg, bg, mask=mask)
				white_backgroud = bg + black_background


				output_dir = args["image"] + 'output1/'
				if not os.path.exists(output_dir):
					os.mkdir(output_dir)
				cv2.imwrite(output_dir+'p_' + img_name, face_part_pts_img)
				cv2.imwrite(output_dir + 'c_' + img_name, croped)
				cv2.imwrite(output_dir + 'b_' + img_name, black_background)

				#cv2.imshow('face_part_crop', face_part_pts_img)
				#cv2.imshow("crop", croped)
				#cv2.imshow("mask", mask)
				#cv2.imshow("black_background", black_background)
				#cv2.imshow("white_backgroud", white_backgroud)
				#cv2.waitKey(0)
				#cv2.destroyAllWindows()
				#args["image"] + img_name




