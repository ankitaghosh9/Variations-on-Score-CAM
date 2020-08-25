import operator
import numpy as np
import matplotlib.pyplot as plt


def sortingmap(cam_map):

	cam_list=[]
	h,w=cam_map.shape

	for i in range(0,h):
		for j in range(0,w):
			cam_list.append([cam_map[i][j],i,j])

	sorted_cam_list=sorted(cam_list, key=operator.itemgetter(0), reverse=True)

	return sorted_cam_list


def insertion(inp_img, cam_map, insertion_num):

	x_points=[]
	y_points=[]
	graph_points=[]
	h,w=inp_img.shape
	new_inp_img=np.zeros((h,w))

	ins_list = sortingmap(cam_map)
	
	pointer=0
	while(True):
		for j in range(pointer,pointer+insertion_num):
			new_inp_img[ins_list[j][1]][ins_list[j][2]]=inp_img[ins_list[j][1]][ins_list[j][2]]

		pointer=pointer+insertion_num

		#send the new image through model for prediction
		#find the percentage prediction of the same class
		x_points.append(pointer/(w*h)) #appending x_axis points
		y_points.append(0.5) #appending the predictions as y_axis points

		print(new_inp_img)
		#plt.imshow(inp_img)
		#plt.show()
		if pointer==w*h:
			break

	graph_points.append(x_points)
	graph_points.append(y_points)
	print(graph_points)
	return graph_points

def insertiongraph(inp_img, gcam, gcampp, smcampp, sccam, isscam1, isscam2,insertion_num):

	
	gcam_points=insertion(inp_img,gcam,insertion_num)
	gcampp_points=insertion(inp_img,gcampp,insertion_num)
	smcampp_points=insertion(inp_img,smcampp,insertion_num)
	sccam_points=insertion(inp_img,sccam,insertion_num)
	isscam1_points=insertion(inp_img,isscam1,insertion_num)
	isscam2_points=insertion(inp_img,isscam2,insertion_num)


	fig, axs = plt.subplots(2, 3)
	axs[0, 0].plot(gcam_points[0], gcam_points[1])
	axs[0, 0].set_title('Grad-CAM')
	axs[1, 0].plot(gcampp_points[0], gcampp_points[1])
	axs[1, 0].set_title('Grad-CAM++')
	axs[0, 1].plot(smcampp_points[0], smcampp_points[1])
	axs[0, 1].set_title('Smooth Grad-CAM++')
	axs[1, 1].plot(sccam_points[0], sccam_points[1])
	axs[1, 1].set_title('Score-CAM')
	axs[0, 2].plot(isscam1_points[0], isscam1_points[1])
	axs[0, 2].set_title('ISS-CAM1')
	axs[1, 2].plot(isscam2_points[0], isscam2_points[1])
	axs[1, 2].set_title('ISS-CAM2')

	for ax in axs.flat:
		ax.set(xlabel='Pixels added', ylabel='Prediction')

	for ax in axs.flat:
		ax.label_outer()

	plt.show()


inp_arr=np.array([[1,1,2],[2,3,4],[3,4,5]])
cam_arr=np.array([[1,1,2],[2,3,4],[3,4,5]])
insertiongraph(inp_arr,cam_arr,cam_arr,cam_arr,cam_arr,cam_arr,cam_arr,1)