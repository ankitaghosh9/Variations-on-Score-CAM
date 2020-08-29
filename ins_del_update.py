import operator
import numpy as np
import matplotlib.pyplot as plt


def sortingmap(cam_map):

	cam_list=[]
	h,w=cam_map.shape
	# h=224
	# w=224

	for i in range(0,h):
		for j in range(0,w):
			cam_list.append([cam_map[i][j],i,j])

	sorted_cam_list=sorted(cam_list, key=operator.itemgetter(0), reverse=True)

	#print(sorted_cam_list)
	return sorted_cam_list


def deletion(inp_img, cam_map, deletion_num):

	x_points=[]
	y_points=[]
	graph_points=[]
	inp_image=inp_img.copy()
	d ,h ,w =inp_img.shape
	# d=3
	# h=224
	# w=224
	del_list = sortingmap(cam_map)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
	img =torch.tensor(inp_image, device=device).float().view(1,3,224,224)
	x = X(img)
	predicted_class = x.max(1)[-1]
	_, index = torch.max(x, 1)
	percentage = torch.nn.functional.softmax(x, dim=1)[0] * 100 
	pred_value= (percentage[index[0]].item())/100
	x_points.append(0)
	y_points.append(pred_value)

	pointer=0
	while(True):
		for j in range(pointer,pointer+deletion_num):
			inp_image[0][del_list[j][1]][del_list[j][2]]=0
			inp_image[1][del_list[j][1]][del_list[j][2]]=0
			inp_image[2][del_list[j][1]][del_list[j][2]]=0


		pointer=pointer+deletion_num

		#send the new image through model for prediction
		img =torch.tensor(inp_image, device=device).float().view(1,3,224,224)
		x = X(img)
		#find the percentage prediction of the same class
		percentage = torch.nn.functional.softmax(x, dim=1)[0] * 100 
		pred_value= (percentage[index[0]].item())/100
    #appending x_axis points
		x_points.append(pointer/(w*h)) 
		#appending the predictions as y_axis points
		y_points.append(pred_value)

		#print(inp_image)
		#plt.imshow(inp_img)
		#plt.show()
		if pointer==w*h:
			break

	graph_points.append(x_points)
	graph_points.append(y_points)
	#print(graph_points)
	return graph_points

def deletiongraph(inp_img, gcam, gcampp, smcampp, sccam, isscam1, isscam2,deletion_num):

	
	gcam_points=deletion(inp_img,gcam,deletion_num)
	gcampp_points=deletion(inp_img,gcampp,deletion_num)
	smcampp_points=deletion(inp_img,smcampp,deletion_num)
	sccam_points=deletion(inp_img,sccam,deletion_num)
	isscam1_points=deletion(inp_img,isscam1,deletion_num)
	isscam2_points=deletion(inp_img,isscam2,deletion_num)


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
		ax.set(xlabel='Pixels removed', ylabel='Prediction')

	for ax in axs.flat:
		ax.label_outer()

	plt.show()

def insertion(inp_img, cam_map, insertion_num):

	x_points=[]
	y_points=[]
	graph_points=[]
	d,h,w=inp_img.shape
	new_inp_img=np.zeros((3,h,w))

	ins_list = sortingmap(cam_map)
 
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	img =torch.tensor(inp_img, device=device).float().view(1,3,224,224)
	x = X(img)
	predicted_class = x.max(1)[-1]
	_, index = torch.max(x, 1)
	percentage = torch.nn.functional.softmax(x, dim=1)[0] * 100 
	pred_value= (percentage[index[0]].item())/100
	x_points.append(0)
	y_points.append(0)
	
	pointer=0
	while(True):
		for j in range(pointer,pointer+insertion_num):
			new_inp_img[0][ins_list[j][1]][ins_list[j][2]]=inp_img[0][ins_list[j][1]][ins_list[j][2]]
			new_inp_img[1][ins_list[j][1]][ins_list[j][2]]=inp_img[1][ins_list[j][1]][ins_list[j][2]]
			new_inp_img[2][ins_list[j][1]][ins_list[j][2]]=inp_img[2][ins_list[j][1]][ins_list[j][2]]


		pointer=pointer+insertion_num

		#send the new image through model for prediction
		img =torch.tensor(new_inp_img, device=device).float().view(1,3,224,224)
		x = X(img)
		#find the percentage prediction of the same class
		percentage = torch.nn.functional.softmax(x, dim=1)[0] * 100 
		pred_value= (percentage[index[0]].item())/100
    #appending x_axis points
		x_points.append(pointer/(w*h)) 
		#appending the predictions as y_axis points
		y_points.append(pred_value)

		#print(new_inp_img)
		#plt.imshow(inp_img)
		#plt.show()
		if pointer==w*h:
			break

	graph_points.append(x_points)
	graph_points.append(y_points)
	#print(graph_points)
	return graph_points

def insertiongraph(inp_img, gcam, gcampp, smcampp, sccam, isscam1, isscam2, insertion_num):

	
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
