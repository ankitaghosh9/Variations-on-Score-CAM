###################################### ALGORITHM CODE ##################################################

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

def deletiongraph(inp_img, gcam, gcampp, smcampp, sccam, isscam1, isscam2,deletion_num):

	
	gcam_points=deletion(inp_img,gcam,deletion_num)
	gcampp_points=deletion(inp_img,gcampp,deletion_num)
	smcampp_points=deletion(inp_img,smcampp,deletion_num)
	sccam_points=deletion(inp_img,sccam,deletion_num)
	isscam1_points=deletion(inp_img,isscam1,deletion_num)
	isscam2_points=deletion(inp_img,isscam2,deletion_num)


	plt.plot(gcam_points[0], gcam_points[1],label="Grad-CAM")
	plt.plot(gcampp_points[0], gcampp_points[1], label="Grad-CAM++")
	plt.plot(smcampp_points[0], smcampp_points[1],label="SGrad-CAM++")
	plt.plot(sccam_points[0], sccam_points[1], label="Score-CAM")
	plt.plot(isscam1_points[0], isscam1_points[1],label="ISS-CAM1")
	plt.plot(isscam2_points[0], isscam2_points[1],label="ISS-CAM2")

	plt.xlabel('Pixels removed')
	plt.ylabel('Prediction')

	plt.legend()

	plt.show()


def insertiongraph(inp_img, gcam, gcampp, smcampp, sccam, isscam1, isscam2,insertion_num):

	
	gcam_points=insertion(inp_img,gcam,insertion_num)
	gcampp_points=insertion(inp_img,gcampp,insertion_num)
	smcampp_points=insertion(inp_img,smcampp,insertion_num)
	sccam_points=insertion(inp_img,sccam,insertion_num)
	isscam1_points=insertion(inp_img,isscam1,insertion_num)
	isscam2_points=insertion(inp_img,isscam2,insertion_num)


	plt.plot(gcam_points[0], gcam_points[1],label="Grad-CAM")
	plt.plot(gcampp_points[0], gcampp_points[1], label="Grad-CAM++")
	plt.plot(smcampp_points[0], smcampp_points[1],label="SGrad-CAM++")
	plt.plot(sccam_points[0], sccam_points[1], label="Score-CAM")
	plt.plot(isscam1_points[0], isscam1_points[1],label="ISS-CAM1")
	plt.plot(isscam2_points[0], isscam2_points[1],label="ISS-CAM2")

	plt.xlabel('Pixels inserted')
	plt.ylabel('Prediction')

	plt.legend()

	plt.show()
  
  ################################# ALROGITHM CODE END #########################################################################
  
  ################################## STORING THE CAM ARRAYS IN LIST DURING MODEL EVALUATION #######################################
  
df1 = df[110:120]
#print(df1)
#x5 = [['/gdrive/My Drive/ILSVRC2012_val_00000653.JPEG']]
#for i in range(len(df1)):
c = 0

#RISE METRICS INITIALIZATION
input_list=[]
scorecam_map_list=[]
scorecam_map1_list=[]
scorecam_map2_list=[]
gradcam_map_list=[]
gradcampp_map_list=[]
smpp_map_list=[]

for index, row in df1.iterrows():
  c += 1
  if c == 2:
    break
  X = model.eval()

  x_model_dict = dict(type='vgg', arch=X, layer_name='features', input_size=(224, 224))

  input_image = load_image(row['image_name'])
  input_ = apply_transforms(input_image)
  np_input_=input_.view(3,224,224).detach().cpu().numpy()
  input_list.append(np_input_)
  
  if torch.cuda.is_available():
    input_cuda = input_.cuda()
  
  x = X(input_cuda)
  predicted_class = x.max(1)[-1]


  #pred1 = X(input_cuda)[0][0]
  #predscore1 = X(input_cuda)[0][218]
  #predscore2 = X(input_cuda)[0][219]
  #p1 = X(input_cuda).max(1)

  #print(X(input_cuda)[0])
  ##print(p1)
  #print(pred1, predscore1, predscore2)
  #print(X(input_cuda)[0])
  
  print(predicted_class)

  _, index = torch.max(x, 1)
  #print(index)
  percentage = torch.nn.functional.softmax(x, dim=1)[0] * 100 
  #print(percentage)                   #Yc
  
  print(index[0], percentage[index[0]].item())

  x_scorecam = ScoreCAM(x_model_dict)
  scorecam_map = x_scorecam(input_cuda)
  np_scorecam_map=scorecam_map.view(224,224).detach().cpu().numpy()
  scorecam_map_list.append(np_scorecam_map)

  smoothpp = SmoothGradCAMpp(x_model_dict)
  smpp_map = smoothpp(input_cuda)
  np_smpp_map=smpp_map.view(224,224).detach().cpu().numpy()
  smpp_map_list.append(np_smpp_map)

  x_scorecam1 = ISS_CAM2(x_model_dict) 
  scorecam_map1 = x_scorecam1(input_cuda)
  np_scorecam_map1=scorecam_map1.view(224,224).detach().cpu().numpy()
  scorecam_map1_list.append(np_scorecam_map1)

  x_scorecam2 = ISS_CAM2(x_model_dict)
  scorecam_map2 = x_scorecam2(input_cuda)
  np_scorecam_map2=scorecam_map2.view(224,224).detach().cpu().numpy()
  scorecam_map2_list.append(np_scorecam_map2)

  gradcam = GradCAM(x_model_dict)
  gradcam_map = gradcam(input_cuda)
  np_gradcam_map=gradcam_map.view(224,224).detach().cpu().numpy()
  gradcam_map_list.append(np_gradcam_map)

  gradcampp = GradCAMpp(x_model_dict)
  gradcampp_map = gradcampp(input_cuda)
  np_gradcampp_map=gradcampp_map.view(224,224).detach().cpu().numpy()
  gradcampp_map_list.append(np_gradcampp_map)

  basic_visualize(input_cuda.cpu(), \
                  gradcam_map.type(torch.FloatTensor).cpu(), \
                  gradcampp_map.type(torch.FloatTensor).cpu(), \
                  smpp_map.type(torch.FloatTensor).cpu(), \
                  scorecam_map.type(torch.FloatTensor).cpu(), \
                  scorecam_map1.type(torch.FloatTensor).cpu(), \
                  scorecam_map2.type(torch.FloatTensor).cpu())
  
  ################################## END OF MODEL EVALUATION CODE ########################################
  
  ################################### ALGORITHM EXECUTION CODE ###########################################
  
for i in range(0,len(input_list)):
  inp_img=input_list[i]
  #print(inp_img)
  #print(inp_img.shape)
  gcam=gradcam_map_list[i]
  #print(gcam.shape)
  gcampp=gradcampp_map_list[i]
  smcampp=smpp_map_list[i]
  sccam=scorecam_map_list[i]
  isscam1=scorecam_map1_list[i]
  isscam2=scorecam_map2_list[i]

  deletiongraph(inp_img, gcam, gcampp, smcampp, sccam, isscam1, isscam2,224)
  insertiongraph(inp_img, gcam, gcampp, smcampp, sccam, isscam1, isscam2,224)
  
  
  ###################################### ALGORITHM EXECUTION CODE END ####################################################
  
  ############################################### END #####################################################################
