# Training DenseFuse network
# auto-encoder

import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam,SGD
from torch.autograd import Variable
import utils
# from net import DenseFuse_net,medtrain
from net_dcd import DenseFuse_net,medtrain,DY_MedFuse_net
from args_fusion import args
import pytorch_msssim
# from torch_ssim import ssim as tcssim
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
# from skimage.measure import compare_ssim as ssim
from pytorchtools import EarlyStopping  ##考虑l2正则化与早停止的利弊

##训练模型需要损失函数与框架


def main():
	# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
	# original_imgs_path = utils.list_images(args.dataset,'s1')  ##根据后缀查找图片
	original_imgs_path = utils.list_images(args.dataset,'ct1')  ##根据后缀查找图片
	valid_imgs_path = utils.list_images(args.valid_path,'_1.')  ##根据后缀查找图片

	train_num = 17232  ##nyu2
	# train_num = 40000 ##coco
	# train_num = 130 ##medical
	# train_num = 9900

	original_imgs_path = original_imgs_path[:train_num] ##挑选40000张图片,然后顺序打乱
	random.shuffle(original_imgs_path)
	# for i in range(5):
	i = 1    ##ssim的权重
	train(i, original_imgs_path,valid_imgs_path)


def train(i, original_imgs_path,valid_imgs_path):

	batch_size = args.batch_size

	# load network model, RGB
	in_c = 1 # 1 - gray; 3 - RGB
	if in_c == 1:
		img_model = 'L'
	else:
		img_model = 'RGB'
	input_nc = in_c
	output_nc = in_c

	##train pre-train model
	# densefuse_model = DenseFuse_net(input_nc, output_nc)

	##固定参数训练
	densefuse_model = medtrain(input_nc, output_nc)

	##finetune model
	# densefuse_model = DenseFuse_net(input_nc, output_nc)
	if args.trained_model is not None:
		print('Resuming, initializing using weight from {}.'.format(args.trained_model)) ##初始化权重
		densefuse_model.load_state_dict(torch.load(args.trained_model))
	# optimizer = SGD(filter(lambda p: p.requires_grad, densefuse_model.parameters()), lr=0.0001, momentum=0.90,
	# 					  weight_decay=0.0005)  # 关键是优化器中通filter来过滤掉那些不可导的参数
	# # optimizer =Adam(filter(lambda p: p.requires_grad, densefuse_model.parameters()))

	optimizer = Adam(densefuse_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim

	if args.cuda:
		densefuse_model.cuda()

	tbar = trange(args.epochs)  ##类似于range(),每个epoch输出打印进度条
	print('Start training.....')

	# creating save path
	temp_path_model = os.path.join(args.save_model_dir, args.ssim_path[i])
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	temp_path_loss = os.path.join(args.save_loss_dir, args.ssim_path[i])
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	imgee_set_ir2 = []
	imgee_set_s1s2 = []
	image_set_groudtruth= []
	# load training database
	image_set_ir1, batches = utils.load_dataset(original_imgs_path, batch_size)

	##multi-focus set
	# for j in range(len(image_set_ir1)):
	# 	imgee_set_ir2.append(image_set_ir1[j].replace('s1', 's2'))
	# 	image_set_groudtruth.append(image_set_ir1[j].replace('s1', 'image'))
	# 	imgee_set_s1s2.append((image_set_ir1[j], imgee_set_ir2[j]))

	#medical image set
	for j in range(len(image_set_ir1)):
		imgee_set_ir2.append(image_set_ir1[j].replace('ct1', 'mr2'))
		imgee_set_s1s2.append((image_set_ir1[j], imgee_set_ir2[j]))

	##mri-pet
	# for j in range(len(image_set_ir1)):
	# 	imgee_set_ir2.append(image_set_ir1[j].replace('vis_','ir'))
	# 	imgee_set_s1s2.append((image_set_ir1[j], imgee_set_ir2[j]))

	Loss_pixel = []
	Loss_ssim = []
	Loss_all = []
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		densefuse_model.train()
		count = 0
		for batch in range(batches):
			##获得batchs,每个batch包含四对样本
			# image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)] ##batch对应的样本
			image_paths = imgee_set_s1s2[batch * batch_size:(batch * batch_size + batch_size)]

			groudtruth_paths = image_set_groudtruth[batch * batch_size:(batch * batch_size + batch_size)] ##multi-focus image groudtruth
			img1,img2,gts = utils.get_train_images_auto(image_paths,groudtruth_paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model)

			# img1, img2, gts = utils.get_train_images_auto(image_paths, groudtruth_paths=None, height=args.HEIGHT,
			# 											  width=args.WIDTH, mode=img_model)##medical image

			count += 1
			optimizer.zero_grad()
			img1 = Variable(img1, requires_grad=False)
			img2 = Variable(img2, requires_grad=False)

			if args.cuda:
				img1 = img1.cuda()
				img2 = img2.cuda()
				if isinstance(gts, list):
					print("no groudtruth")
				else:
					gts = gts.cuda()
			# get fusion image
			# encoder
			en = densefuse_model.encoder(img1,img2)
			# decoder
			outputs = densefuse_model.decoder(en)
			# resolution loss
			# x = Variable(gts.data.clone(), requires_grad=False) ##groudtruth
			x = Variable(img1.data.clone(), requires_grad=False)

			ssim_loss_value = 0.
			pixel_loss_value = 0.

			##pre-trained model
			# for output in outputs:
			# 	pixel_loss_temp = mse_loss(output, Variable(gts.data.clone(),
			# 												requires_grad=False))  # + mse_loss(output,  Variable(img2.data.clone(), requires_grad=False) )
			# 	ssim_loss_temp = ssim_loss(output, Variable(gts.data.clone(), requires_grad=False),
			# 							   normalize=True)  # + ssim_loss(output, Variable(img2.data.clone(), requires_grad=False), normalize=True)

			# ##medical model
			for output in outputs:
				pixel_loss_temp = mse_loss(output, x) + mse_loss(output,  Variable(img2.data.clone(), requires_grad=False) )
				ssim_loss_temp = ssim_loss(output, x, normalize=True) + ssim_loss(output, Variable(img2.data.clone(), requires_grad=False), normalize=True)

			##pet-mri model
			# for output in outputs:
			# 	pixel_loss_temp = mse_loss(output, x) + mse_loss(output,  Variable(img2.data.clone(), requires_grad=False) ) #+ 300*tf.reduce_mean(tf.square(gradient(output.cpu()) -gradient (img1)))
			# 	ssim_loss_temp = ssim_loss(output, x, normalize=True) + ssim_loss(output, Variable(img2.data.clone(), requires_grad=False), normalize=True)
			# 	# gradient_loss_temp = 300*mse_loss(gradient(output),gradient (x)) + 0*mse_loss(gradient(output),gradient (Variable(img2.data.clone(), requires_grad=False)))
				# gradient_loss_temp += gradient_loss_temp
				# gradient_loss_value = gradient_loss_temp
				ssim_loss_value += (2-ssim_loss_temp)
				pixel_loss_value += pixel_loss_temp
			ssim_loss_value /= len(outputs)
			# gradient_loss_value /= len(outputs)
			pixel_loss_value /= len(outputs)

			# total loss
			total_loss = args.ssim_weight[i] * ssim_loss_value #+ pixel_loss_value
			# total_loss = pixel_loss_value + 1*gradient_loss_value ##梯度损失

			total_loss.backward()
			optimizer.step()

			all_ssim_loss += ssim_loss_value.item()
			all_pixel_loss += pixel_loss_value.item()
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), e + 1, count, batches,
								  all_pixel_loss / args.log_interval,
								  all_ssim_loss / args.log_interval,
								  (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)

				all_ssim_loss = 0.
				all_pixel_loss = 0.

		densefuse_model.eval()
		# densefuse_model.cpu()
		save_model_filename = args.ssim_path[i] + '/' "epoch_" + str(e) + "_" + \
							  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
								  i] + ".model"
		save_model_path = os.path.join(args.save_model_dir, save_model_filename)
		##earlystopping valid
		early_stopping = EarlyStopping(save_name=save_model_path,
									   patience=10, verbose=True)

		valid_set_c1c2=[]
		valid_set_ir2 =[]
		for va in range(len(valid_imgs_path)):
			valid_set_ir2.append(valid_imgs_path[va].replace('_1', '_2'))
			valid_set_c1c2.append((valid_imgs_path[va], valid_set_ir2[va]))

		valid_img1, valid_img2, gts = utils.get_train_images_auto(valid_set_c1c2, groudtruth_paths=None, height=args.HEIGHT,
													  width=args.WIDTH, mode=img_model)  ##medical image

		valid_img1 = Variable(valid_img1, requires_grad=False)
		valid_img2 = Variable(valid_img2, requires_grad=False)
		if args.cuda:
			valid_img1 = valid_img1.cuda()
			valid_img2 = valid_img2.cuda()
		valid_en = densefuse_model.encoder(valid_img1, valid_img2)
		valid_outputs = densefuse_model.decoder(valid_en)

		valid_ssim_loss_value = 0.
		valid_pixel_loss_value = 0.

		for valid_output in valid_outputs:
			valid_pixel_loss_value = mse_loss(valid_output, Variable(valid_img1.data.clone(), requires_grad=False)) + mse_loss(valid_output, Variable(valid_img2.data.clone(), requires_grad=False))
			valid_ssim_loss_value = ssim_loss(valid_output, Variable(valid_img1.data.clone(), requires_grad=False), normalize=True) + ssim_loss(valid_output, Variable(valid_img2.data.clone(),
																							   requires_grad=False),
																			  normalize=True)

			valid_ssim_loss_value += (2 - valid_ssim_loss_value)
			valid_pixel_loss_value += valid_pixel_loss_value
		valid_ssim_loss_value /= len(valid_outputs)
		valid_pixel_loss_value /= len(valid_outputs)

		# total loss
		valid_loss = valid_ssim_loss_value #valid_pixel_loss_value + args.ssim_weight[i] * valid_ssim_loss_value

		early_stopping(valid_loss, densefuse_model)

		if early_stopping.early_stop:
			print("Early stopping")
			break


	# pixel loss
	loss_data_pixel = np.array(Loss_pixel)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
	# SSIM loss
	loss_data_ssim = np.array(Loss_ssim)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
	# all loss
	loss_data_total = np.array(Loss_all)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_total_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_total': loss_data_total})

	# evaluation
	# densefuse_model.eval()
	# # densefuse_model.cpu()
	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_dy_decoder1_" + args.ssim_path[i] + ".model"
	save_model_path1 = os.path.join(args.save_model_dir, save_model_filename)

	state = {'model': densefuse_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': args.epochs}
	torch.save(state, save_model_path1)
	# torch.save(densefuse_model.state_dict(), save_model_path1) ##save model



	print("\nDone, trained model saved at", save_model_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Gradient_Net(torch.nn.Module):
  def __init__(self):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

    self.weight_x = torch.nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = torch.nn.Parameter(data=kernel_y, requires_grad=False)

  def forward(self, x):
    grad_x = torch.nn.functional.conv2d(x, self.weight_x)
    grad_y = torch.nn.functional.conv2d(x, self.weight_y)
    gradient = torch.abs(grad_x) + torch.abs(grad_y)
    return gradient

def gradient(x):
	gradient_model = Gradient_Net().to(device)
	g = gradient_model(x)
	return g


if __name__ == "__main__":
	main()
