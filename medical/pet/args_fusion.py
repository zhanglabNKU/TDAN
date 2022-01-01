
class args():

	# training args
	epochs = 100 #"number of training epochs, default is 2"
	batch_size = 8 #"batch size for training, default is 4"
	dataset = "/home/hww/fusion0820/NYUv2pics/crop_path_13/"
	# dataset = "/home/hww/fusion0820/data/train/"
	# dataset = "/home/hww/Downloads/coco/crop_path/"
	# dataset = "/home/hww/fusion0820/MF_dataset/Code/data/"
	valid_path = "/home/hww/fusion0820/data/valid/"
	HEIGHT = 256
	WIDTH = 256

	save_model_dir = "./models/dcd_encoder" # "path to folder where trained model will be saved."
	save_epoch_model_dir = "./models/epoch_models" # "path to folder where trained model will be saved."
	save_loss_dir = "./models/dcd_encoder/loss"  # "path to folder where trained model will be saved."

	image_size = 256 #"size of training images, default is 256 X 256"
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"
	ssim_weight = [1,10,100,1000,10000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 1e-4 #"learning rate, default is 0.001"
	lr_light = 1e-4  # "learning rate, default is 0.001"
	log_interval = 5 #"number of images after which the training loss is logged, default is 500"
	resume = '/home/hww/fusion0820/densefuse-pytorch-master/models_pre_nyu_med20200918/1e1/Final_epoch_4_Fri_Sep_18_12_39_01_2020_multi-focus_1e1.model'
	# resume = './models/1e0/Epoch_181_iters_1000_Tue_Oct_27_17_01_30_2020_1e0.model'

	# resume = './models/1e0/Epoch_100_iters_1000_Wed_Oct_28_06_50_34_2020_1e0.model'

	# trained_model = '/home/hww/fusion0820/20210421/dcd_encoder/models/dcd_encoder/1e1/Final_epoch_20_Tue_May_25_04_47_35_2021_dy_decoder1_1e1.model'
	# trained_model = './models/dcd_encoder/1e3/Final_epoch_3000_Tue_May__4_23_53_44_2021_multi-focus_1e3.model'
	# trained_model = None
	# resume = None


	resume_auto_en = None
	resume_auto_de = None
	resume_auto_fn = None

	# for test Final_cat_epoch_9_Wed_Jan__9_04_16_28_2019_1.0_1.0.model
	# model_path_gray = "./models/1e2/Final_epoch_4_Sat_Sep_12_12_14_06_2020_1e2.model"
	# model_path_gray = "./models/1e2/Final_epoch_4_Tue_Sep_15_19_52_49_2020_1e2.model"
	# model_path_gray = "./models/1e1/Final_epoch_4_Fri_Sep_18_12_39_01_2020_multi-focus_1e1.model"##nyu

	##medical image mogdel
	# model_path_gray = "./models_pre_nyu_med20200918/1e0/Final_epoch_200_Fri_Sep_18_15_50_02_2020_multi-focus_1e0.model"
	# model_path_gray = "./models_pre_feafreez20200920/1e0/Final_epoch_500_Sat_Sep_19_22_44_35_2020_multi-focus_1e0.model"
	# model_path_gray ='./models_no_pre/models/1e0/Final_epoch_1000_Wed_Oct_21_10_20_37_2020_multi-focus_1e0.model'

	##dcd_encoder
	# model_path_gray = "./models/1e0/Final_epoch_500_Mon_May__3_13_01_41_2021_multi-focus_1e0.model"
	# model_path_gray = "./models/dcd_encoder/1e0/Final_epoch_1000_Mon_May__3_18_49_04_2021_multi-focus_1e0.model"
	# model_path_gray = "./models/dcd_encoder/1e0/Final_epoch_2000_Tue_May__4_00_18_20_2021_multi-focus_1e0.model"
	# model_path_gray = "./models/dcd_encoder/1e3/Final_epoch_3000_Tue_May__4_23_53_44_2021_multi-focus_1e3.model"  ##不同模态不共享特征提取模块
	# model_path_gray = "./models/dcd_encoder/1e3/Final_epoch_3000_Mon_May_10_00_13_23_2021_multi-focus_1e3.model" ##不同模态不共享特征提取模块
	# model_path_gray = "./models/dcd_encoder/1e1/Final_epoch_2000_Tue_May_11_22_31_06_2021_multi-focus_1e1.model" ##不同模态不共享特征提取模块 addition


	# model_path_gray = "/home/hww/fusion0820/20210421/dcd_encoder/models/dcd_encoder/1e1/Final_epoch_20_Tue_May_25_04_47_35_2021_dy_decoder1_1e1.model" ##multi-focus 引导医学图像
	# model_path_gray = "/home/hww/fusion0820/20210421/dcd_encoder/models/dcd_encoder/1e1/Final_epoch_10_Tue_May_25_11_43_56_2021_dy_decoder1_1e1.model" ##multi-focus 引导医学图像
	# model_path_gray = "/home/hww/fusion0820/20210421/dcd_encoder/models/epoch_models/1e1/epoch_1_Tue_May_25_21_06_50_2021_1e1.model" ##multi-focus 引导医学图像

	# model_path_gray = "/home/hww/fusion0820/20210421/dcd_encoder+decoder/models/1e0/Final_epoch_3000_Tue_May_11_01_28_26_2021_dy_decoder1_1e0.model" ##encoder+decoder1
	# model_path_gray = "./models/dcd_encoder/1e0/Final_epoch_2000_Tue_May__4_00_18_20_2021_multi-focus_1e0.model"##encoder,no_decoder,result_pre
	# model_path_gray ="/home/hww/fusion0820/20210421/dcd_encoder+decoder/models/1e1/Final_epoch_1000_Thu_May_20_22_59_29_2021_dy_decoder1_1e1.model" ##dy_decoder1,no encoder

	# model_path_gray ="/home/hww/fusion0820/20210421/dy_fuse/models/1e1/Final_epoch_20_Tue_May_25_01_00_57_2021_dy_decoder1_1e1.model" ##dy_decoder1,no encoder

	model_path_gray ="/home/hww/fusion0820/20210421/dcd_encoder+decoder/models/1e0/123/Final_epoch_3000_Thu_May_13_17_18_58_2021_dy_decoder123_1e0.model" ##dy_decoder123

	##multi_focus model
	# model_path_gray = "./models_pre_nyu_med20200918/1e0/Final_epoch_200_Fri_Sep_18_15_50_02_2020_multi-focus_1e0.model"
	# model_path_gray ='./models/1e0/Epoch_100_iters_1000_Wed_Oct_28_06_50_34_2020_1e0.model'

	##vis-ir



	model_path_rgb = "./models/densefuse_rgb.model"




