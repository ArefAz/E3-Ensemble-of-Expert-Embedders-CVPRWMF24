import torch
import os
from CustomDataset import CustomDataset, ExemplarDataset
from sklearn.metrics import accuracy_score,f1_score, classification_report

from tqdm import tqdm
from iCaRLModel import iCaRLModel

import os
import numpy as np
import torch
from torchvision.io import read_image
from PIL import Image


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from pathlib import Path
from torchvision.models import resnet18,resnet50
import copy

from mislnet import MISLNet

# Had to add this because I was having 'runtime error: too many open files'
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################## HYPERPARAMETERS #####################################

num_epochs = 200
lr = 1e-4

####################################################################################


transform = transforms.Compose([
		transforms.RandomCrop(256),
		transforms.ToTensor(),
		# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def get_acc_f1_per_class(dataset_t, icarl):

	dataloader_t = DataLoader(dataset_t, batch_size=32, num_workers=4, shuffle=True)
	result = []
	truth = []

	for _, img, label in tqdm(dataloader_t, desc="Evaluating"):

		result.extend(icarl.classify(img.to(device)))
		truth.extend(label)

	result = [t.item() for t in result]
	truth = [t.item() for t in truth]

	# Print classification report
	return classification_report(truth, result, digits=4)


#############################  LOAD FOUNDATION MODEL AND EXEMPLAR SET #################################

# Checkpoint
lightning_checkpoint_path = '/media/nas2/Aref/share/continual_learning/models/mislnet/epoch=32-step=120681-v_loss=0.0376-v_acc=0.9874.ckpt'

checkpoint = torch.load(lightning_checkpoint_path)
model_state_dict = checkpoint['state_dict']
model_state_dict = {key.replace('classifier.', '', 1): value for key, value in model_state_dict.items()}

# Instantiate iCaRLModel
icarl = iCaRLModel(model_state_dict, 'mislnet', total_memory=1000)

real_exemplars = 'exemplar-set-real-mislnet.pt'
gan_exemplars = 'exemplar-set-gan-mislnet.pt'

if os.path.exists(real_exemplars) and os.path.exists(gan_exemplars):
	# If already computed, you can just load the exemplars and assign it to the icarl class
	exemplar_real = torch.load('exemplar-set-real-mislnet.pt')
	exemplar_gan = torch.load('exemplar-set-gan-mislnet.pt')


else:
	# This portion selects exemplar set from the entire training data. Datasize: 117K each real and gan
	txt_real_file_paths = ['/media/nas2/Aref/share/continual_learning/dataset_file_paths/db-real/train.txt']
	train_dataset = CustomDataset(txt_file_paths=txt_real_file_paths, transform=transform)
	result = icarl.select_exemplars(train_dataset)
	torch.save(result[0],real_exemplars)

	txt_gan_file_paths = ['/media/nas2/Aref/share/continual_learning/dataset_file_paths/db-gan/train.txt']
	train_dataset = CustomDataset(txt_file_paths=txt_gan_file_paths, transform=transform)
	result = icarl.select_exemplars(train_dataset)
	torch.save(result[0],gan_exemplars)


icarl.assign_exemplars(exemplar_real,0)
icarl.assign_exemplars(exemplar_gan,1)

#####################################################################################################

# Calculate mean of exemplars for classification
with torch.no_grad():
	icarl.calculate_mean_exemplars()


##############################  INCREMENTALLY LEARN NEW GENERATORS  #################################

generator_names = 	[ 'TT'
			, 'SD'
			, 'EG3D'
			, 'DALLE2'
			]

test_sets = [ 'GAN'
			, 'TT'
			, 'SD'
			, 'EG3D'
			, 'DALLE2'
			]

generators_file_path = ['/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-tt/'			# Task 2
			 , '/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-sd-500/'				# Task 3
			 , '/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-eg3d/'					# Task 4
			 , '/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-dalle2/'				# Task 5
			 ]

test_file_paths = ['/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-gan-500/test.txt'		# Task 1
			 , '/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-tt/test.txt'			# Task 2
			 , '/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-sd-500/test.txt'		# Task 3
			 , '/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-eg3d/test.txt'			# Task 4
			 , '/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-dalle2/test.txt'		# Task 5
			 ]

classification_report_arr = []
final_result = ''

for i in range(len(generators_file_path)):

	######################### UPDATE THE NETWORK ###############################
	print(f'Updating network with data in {generators_file_path[i]}')
	# Adjust the number of empty strings to be one less than the desired output
	train_file_paths = [''] * (i + 2) + [generators_file_path[i] + 'train.txt']

	new_train_dataset = CustomDataset(txt_file_paths=train_file_paths, transform=transform)

	icarl.update_representation(new_train_dataset, num_epochs=num_epochs, learning_rate=lr)


	######################### SELECT EXEMPLARS #################################
	print("Selecting exemplars")
	train_data_path = [generators_file_path[i] + 'train.txt']
	new_train_dataset = CustomDataset(txt_file_paths=train_data_path, transform=transform)
	new_exemplars = icarl.select_exemplars(new_train_dataset)

	icarl.assign_exemplars(new_exemplars[0], i+2, append=False)

	icarl.reduce_exemplar_sets()

	######################### TEST REAL VS CURRENT GENERATOR ####################

	classification_report_temp_arr = []

	for j in range(len(test_file_paths[:i+2])):
		# If i > j-1, need to add empty strings to make dataset with class
		# if i>j-1:
		# 	back_add = ['']*(i-j+1)
		# else:
		# 	back_add = []

		test_data_paths = ['/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-real-500/test.txt'] +\
						[''] * j + [test_file_paths[j]] 				#+ back_add


		print(f'Testing Real Vs {test_sets[j]} in path: {test_data_paths}')
		final_result+='\n'+ f'Testing Real Vs {test_sets[j]} in path: {test_data_paths}:\n'

		test_dataset = CustomDataset(txt_file_paths=test_data_paths, transform=transform)
		accuracy_report = get_acc_f1_per_class(test_dataset, icarl)

		print(accuracy_report)
		final_result+=accuracy_report

		classification_report_temp_arr.append(accuracy_report)
	

	classification_report_arr.append(classification_report_temp_arr)



	print(final_result)


	


