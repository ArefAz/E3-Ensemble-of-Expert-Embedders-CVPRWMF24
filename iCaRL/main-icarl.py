import torch
import os
from CustomDataset import CustomDataset
from tqdm import tqdm
from iCaRLModel import iCaRLModel

import os
import numpy as np
import torch
from torchvision.io import read_image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18,resnet50

from HelperFunctions import *

# Had to add this because I was having 'runtime error: too many open files'
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################## HYPERPARAMETERS #####################################
num_epochs = 1
lr = 1e-5
####################################################################################

transform = transforms.Compose([
		transforms.RandomCrop(256),
		transforms.ToTensor(),
])
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

generator_names = 	[ 'TT', 'SD', 'EG3D', 'DALLE2']
test_sets = [ 'GAN', 'TT', 'SD', 'EG3D', 'DALLE2']

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

final_result = 'Task\tTrainedOn\tTestedOn\tAccuracy\tROC\n'


######################### TEST WITHOUT TRAINING ###############################
for j in range(len(test_file_paths)):

		test_data_paths = ['/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-real-500/test.txt'] +\
						[''] * j + [test_file_paths[j]] 				#+ back_add

		print(f'Testing Real Vs {test_sets[j]} in path: {test_data_paths}')
		
		test_dataset = CustomDataset(txt_file_paths=test_data_paths, transform=transform)
		accuracy, roc_auc = classification_report_auroc(test_dataset, icarl)

		final_result+=f'0\tGAN\t{test_sets[j]}\t{accuracy}\t{roc_auc}\n'

###############################################################################

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


	for j in range(len(test_file_paths)):

		test_data_paths = ['/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-real-500/test.txt'] +\
						[''] * j + [test_file_paths[j]] 				#+ back_add

		print(f'Testing Real Vs {test_sets[j]} in path: {test_data_paths}')
		
		test_dataset = CustomDataset(txt_file_paths=test_data_paths, transform=transform)
		accuracy, roc_auc = classification_report_auroc(test_dataset, icarl)

		final_result+=f'{i+1}\t{generator_names[i]}\t{test_sets[j]}\t{accuracy}\t{roc_auc}\n'


	print(final_result)

print(final_result)
