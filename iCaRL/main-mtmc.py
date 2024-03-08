import torch
import os
from CustomDataset import CustomDataset

import os
import numpy as np
import torch
import torch
import torchvision.transforms as transforms

from HelperFunctions import *
from MTMCiCaRLModel import MTMCiCaRLModel


# Had to add this because I was having 'runtime error: too many open files'
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################## HYPERPARAMETERS #####################################

num_epochs = 150
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
mtmc_model = MTMCiCaRLModel(model_state_dict, 'mislnet', total_memory=1000)

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
	result = mtmc_model.select_exemplars(train_dataset)
	torch.save(result[0],real_exemplars)

	txt_gan_file_paths = ['/media/nas2/Aref/share/continual_learning/dataset_file_paths/db-gan/train.txt']
	train_dataset = CustomDataset(txt_file_paths=txt_gan_file_paths, transform=transform)
	result = mtmc_model.select_exemplars(train_dataset)
	torch.save(result[0],gan_exemplars)


mtmc_model.assign_exemplars(exemplar_real,0)
mtmc_model.assign_exemplars(exemplar_gan,1)

#####################################################################################################

# Calculate mean of exemplars for classification
with torch.no_grad():
	mtmc_model.calculate_mean_exemplars()


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

	mtmc_model.update_representation(new_train_dataset, num_epochs=num_epochs, learning_rate=lr)


	######################### SELECT EXEMPLARS #################################
	print("Selecting exemplars")
	train_data_path = [generators_file_path[i] + 'train.txt']
	new_train_dataset = CustomDataset(txt_file_paths=train_data_path, transform=transform)
	new_exemplars = mtmc_model.select_exemplars(new_train_dataset)

	mtmc_model.assign_exemplars(new_exemplars[0], i+2, append=False)

	mtmc_model.reduce_exemplar_sets()

	######################### TEST REAL VS CURRENT GENERATOR ####################

	classification_report_temp_arr = []

	for j in range(len(test_file_paths[:i+2])):

		test_data_paths = ['/media/nas2/Aref/share/continual_learning/dataset_file_paths/dn-real-500/test.txt'] +\
						 [test_file_paths[j]] 				#+ back_add

		print(f'Testing Real Vs {test_sets[j]} in path: {test_data_paths}')
		final_result+='\n'+ f'Testing Real Vs {test_sets[j]} in path: {test_data_paths}:\n'
		
		test_dataset = CustomDataset(txt_file_paths=test_data_paths, transform=transform)
		accuracy_report, roc_auc = classification_report_auroc(test_dataset, mtmc_model)

		print(accuracy_report, roc_auc)
		final_result+=accuracy_report + '\nROC_AUC:' + str(roc_auc)

		classification_report_temp_arr.append(accuracy_report)
	

	classification_report_arr.append(classification_report_temp_arr)



	print(final_result)
