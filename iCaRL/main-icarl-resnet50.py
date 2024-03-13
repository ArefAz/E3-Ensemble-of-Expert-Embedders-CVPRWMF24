import torch
import os
from lib.CustomDataset import CustomDataset
from iCaRLModel import iCaRLModel
import os
import csv
import torch
from lib.HelperFunctions import *

# Change this to FileLists
from lib.FileListsTemp import *

from datetime import datetime

print("Script Start Time =", datetime.now().strftime("%H:%M:%S"))

# Had to add this because I was having 'runtime error: too many open files'
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the output filenames
accuracy_csv, aucroc_csv = 'iCaRLResultsAccuracyResNet50.csv', 'iCaRLResultsROCAUCResNet50.csv'

############################## HYPERPARAMETERS #####################################
num_epochs = 150
lr = 1e-5

#############################  LOAD FOUNDATION MODEL AND EXEMPLAR SET #################################

# Checkpoint
checkpoint = torch.load(resnet50_lightning_checkpoint_path)
model_state_dict = checkpoint['state_dict']
model_state_dict = {key.replace('classifier.', '', 1): value for key, value in model_state_dict.items()}

# Instantiate iCaRLModel
icarl = iCaRLModel(model_state_dict, 'resnet50', total_memory=1000, feature_size=2048)

if not (os.path.exists(real_exemplars_resnet50) and os.path.exists(gan_exemplars_resnet50)):
	# This portion selects exemplar set from the entire training data. Datasize: 117K each real and gan
	txt_real_file_paths = [train_real_file_paths]
	train_dataset = CustomDataset(txt_file_paths=txt_real_file_paths)
	result = icarl.select_exemplars(train_dataset)
	torch.save(result[0],real_exemplars_resnet50)

	txt_gan_file_paths = [train_gan_file_paths]
	train_dataset = CustomDataset(txt_file_paths=txt_gan_file_paths)
	result = icarl.select_exemplars(train_dataset)
	torch.save(result[0],gan_exemplars_resnet50)

# If already computed, you can just load the exemplars and assign it to the icarl class
exemplar_real = torch.load(real_exemplars_resnet50)
exemplar_gan = torch.load(gan_exemplars_resnet50)
icarl.assign_exemplars(exemplar_real,0)
icarl.assign_exemplars(exemplar_gan,1)

#####################################################################################################

# Calculate mean of exemplars for classification
with torch.no_grad():
	icarl.calculate_mean_exemplars()


##############################  INCREMENTALLY LEARN NEW GENERATORS  #################################
final_result = 'Task,TrainedOn,TestedOn,Accuracy,ROC\n'

accuracy_list, roc_auc_list = [], []
######################### TEST WITHOUT TRAINING TASK 0 #########################
accuracy_temp, roc_temp = [], []
for j in range(len(test_file_paths)):

	test_data_paths = [test_file_paths_real] +\
					[''] * j + [test_file_paths[j]] 				#+ back_add

	print(f'Testing Real Vs {test_sets[j]} in path: {test_data_paths}')
	
	test_dataset = CustomDataset(txt_file_paths=test_data_paths)
	accuracy, roc_auc = classification_report_auroc(test_dataset, icarl)

	final_result+=f'0,GAN,{test_sets[j]},{accuracy},{roc_auc}\n'
	
	accuracy_temp.append(accuracy)
	roc_temp.append(roc_auc)

accuracy_list.append(accuracy_temp)
roc_auc_list.append(roc_temp)

###############################################################################


for i in range(len(generators_file_path)):

	######################### UPDATE THE NETWORK ###############################
	print(f'Updating network with data in {generators_file_path[i]}')
	# Adjust the number of empty strings to be one less than the desired output
	train_file_paths = [''] * (i + 2) + [generators_file_path[i] + 'train.txt']

	new_train_dataset = CustomDataset(txt_file_paths=train_file_paths)

	icarl.update_representation(new_train_dataset, num_epochs=num_epochs, learning_rate=lr)


	######################### SELECT EXEMPLARS #################################
	print("Selecting exemplars")
	train_data_path = [generators_file_path[i] + 'train.txt']
	new_train_dataset = CustomDataset(txt_file_paths=train_data_path)
	new_exemplars = icarl.select_exemplars(new_train_dataset)

	icarl.assign_exemplars(new_exemplars[0], i+2, append=False)

	icarl.reduce_exemplar_sets()

	######################### TEST REAL VS CURRENT GENERATOR ####################

	accuracy_temp, roc_temp = [], []
	for j in range(len(test_file_paths)):

		test_data_paths = [test_file_paths_real] +\
						[''] * j + [test_file_paths[j]] 				#+ back_add

		print(f'Testing Real Vs {test_sets[j]} in path: {test_data_paths}')
		
		test_dataset = CustomDataset(txt_file_paths=test_data_paths)
		accuracy, roc_auc = classification_report_auroc(test_dataset, icarl)
		accuracy_temp.append(accuracy)
		roc_temp.append(roc_auc)
		final_result+=f'{i+1},{generator_names[i]},{test_sets[j]},{accuracy},{roc_auc}\n'

	accuracy_list.append(accuracy_temp)
	roc_auc_list.append(roc_temp)

	print(final_result)
print(accuracy_list, roc_auc_list)

# Writing to the csv file
with open(accuracy_csv, mode='w', newline='') as file:
	writer = csv.writer(file)
	writer.writerows(accuracy_list)

with open(aucroc_csv, mode='w', newline='') as file:
	writer = csv.writer(file)
	writer.writerows(roc_auc_list)

print(f'Data written to {accuracy_csv} and {aucroc_csv}.')
print("Script Start Time =", datetime.now().strftime("%H:%M:%S"))
