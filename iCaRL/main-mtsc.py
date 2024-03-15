import torch
import os
from lib.CustomDataset import CustomDataset
from MTSCiCaRLModel import MTSCiCaRLModel
import os
import csv
import torch
from lib.HelperFunctions import *

from lib.FileLists import *
from datetime import datetime
import argparse

print("Script Start Time =", datetime.now().strftime("%H:%M:%S"))

# Had to add this because I was having 'runtime error: too many open files'
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################## ARGPARSE SETUP ######################################
parser = argparse.ArgumentParser(description="Process command line arguments.")
# Add arguments
parser.add_argument('--model', type=str, help="Model name", required=True)
parser.add_argument('--checkpoint', type=str, help="Path to the checkpoint file", required=True)
parser.add_argument('--feature_size', type=int, help="Feature size", required=True)
parser.add_argument('--epochs', type=int, help="Number of epochs to train", required=False, default=150)
parser.add_argument('--lr', type=float, help="Learning rate", required=False, default=1e-5)
parser.add_argument('--distil_gamma', type=float, help="Distillation paramenter", required=False, default=0.5)
parser.add_argument('--lambda_param', type=float, help="Lambda parameter used for adding BCE loss to iCaRL", required=False, default=0.1)
parser.add_argument('--temperature', type=float, help="Temperature used for KL divergence loss", required=False, default=2.0)

# Parse the arguments
args = parser.parse_args()
print(f"Running Script for Model: {args.model} using checkpoint: {args.checkpoint} and feature size is \
	  {args.feature_size}")
# Specify the output filenames
accuracy_csv, aucroc_csv = f'MTSCResultsAccuracy_{args.model}.csv', f'MTSCResultsROCAUC_{args.model}.csv'

############################## HYPERPARAMETERS #####################################
num_epochs = args.epochs
lr = args.lr
distil_gamma=args.distil_gamma
lambda_param = args.lambda_param
temperature = args.temperature

#############################  LOAD FOUNDATION MODEL AND EXEMPLAR SET #################################

# Checkpoint
checkpoint = torch.load(args.checkpoint)
model_state_dict = checkpoint['state_dict']
model_state_dict = {key.replace('classifier.', '', 1): value for key, value in model_state_dict.items()}

# Instantiate MTSC method
mtsc_model = MTSCiCaRLModel(model_state_dict, args.model, total_memory=1000, feature_size=args.feature_size)

real_exemplars=f'checkpoints/exemplar-set-real-{args.model}.pt'
gan_exemplars=f'checkpoints/exemplar-set-gan-{args.model}.pt'

if not(os.path.exists(real_exemplars) and os.path.exists(gan_exemplars)):
	# This portion selects exemplar set from the entire training data. Datasize: 117K each real and gan
	txt_real_file_paths = [train_real_file_paths]
	train_dataset = CustomDataset(txt_file_paths=txt_real_file_paths)
	result = mtsc_model.select_exemplars(train_dataset)
	torch.save(result[0],real_exemplars)

	txt_gan_file_paths = [train_gan_file_paths]
	train_dataset = CustomDataset(txt_file_paths=txt_gan_file_paths)
	result = mtsc_model.select_exemplars(train_dataset)
	torch.save(result[0],gan_exemplars)

# If already computed, you can just load the exemplars and assign it to the mtsc_icarl class
exemplar_real = torch.load(real_exemplars)
exemplar_gan = torch.load(gan_exemplars)

mtsc_model.assign_exemplars(exemplar_real,0)
mtsc_model.assign_exemplars(exemplar_gan,1)

#####################################################################################################


##############################  INCREMENTALLY LEARN NEW GENERATORS  #################################
final_result = 'Task,TrainedOn,TestedOn,Accuracy,ROC\n'

accuracy_list, roc_auc_list = [], []
######################### TEST WITHOUT TRAINING TASK 0 #########################
accuracy_temp, roc_temp = [], []
for j in range(len(test_file_paths)):
	test_data_paths = [test_file_paths_real] +\
					[test_file_paths[j]] 				#+ back_add

	print(f'Testing Real Vs {test_sets[j]} in path: {test_data_paths}')
	
	test_dataset = CustomDataset(txt_file_paths=test_data_paths)
	accuracy, roc_auc = classification_report_auroc(test_dataset, mtsc_model)

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

	mtsc_model.update_representation(new_train_dataset, num_epochs=num_epochs, learning_rate=lr, temperature=temperature, distil_gamma = distil_gamma, lambda_param=lambda_param)


	######################### SELECT EXEMPLARS #################################
	print("Selecting exemplars")
	train_data_path = [generators_file_path[i] + 'train.txt']
	new_train_dataset = CustomDataset(txt_file_paths=train_data_path)
	new_exemplars = mtsc_model.select_exemplars(new_train_dataset)

	mtsc_model.assign_exemplars(new_exemplars[0], i+2, append=False)

	mtsc_model.reduce_exemplar_sets()

	######################### TEST REAL VS CURRENT GENERATOR ####################

	accuracy_temp, roc_temp = [], []
	for j in range(len(test_file_paths)):

		test_data_paths = [test_file_paths_real] + ['']*j +\
						 [test_file_paths[j]] 

		print(f'Testing Real Vs {test_sets[j]} in path: {test_data_paths}')
		
		test_dataset = CustomDataset(txt_file_paths=test_data_paths)
		accuracy, roc_auc = classification_report_auroc(test_dataset, mtsc_model)
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
print("Script End Time =", datetime.now().strftime("%H:%M:%S"))
