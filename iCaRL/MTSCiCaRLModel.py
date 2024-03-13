import torch
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from mislnet import MISLNet
from lib.CustomDataset import ExemplarDataset
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MTSCiCaRLModel:
	def __init__(self, model_state_dict, neural_network='resnet50', total_memory=2000 , n_class=2, feature_size=200):
		'''
		memory_per_task:      Defaults to 1000 per class
		model_state_dict:    This is the state dict of pretrained model
		memory_size:          Total memory buffer budget
		feature_size:        Size of the feature extractor
		'''
		
		
		self.feature_size = feature_size
		self.exemplar_sets = {} # Dictionary to store exemplars for each class
		self.compute_exemplar_means = True
		self.n_class = n_class
		self.total_memory = total_memory

		# self.memory_per_task = total_memory/self.n_class
			
		if neural_network=='mislnet':
			self.classifier = MISLNet(num_classes=2)
			self.classifier.load_state_dict(model_state_dict)
			self.classifier.to(device)
			self.classifier.eval()
		else:
			raise NotImplementedError("Currently MISLNet is supported on MTSC")

		
	def reduce_exemplar_sets(self):
		# Reduce the number of exemplars per class to fit in memory
		# exemplar for real stays constant.
		print(f'Number of classes in icarl: {self.n_class}')
		for i in range(1, self.n_class):
			self.exemplar_sets[i] = self.exemplar_sets[i][:self.memory_per_task]
		


	def assign_exemplars(self, exemplars, selected_class, append=False):
		'''
		Assigns the calculated exemplars to the exemplar sets dictionary
		exemplars: N number of images that is closest to the mean of entire training set of that class (Nx256x256x3 tensor)
		selected_class: Class ID of the exemplars passed
		append: False if just need to assign. If True, need to append to the dictionary for that key.
		'''
		if not append:
			self.exemplar_sets[selected_class] = exemplars
		else:
			self.exemplar_sets[selected_class].extend(exemplars)

	
	def select_exemplars(self, class_dataset):
		'''
		Select exemplars for a class using herding
		Total number of exemplars per task: self.memory_per_task
		class_dataset: consists label and images for the given class: idx, img, label
					   should only have one class in the dataset   
		'''
		
		selected_class = class_dataset[0][2]

		feature_extractor = MISLNet(num_classes=self.n_class)  # Create a new instance of the model
		feature_extractor.load_state_dict(self.classifier.state_dict())  # Copy parameters and buffers
		# Make the last layer feature extractor
		feature_extractor.output = nn.Identity()
		
		feature_extractor.to(device) # Transfer feature extractor to cuda
		feature_extractor.eval()

		with torch.no_grad():
			print("Calculating current class mean")
			features_list = []
			
			# Use DataLoader for batch processing to speed up feature extraction
			dataloader = DataLoader(class_dataset, batch_size=32, num_workers=2, shuffle=False, pin_memory=True if device == 'cuda' else False)
			
			for _, images, _ in tqdm(dataloader, desc="Processing"):
				images = images.to(device)
				features = feature_extractor(images)  # Extract features for the batch
				features = F.normalize(features, p=2, dim=1)  # L2 Normalize the features
				features_list.append(features.cpu())  # Move features to CPU to reduce GPU memory usage
	
			# Concatenate all features and compute the mean, and then normalize
			all_features = torch.cat(features_list, dim=0)
			class_mean_feature = all_features.mean(dim=0, keepdim=True)
			class_mean_feature = F.normalize(class_mean_feature, p=2, dim=1)  # Normalize the mean feature
			
			print(f"Class mean for {selected_class} -> Calculated")
	
			exemplar_set = []
			selected_image = []
	
			# Pre-compute and store features for all images to avoid recomputing
			precomputed_features = all_features.clone().to(device)  # Move precomputed features to device for comparison
			

			# Real image stays constant, all else is divided up 
			self.memory_per_task = int((self.total_memory/2)/(self.n_class - 1))

			print(f"New memory budget per task: {self.memory_per_task}")
			
			for k in tqdm(range(self.memory_per_task), desc="Calculating exemplars"):
				if k > 0:
					# Use precomputed features to avoid redundant computation
					exemplar_features = precomputed_features[selected_image]  # Use indexing to fetch selected exemplar features
					exemplar_feature_sum = exemplar_features.sum(dim=0, keepdim=True)
				else:
					exemplar_feature_sum = torch.zeros((1, self.feature_size)).to(device)
	
				# Calculate the intermediate mean, termed "mu_tmp" here
				mu_tmp = (exemplar_feature_sum + precomputed_features) / (k + 1)
				mu_tmp = F.normalize(mu_tmp, p=2, dim=1)
	
				# Find the image index with the minimum distance
				distances = torch.norm(class_mean_feature - mu_tmp.to('cpu'), dim=1)
				image_idx_min = torch.argmin(distances).item()
	
				# Track selected images
				selected_image.append(image_idx_min)
	
				# Append the corresponding image to the exemplar set
				exemplar_set.append(class_dataset[image_idx_min][1])
	
			return exemplar_set, selected_image
			

	def calculate_mean_exemplars(self):

		#####################################################################################
		feature_extractor = MISLNet(num_classes=self.n_class)  # Create a new instance of the model
		feature_extractor.load_state_dict(self.classifier.state_dict())  # Copy parameters and buffers
		# Make the last layer feature extractor
		feature_extractor.output = nn.Identity()
		
		feature_extractor.to(device) # Transfer feature extractor to cuda
		feature_extractor.eval()
		#####################################################################################

		mean_exemplar_set = []
		for key, images in self.exemplar_sets.items():
			print(f"Calculating mean of exemplars for the class {key}")
			running_feature_sum = torch.zeros((1, self.feature_size), device=device)

			for image in images:
				image = image.unsqueeze(0).to(device)
				features = feature_extractor(image)
				features = F.normalize(features, p=2, dim=1)  # L2 Normalize the features
				running_feature_sum += features

			mean_feature = running_feature_sum / len(images)
			mean_feature = F.normalize(mean_feature, p=2, dim=1)
			mean_exemplar_set.append(mean_feature)

		print("Mean of exemplar sets calculated")
		self.mean_exemplar_set = torch.cat(mean_exemplar_set, dim=0)
		self.compute_exemplar_means = False


	def classify(self, image_batch):
		image_batch = image_batch.to(device)

		#####################################################################################
		feature_extractor = MISLNet(num_classes=self.n_class)  # Create a new instance of the model
		feature_extractor.load_state_dict(self.classifier.state_dict())  # Copy parameters and buffers
		# Make the last layer feature extractor
		feature_extractor.output = nn.Identity()
		
		feature_extractor.to(device) # Transfer feature extractor to cuda
		feature_extractor.eval()
		#####################################################################################
		
		with torch.no_grad():
			if self.compute_exemplar_means:
				self.calculate_mean_exemplars()

			batch_feature = feature_extractor(image_batch)
			batch_feature = F.normalize(batch_feature, p=2, dim=1)

			batch_expanded = batch_feature.unsqueeze(1)  # Shape becomes [batch_size, 1, feature_size]
			mean_exemplar_set_expanded = self.mean_exemplar_set.unsqueeze(0) # Shape becomes [1, classes, feature_size]
			
			distances = torch.norm(batch_expanded - mean_exemplar_set_expanded, p=2, dim=2).detach().cpu()  # Shape becomes [32, num_classes]
			
			distances_std = (distances - distances.mean()) / distances.std()
			pseudo_probabilities = F.softmax(-distances_std, dim=1).detach().cpu().numpy()
			# Find the index of the closest reference  for each element in the batch
			closest_indices = distances.argmin(dim=1)

			return closest_indices, pseudo_probabilities

	# def classify(self, image_batch):
	# 	image_batch = image_batch.to(device)
	# 	with torch.no_grad():
	# 		batch_predictions = self.classifier(image_batch)
	# 		# Classify 0 as 0, anything else is synthetic so should be 1
	# 		# batch_predictions = torch.where(batch_predictions == 0, torch.tensor(0), torch.tensor(1))
	# 		argmax_indices = torch.argmax(batch_predictions, dim=1)  # Shape: batch_size
	# 		binary_predictions = (argmax_indices != 0).detach().cpu().int()
			
	# 		pseudo_probabilities = F.softmax(batch_predictions, dim=1).detach().cpu().numpy()
	# 		return binary_predictions, pseudo_probabilities


	def expanded_network(self):
		# This method increases the last layer of network by one every time it is called.
		# Keeps weights and biases constant for the previous known classes
		
		# Get the existing weights and bias from the last layer
		existing_weights = self.classifier.output.weight.data
		existing_bias = self.classifier.output.bias.data
		
		# Create a new layer with out_features increased by 1
		new_out_features = self.n_class  # Increases the number of output features by number of new classes
		new_layer = nn.Linear(in_features=200, out_features=new_out_features)
		
		# Initialize the new layer weights and bias with existing values and add zeros for the new node
		new_weights = torch.zeros((new_out_features, 200))
		new_bias = torch.zeros((new_out_features))
		
		# Copy the existing weights and bias to the new weights and bias
		new_weights[:existing_weights.size(0), :existing_weights.size(1)] = existing_weights
		new_bias[:existing_bias.size(0)] = existing_bias
		
		# Assign the new weights and bias to the new layer
		new_layer.weight.data = new_weights
		new_layer.bias.data = new_bias
		print(new_layer)
		return new_layer

	
	def update_representation(self, new_dataset, num_epochs = 50, new_class=1, learning_rate=1e-5, temperature=2, distil_gamma = 0.5, lambda_param=0.5):
		''' 
		Combine new data with exemplars from memory
		new_dataset:       dataset with new data
		new_class_id:   class ID of the new dataset
		'''


		def kl_divergence_with_temperature(teacher_logits, student_logits, temperature):
			"""
			Calculate the KL divergence with temperature scaling between teacher and student logits.
		
			Parameters:
			- teacher_logits: Logits from the teacher model (torch.Tensor).
			- student_logits: Logits from the student model (torch.Tensor).
			- temperature: Temperature parameter to soften the probability distributions (float).
		
			Returns:
			- KL divergence loss (torch.Tensor).
			"""
			# Soften the probabilities of both teacher and student outputs
			teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
			# student_probs = F.softmax(student_logits / temperature, dim=-1)
			# Calculate the log probabilities of student outputs
			student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
			# Calculate the KL divergence
			kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
			
			return kl_div

		exemplar_dataset = ExemplarDataset(self.exemplar_sets)  # This creates dataset from dictionary exemplar_sets
		combined_dataset = ConcatDataset([new_dataset, exemplar_dataset]) 

		combined_dataloader = DataLoader(combined_dataset, batch_size=32, num_workers=2, shuffle=True, pin_memory=True if device == 'cuda' else False)
		
		########################################################
		# First, get ouput logits for all combined dataset. Store them for distillation loss.
		self.n_class+=new_class

		self.classifier.output = self.expanded_network()
		# print(self.feature_extractor)
		
		self.classifier.to(device)
		self.classifier.eval()
		
		q = torch.zeros(len(combined_dataset), self.n_class).cuda()

		print("Getting logits for teacher model")
		with torch.no_grad():
			for indices, images, labels in tqdm(combined_dataloader):
				indices = indices.cuda()
				images = images.cuda()
				g = F.sigmoid(self.classifier(images.to(device)))
				q[indices] = g.data
			
			q = q.cuda()

		
		########################################################
		# Now, train the model to update representation of classifier 

		# Define optimizers and loss functions here
		optimizer = optim.SGD(self.classifier.parameters(), lr=learning_rate, weight_decay=0.00001)
		classification_loss = nn.CrossEntropyLoss()
		bce_loss = nn.BCEWithLogitsLoss()
		
		self.classifier.train()
		
		print("Training model with classification, distillation and BL(BCE) loss")
		for epoch in tqdm(range(num_epochs)):
			for i, (indices, images, labels) in enumerate(combined_dataloader):
				images = images.to(device)
				labels = labels.to(device)
				optimizer.zero_grad()
				g = self.classifier(images)
				
				# Classification loss
				cl_loss = classification_loss(g, labels)

				# Distillation loss
				g_sig = F.sigmoid(g)
				q_i = q[indices]
				distil_loss = kl_divergence_with_temperature(q_i, g_sig, temperature=temperature)
				
				#dist_loss = dist_loss / self.n_known
				icarl_loss = (1-distil_gamma) * cl_loss + distil_gamma * distil_loss

				#############################################################################################
				# Get BCE. Logits for 0 is still binary class 0 but logits for other generators are 1
				binary_labels = (labels != 0).float().unsqueeze(1)
				log_prob = torch.nn.functional.log_softmax(g, dim=1)
				mask = binary_labels.expand_as(log_prob)

				log_prob_selected = log_prob * mask
				dG = log_prob_selected[:,1:].sum(dim=1, keepdim=True) # This calculates dG
				
				# Since for binary_labels == 0 we want to consider only log_probs[:, 0]
				# we need to subtract the summed log probabilities from log_probs[:, 0] when binary_labels == 0
				binary_labels_complement = 1 - binary_labels  # Inverts 0s and 1s
				dR = log_prob[:, 0].unsqueeze(1) * binary_labels_complement

				# Combine the two parts to get the final loss tensor. As per the paper
				bl_loss = -(dR + dG).mean()

				if torch.isnan(bl_loss).any():
					print("Loss is NaN. Inspecting dG and binary_labels...")
					print("log_probs:", log_prob)
					print("Binary Labels:", binary_labels)
					print("Bl Loss:", bl_loss.item())
					print("iCaRL Loss:", icarl_loss.item())
					sys.exit("Stopping training due to NaN loss.")
				#############################################################################################
					

				# Loss is combination of icarl and bce loss
				loss = icarl_loss + lambda_param * bl_loss

				loss.backward()
				optimizer.step()

				if (i+1) % 5 == 0:
					print ('Epoch [%d/%d], Iter [%d/%d], Total Loss: %.4f iCaRL Loss: %.4f BCE Loss: %.4f' 
						   %(epoch+1, num_epochs, i+1, len(combined_dataset)//32, loss.item(), icarl_loss.item(), bl_loss.item()))
				
				
		# # Transfer classifier  to eval mode
		self.classifier.eval()
		
		# Set this flag to True, since exemplar mean needs to be recalculated
		self.compute_exemplar_means = True
		
