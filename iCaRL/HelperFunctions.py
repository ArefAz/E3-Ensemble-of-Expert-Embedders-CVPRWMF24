import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # For the progress bar
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classification_report_auroc(dataset_t, model):
	"""
	Returns classification report and AUROC
	"""
	dataloader_t = DataLoader(dataset_t, batch_size=32, num_workers=4, shuffle=True)
	result, prediction_score, truth = [], [], []

	for _, img, label in tqdm(dataloader_t, desc="Evaluating"):

		prediction, softmax = model.classify(img.to(device))
		truth.extend(label)

		prediction_score.extend(np.sum(softmax[:,1:], axis=1).tolist())
		result.extend(prediction)

	result = [t.item() for t in result]
	truth = [t.item() for t in truth]
	prediction_score = [t for t in prediction_score]

	# For AUC-ROC, convert to binary classifier
	binary_truth = [1 if x != 0 else 0 for x in truth]
	return accuracy_score(truth, result), roc_auc_score(binary_truth, prediction_score)