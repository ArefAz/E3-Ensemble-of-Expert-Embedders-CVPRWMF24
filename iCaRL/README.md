# iCaRL Methods Implementation README

Welcome to the implementation guide for iCaRL-based methods. This README will guide you through running scripts for different continual learning frameworks focusing on iCaRL, MTMC (Multi Task Multi Classifier), and MTSC (Multi Task Single Classifier).

## Reference Papers

- **iCaRL** implementation is based on the paper by Rebuffi SA, Kolesnikov A, Sperl G, Lampert CH titled *iCaRL: Incremental Classifier and Representation Learning*. Presented at the IEEE Conference on Computer Vision and Pattern Recognition 2017 (pp. 2001-2010).

- **MTMC** and **MTSC** implementations are derived from the work of F. Marra, C. Saltori, G. Boato, and L. Verdoliva in *Incremental learning for the detection and classification of GAN-generated images*. 2019 IEEE International Workshop on Information Forensics and Security (WIFS), Dec. 2019, pp. 1–6. DOI: 10.1109/WIFS47025.2019.9035099.

## Getting Started

To run the scripts, you must first configure the settings in the `master_config.yaml` file. Below are the configurable parameters:

### Configuration Parameters

- **cl_framework**: Specifies the continual learning framework. For iCaRL-based methods, choose between `icarl`, `mtsc`, or `mtmc`.
  
- **experiment_setup**: Determines the experiment type. Options are:
  1. `adapt_one_new_generator`
  2. `adapt_multiple_new_generators`
  
  These correspond to experiments 1 and 2 as described in our paper.

### Model Parameters

- **model**: Choose between `resnet50` and `mislnet` for the model architecture.
  
- **checkpoint**: Path to the state dict of the trained model (either `resnet50` or `mislnet`) pre-trained on GAN vs. Real data.
  
- **feature_size**: Defines the size of the latent feature produced by the model's penultimate layer.

- **epochs**: Number of epochs to train for new generator's data.

- **lr**: Learning rate for the training process.

## Running the Scripts

Once you have configured your `master_config.yaml` with the appropriate parameters, you can initiate the training process by running the `master_script.py` script. This will start the training, and the results will be outputted both on-screen and saved to a CSV file for further analysis.

```bash
python master_script.py
