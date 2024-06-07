# E3: Ensemble of Expert Embedders for Adapting Synthetic Image Detectors to New Generators Using Limited Data

This is the official implementation of the CVPRWMF24 paper titled: **[E3: Ensemble of Expert Embedders for Adapting Synthetic Image Detectors to New Generators Using Limited Data](https://arxiv.org/abs/2404.08814)**

**Work is still in progress to imporve the interface for easy proto-typing.**

This codebase (expect for iCaRL, MT-MC, and MT-SC) is based on [Pytorch Lightning](https://lightning.ai/docs/pytorch/.stable/). 
Other continual learning methods are benchmarked using the implementations
 provided by [UDIL](https://github.com/Wang-ML-Lab/unified-continual-learning).

In order to train our model (E3) in a continual learning manner to detect synthetic images from a diverse set of generators, we perform the following steps:

1.  First, we need to train a baseline detector model using a large dataset of GAN-generated and real images. This is done via `python train.py -c configs/configs.yaml`. In the code and configs, we refer to the GAN-generated dataset and the real dataset as `db-gan` and `db-real`, respectively. 
For each dataset, three txt files should be specified in the config file for training, validation, and testing. 
To perform the tests for the baseline detector, you should copy the checkpoint path of the trained model into the expert_ckpt field in the config file. Next, run `python test.py -c configs/configs.yaml`.
You can modify many of the pytorch lightning settings in the same config file.
2. Next, we train the ensemble of expert embedders (E3), each of which specialized to detect forensic traces of only one synthetic image generator.
This is done via `python only_ft.py`. This will create as many expert embedders as there are synthetic image datasets provided in the config file.
Each synthetic dataset should have three txt files, listing the file paths for training, validation, and testing. The format for these files should be like: 
`datasets/dataset_file_paths/dn-{dataset-name}/{train/test/val}.txt`. These data will be combined with real images, referred to as `dn-real` (or `R` as in the paper)
  to create 
a complete fine-tuning dataset for each generator. After all the experts are created, a txt file containing the checkpoint paths of all the embedders
 will be created.
 3. The next step is to use the E3, created in the previous step, to train a new synthetic image detector that is able to detect synthetic images from
 any of the generators in our dataset. First we need to copy the embedders paths into the default config file for this step, 
 which is `configs/cl_configs.yaml`. Then we run `python continual_train.py` to train the Expert Knowledge Fusion Network (EKFN). Make sure the `load_from_ckpt`
 option is set to `True`; otherwise, the code will first train an expert embedder for each step, and then train the EKFN, this is redundant if 
 you have completed step 2. This script will also perform the tests as the training progresses. At the end of the training, two csv files containing 
 the results in terms of accuracy and AUC will be saved to `acc_matrix.csv` and `auc_matrix.csv`. Each row of these matrices represent the performance
 of the model after seeing the generator associated with that row. The numbers reported in the paper are calculated as the average of each row
 up to the generator that the model has seen.
 

 **Notes**:

 1. `rotate.py` script can be used to run the first experiment in the paper (adapting to a single new generator), .
 2. You can find the file names of all the images used in our experiments in `datasets/dataset_file_paths`. Since all of our data is 
 publicly available, we do not provide the data.

 If you use our code, please consider citing us as below, thank you!
 ```@inproceedings{azizpour2024e3,
  title     = {E3: Ensemble of Expert Embedders for Adapting Synthetic Image Detectors to New Generators Using Limited Data},
  author    = {Azizpour, Aref and Nguyen, Tai D and Shrestha, Manil and Xu, Kaidi and Kim, Edward and Stamm, Matthew C},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year      = {2024},
  pages     = {pages},
}
