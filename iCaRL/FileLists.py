lightning_checkpoint_path='/media/nas2/Aref/share/continual_learning/models/mislnet/epoch=97-step=183750-v_loss=0.0654-v_acc=0.9763.ckpt'

real_exemplars='exemplar-set-real-mislnet.pt'
gan_exemplars='exemplar-set-gan-mislnet.pt'

train_real_file_paths='/media/nas2/Aref/share/continual_learning/final_dataset_paths/db-real/train.txt'
train_gan_file_paths='/media/nas2/Aref/share/continual_learning/final_dataset_paths/db-gan/train.txt'

test_file_paths_real='/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-real/test.txt'

generator_names = 	[ 'TT', 'SD', 'EG3D', 'DALLE2']
test_sets = [ 'GAN', 'TT', 'SD', 'EG3D', 'DALLE2']

generators_file_path = ['/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-tt/'			# Task 2
			 , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-sd1/'				# Task 3
			 , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-eg3d/'					# Task 4
			 , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-dalle2/'				# Task 5
			 ]

test_file_paths = ['/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-gan/test.txt'		# Task 1
			 , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-tt/test.txt'			# Task 2
			 , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-sd1/test.txt'		# Task 3
			 , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-eg3d/test.txt'			# Task 4
			 , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-dalle2/test.txt'		# Task 5
			 ]

