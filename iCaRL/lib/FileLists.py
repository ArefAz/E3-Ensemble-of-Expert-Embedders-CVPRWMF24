lightning_checkpoint_path='/media/nas2/Aref/share/continual_learning/models/mislnet/epoch=97-step=183750-v_loss=0.0654-v_acc=0.9763.ckpt'

resnet50_lightning_checkpoint_path='/media/nas2/Aref/share/continual_learning/models/resnet50/epoch=25-step=12194-v_loss=0.1221-v_acc=0.9588.ckpt'

# real_exemplars='checkpoints/exemplar-set-real-mislnet.pt'
# gan_exemplars='checkpoints/exemplar-set-gan-mislnet.pt'

# real_exemplars_resnet50='checkpoints/exemplar-set-real-resnet50.pt'
# gan_exemplars_resnet50='checkpoints/exemplar-set-gan-resnet50.pt'

train_real_file_paths='/media/nas2/Aref/share/continual_learning/final_dataset_paths/db-real/train.txt'
train_gan_file_paths='/media/nas2/Aref/share/continual_learning/final_dataset_paths/db-gan/train.txt'

test_file_paths_real='/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-real/test.txt'

generator_names = 	['2-SD1.4', '3-GLIDE', '4-Mijourney', '5-DALLE-Mini', '6-TT', '7-SD2.1', '8-CIPS', '9-Biggan', '10-VQ-Diff', '11-Diff-gan', '12-SG3', '13-Gansformer', '14-DALLE-2', '15-LD', '16-EG3D', '17-ProjGan', '18-SD1', '19-DDG', '20-DDPM']
test_sets = ['1-GAN', '2-SD1.4', '3-GLIDE', '4-Mijourney', '5-DALLE-Mini', '6-TT', '7-SD2.1', '8-CIPS', '9-Biggan', '10-VQ-Diff', '11-Diff-gan', '12-SG3', '13-Gansformer', '14-DALLE-2', '15-LD', '16-EG3D', '17-ProjGan', '18-SD1', '19-DDG', '20-DDPM']

generators_file_path = ['/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-sd14/'			# Task 2
			 , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-glide/'				# Task 3
			 , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-mj/'					# Task 4
			 , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-dallemini/'				# Task 5
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-tt/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-sd21/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-cips/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-biggan/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-vqdiff/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-diffgan/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-sg3/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-gansformer/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-dalle2/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-ld/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-eg3d/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-projgan/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-sd1/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-ddg/'
             , '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-ddpm/'
			 ]

test_file_paths = ['/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-gan/test.txt'	
			 	, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-sd14/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-glide/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-mj/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-dallemini/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-tt/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-sd21/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-cips/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-biggan/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-vqdiff/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-diffgan/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-sg3/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-gansformer/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-dalle2/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-ld/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-eg3d/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-projgan/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-sd1/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-ddg/test.txt'
				, '/media/nas2/Aref/share/continual_learning/final_dataset_paths/dn-ddpm/test.txt'
			 ]

