
# Bi-directional Wasserstein TiFGAN for representation learning

### Download data

You can download the SpeechCommands dataset from [https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) and the CREMA-D dataset from [https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D).

### Preprocessing steps

1. **Split the dataset in training and test dataset using a text file with all the test files**
 
    CREMA-D splitting:
	```
	python -m preprocessing.split_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV/ --test-list-path=data/cremad_testing_list.txt
	```
2.  **Convert training and test set from wav files to TiFGAN features**
 
    CREMA-D preprocessing:
	```
	python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV_training/ --results-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_Preproc1_training --dataset-name="CREMA-D"  --preproc-type=1
	python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV_test/ --results-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_Preproc1_test --dataset-name="CREMA-D"  --preproc-type=1
	```
  
### Training the GAN

**SpeechCommands training**:
	```
	python -m training.64md_8k --dataset-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_training/input_data --results-path=/media/datastore/c-matsty-data/checkpoints_summaries/<save_dir_name>
	```
