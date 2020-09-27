
# Bi-directional Wasserstein TiFGAN for representation learning

### Download data

You can download the SpeechCommands dataset from [https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) and the CREMA-D dataset from [https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D).

### 1. Preprocessing steps

1. **Split the dataset in training and test dataset using a text file with all the test files**
 
    CREMA-D splitting:
	```
	python -m preprocessing.split_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV/ --test-list-path=data/cremad_testing_list.txt
	```
2.  **Convert training and test set from wav files to TiFGAN features**
 
    SpeechCommands preprocessing:
	```
	python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_AudioWAV_training/ --results-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_training --dataset-name="SpeechCommands"  --preproc-type=2
	python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_AudioWAV_test/ --results-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_test --dataset-name="SpeechCommands"  --preproc-type=2
	```
	
    CREMA-D preprocessing:
	```
	python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV_training/ --results-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_Preproc1_training --dataset-name="CREMA-D"  --preproc-type=1
	python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV_test/ --results-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_Preproc1_test --dataset-name="CREMA-D"  --preproc-type=1
	```
  
  
### 2. Training the GAN

**SpeechCommands training**:

	
	python -m training.64md_8k --dataset-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_training/input_data --results-path=/media/datastore/c-matsty-data/checkpoints_summaries/<save_dir_name>
	
	
### 3. Evaluating learned features over time

We load each checkpoint from training and for each checkpoint we extract features, train a classifier and test it on a given test set. In this way we evaluate the features learned by the GAN over its training.

**SpeechCommands evaluation**:

	python -m feature_evaluation.evaluate_over_time --train-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_training --test-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_test --checkpoints-dir=/media/datastore/c-matsty-data/checkpoints_summaries/<save_dir_name> --evaluation-model="RandomForest"
	

	
### 3. Running the benchmarks

**CREMA-D Benchmarks**:

1. Generate MFCC and FBANK training features:
  
        python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV_training/ --results-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_Preproc_1_training_FBANK --dataset-name=CREMA-D --preproc-type=1 --features-type="fbank"
        python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV_training/ --results-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_Preproc_1_training_MFCC --dataset-name=CREMA-D --preproc-type=1 --features-type="mfcc"
 2. Generate MFCC and FBANK test features:
                
         python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV_test/ --results-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_Preproc_1_test_FBANK --dataset-name=CREMA-D --preproc-type=1 --features-type="fbank" 
         python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV_test/ --results-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_Preproc_1_test_MFCC --dataset-name=CREMA-D --preproc-type=1 --features-type="mfcc" 

3. Run the crema_d_mfcc_fbank_benchmark notebook under notebooks/benchmarks


	
### 4. Generating samples

	python -m sample_generation.generate --results-dir=/media/datastore/c-matsty-data/checkpoints_summaries/bitifgan-results-sc09-run2-gp/ --checkpoint-step=80000 --bidirectional

	
### 5. Evaluate mean absolute reconstruction error over time

1. Normal reconstruction error:

        python -m feature_evaluation.reconstruction_test  --checkpoint-dir=/media/datastore/c-matsty-data/checkpoints_summaries/bitifgan-results-sc09-run2-gp/commands_md64_8k_checkpoints/

2. Smoothness test (Reconstruction error with an epsilon term added to the latent samples before feeding them to the generator)

        python -m feature_evaluation.reconstruction_test  --checkpoint-dir=/media/datastore/c-matsty-data/checkpoints_summaries/bitifgan-results-sc09-run2-gp/commands_md64_8k_checkpoints/ --epsilon=1e-5
	
### 6. Extract TiFGAN discriminator features for a dataset

    python -m feature_extraction.extract_discriminator_features --dataset-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_training/input_data/ --checkpoint-step=99000 --results-dir="/media/datastore/c-matsty-data/checkpoints_summaries/tifgan_spectralnorm_sc09/" --features-path="../saved_discriminator_features.npy" 


### 7. Train post-hoc encoder

    python -m training.train_post_hoc_encoder --checkpoint-step=99000 --results-dir=/media/datastore/c-matsty-data/checkpoints_summaries/tifgan_spectralnorm_sc09/ --encoder-path=/media/datastore/c-matsty-data/checkpoints_summaries/post_hoc_tifgan_encoder


### 8. Extract post hoc encoder features for a dataset
    python -m feature_extraction.extract_posthoc_encoder_features --dataset-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_training/input_data/ --checkpoint-path=/media/datastore/c-matsty-data/checkpoints_summaries/post_hoc_tifgan_encoder/commands_md64_8k_checkpoint_step_199000 --features-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/Posthoc_encoderclear_features/training.npy
