
# Bi-directional Wasserstein TiFGAN for representation learning

This repository contains work done for my Master's thesis in the Machine Learning program in KTH. The work is based on and extends the TiFGAN architecture available at https://github.com/tifgan/stftGAN. The topic of the thesis was unsupervised representation learning and the final title was ["Instability of a bi-directional TiFGAN in unsupervised speech representation learning"](http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1594701&dswid=-6412). 

The TiFGAN architecture is extended to a bi-directional Wasserstein GAN and then evaluated in representation learning. This repository contains code with instructions for the following:
1. Downloading the data
2. Data pre-processing
3. BiTiFGAN training
4. Evaluation of learned representations over training time using a Logistic Regression or Random Forest model
5. Training and evaluation with baseline feature sets such as MFCC, FBANK or TiFGAN spectrogram features.
6. Generating samples using a TiFGAN or BiTiFGAN generator
7. Evaluate BiTiFGAN latent variable mean absolute reconstruction error over time
8. Evaluate BiTiFGAN spectrogram mean absolute reconstruction error over time
9. Extract and evaluate TiFGAN discriminator features for a dataset
10. Training a post-hoc encoder
11. Extracting and evaluating features using the post-hoc encoder
12. Extracting features for BiTiFGAN encoder
13. Qualitative evaluation of BiTiFGAN or TiFGAN post-hoc encoder visualizing reconstructions

### 1. Download data

You can download the SpeechCommands dataset from [https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) and the CREMA-D dataset from [https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D).

### 2. Data pre-processing

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
  
  
### 3. BiTiFGAN training

**SpeechCommands training**:

	
	python -m training.64md_8k --dataset-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_training/input_data --results-path=/media/datastore/c-matsty-data/checkpoints_summaries/bitifgan_normal_training_final_round
	
	
### 4. Evaluation of learned representations over training time using a Logistic Regression or Random Forest model

We load each checkpoint from training and for each checkpoint we extract features, train a classifier and test it on a given test set. In this way we evaluate the features learned by the GAN over its training.

**SpeechCommands evaluation**:

	python -m feature_evaluation.evaluate_over_time --train-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_training --test-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_test --checkpoints-dir=/media/datastore/c-matsty-data/checkpoints_summaries/bitifgan_normal_training_final_round/commands_md64_8k_checkpoints/ --evaluation-model="RandomForest" --save-dir="../normal_bitifgan_eval_res_rf"
	
An alternative is to hold out part of the training set as a validation set and not use a separate test set for evaluation. To do that you must not pass the --test-path parameter and instead you have to define the --holdout-fraction parameter which determines the percentage of actors that will be encompassed in the validation set. You can do this in the following way:

	python -m feature_evaluation.evaluate_over_time --train-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_training --holdout-fraction=0.1 --checkpoints-dir=/media/datastore/c-matsty-data/checkpoints_summaries/bitifgan_normal_training_final_round/commands_md64_8k_checkpoints/ --evaluation-model="RandomForest" --save-dir="../normal_bitifgan_eval_res_rf_holdoutCV"
	

	
### 5. Training and evaluation with baseline feature sets such as MFCC, FBANK or TiFGAN spectrogram features.

**CREMA-D Benchmarks**:

1. Generate MFCC and FBANK training features:
  
        python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV_training/ --results-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_Preproc_1_training_FBANK --dataset-name=CREMA-D --preproc-type=1 --features-type="fbank"
        python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV_training/ --results-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_Preproc_1_training_MFCC --dataset-name=CREMA-D --preproc-type=1 --features-type="mfcc"
 2. Generate MFCC and FBANK test features:
                
         python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV_test/ --results-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_Preproc_1_test_FBANK --dataset-name=CREMA-D --preproc-type=1 --features-type="fbank" 
         python -m preprocessing.preprocess_dataset --dataset-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_AudioWAV_test/ --results-path=/media/datastore/c-matsty-data/datasets/CREMA-D/CREMA-D_Preproc_1_test_MFCC --dataset-name=CREMA-D --preproc-type=1 --features-type="mfcc" 

3. Run the crema_d_mfcc_fbank_benchmark notebook under notebooks/benchmarks


	
### 6. Generating samples using a TiFGAN or BiTiFGAN generator

	python -m sample_generation.generate --results-dir=/media/datastore/c-matsty-data/checkpoints_summaries/bitifgan-results-sc09-run2-gp/ --checkpoint-step=80000 --bidirectional

	
### 7. Evaluate BiTiFGAN latent variable mean absolute reconstruction error over time

1. Normal reconstruction error:

        python -m feature_evaluation.latent_reconstruction_test  --checkpoint-dir=/media/datastore/c-matsty-data/checkpoints_summaries/bitifgan-results-sc09-run2-gp/commands_md64_8k_checkpoints/

2. Smoothness test (Reconstruction error with an epsilon term added to the latent samples before feeding them to the generator)

        python -m feature_evaluation.latent_reconstruction_test  --checkpoint-dir=/media/datastore/c-matsty-data/checkpoints_summaries/bitifgan-results-sc09-run2-gp/commands_md64_8k_checkpoints/ --epsilon=1e-5
	
### 8. Evaluate BiTiFGAN spectrogram mean absolute reconstruction error over time

1. Normal reconstruction error:

        python -m feature_evaluation.spectrogram_reconstruction_test  --checkpoint-dir=/media/datastore/c-matsty-data/checkpoints_summaries/bitifgan-results-sc09-run2-gp/commands_md64_8k_checkpoints/ --dataset-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_test/input_data/

2. Smoothness test (Reconstruction error with an epsilon term added to the latent samples before feeding them to the generator)

        python -m feature_evaluation.spectrogram_reconstruction_test  --checkpoint-dir=/media/datastore/c-matsty-data/checkpoints_summaries/bitifgan-results-sc09-run2-gp/commands_md64_8k_checkpoints/ --epsilon=1e-5 --dataset-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_test/input_data/
	
### 9. Extract and evaluate TiFGAN discriminator features for a dataset


1. Extract features:

	In case you want to apply global average pooling on the extracted features, you can use the --pooling flag argument.
  
        python -m feature_extraction.extract_discriminator_features --dataset-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_training/input_data/ --checkpoint-step=99000 --results-dir="/media/datastore/c-matsty-data/checkpoints_summaries/tifgan_spectralnorm_sc09/" --features-path="../saved_discriminator_features.npy" --selected-layer=4

2. Evaluate features by running the evaluate_discriminator_features_sc09 notebook under notebooks/feature_evaluation	
 

### 10. Train post-hoc encoder

    python -m training.train_post_hoc_encoder --checkpoint-step=99000 --results-dir=/media/datastore/c-matsty-data/checkpoints_summaries/tifgan_spectralnorm_sc09/ --encoder-path=/media/datastore/c-matsty-data/checkpoints_summaries/post_hoc_tifgan_encoder


### 11. Extract and evaluate post hoc encoder features for a dataset


1. Extract features:
  
        python -m feature_extraction.extract_posthoc_encoder_features --dataset-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_training/input_data/ --checkpoint-path=/media/datastore/c-matsty-data/checkpoints_summaries/post_hoc_tifgan_encoder/commands_md64_8k_checkpoint_step_499000  --features-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/Posthoc_encoder_features/training.npy

2. Evaluate features by running the evaluate_posthoc_encoder_features notebook under notebooks/feature_evaluation
 
    
    
    
### 12. Extract Bi-TiFGAN encoder features for a dataset
    python -m feature_extraction.extract_features --dataset-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/SpeechCommands_Preproc_2_training/input_data/ --checkpoint-step=70000 --results-dir=/media/datastore/c-matsty-data/checkpoints_summaries/bitifgan_normal_training_final_round/ --features-path=/media/datastore/c-matsty-data/datasets/SpeechCommands/Bitifgan_features/training.npy

### 13. Qualitative evaluation of BiTiFGAN or TiFGAN post-hoc encoder visualizing reconstructions


1. Evaluate BiTiFGAN encoder features by running the bitifgan_encoder_qualitative_eval notebook under notebooks/feature_evaluation	
2. Evaluate TiFGAN post-hoc encoder features by running the posthoc_encoder_qualitative_eval notebook under notebooks/feature_evaluation	
 
 
