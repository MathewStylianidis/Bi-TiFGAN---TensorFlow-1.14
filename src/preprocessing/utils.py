"""
Utilities module for data preprocessing.
"""

__author__ = "Matthaios Stylianidis"

import os


import ltfatpy
import librosa
import numpy as np
from python_speech_features import mfcc, fbank
from tqdm import tqdm


from signal_processing.mob_gab_phase_grad import modgabphasegrad
from signal_processing.signal_wrapper import SignalWrapper
from signal_processing.ltfat_stft import LTFATStft

CROP_OR_PAD_CODE = "crop_or_pad"
CHUNK_CODE = "chunk"
HIGH_ENERGY_CODE = "high-energy"
dataset_names = ["CREMA-D", "SpeechCommands"]


def get_cremad_file_meta_data(file_name):
    "Gets the metadata from the file_name, such as actor id or label"""
    fields = file_name.split('_')
    return fields[2], fields[0]


def get_speech_commands_file_meta_data(file_name):
    "Gets the metadata from the file_name, such as actor id or the repeated command id"""
    fields = os.path.basename(file_name).split("_")
    return fields[0], fields[2]


def filelines_to_list(file_path):
    """ Reads the lines of a file as a list.

    Args:
        test_list_path (str): Path to the file.

    Returns:
        The list with the lines in the file.
    """
    with open(file_path, "r") as fp:
        lines = fp.readlines()
    return lines


def read_dataset(dataset_name, dataset_path, preproc_type, fixed_len, chunk_overlap):
    """ Reads a dataset from the hard disk into numpy arrays.

    Args:
        dataset_name (str) The name of the dataset to read.
        dataset_path (str): The path to the directory with the dataset.
        preproc_type (str): Code for the type of the preprocessing to be used.
        fixed_len (int): Size of the preprocessed samples.

    Raises:
        Exception: If dataset_name does not match any dataset name defined in the module level list <dataset_names>.

    Returns:
        A 4 element tuple with three numpy arrays: i) The preprocessed input data, ii) Their labels, iii) The actors
            corresponding to each utterance and iv) The sampling rate of the utterances, assuming all of them
            have the same sampling rate.
    """
    if dataset_name == dataset_names[0]:
        return read_crema_dataset(dataset_path, preproc_type, fixed_len, chunk_overlap)
    elif dataset_name == dataset_names[1]:
        return read_speech_commands_dataset(dataset_path, preproc_type, fixed_len, chunk_overlap)
    else:
        raise Exception("Dataset {} is not a valid dataset name.".format(dataset_name))


def add_fixed_length_waveforms(X, signal, preproc_type, fixed_len, chunk_overlap=0):
    """ Extracts one or more fixed length signals from <signal> and adds the results to the given list.

    Args:
        X (list): The list with fixed length signals.
        signal (signal_processing.signal_wrapper.SignalWrapper): The SignalWrapper object with the signal that
            will be turned into one or more fixed length signals.
        preproc_type (str): Defines the technique to be used in order to extract one or more fixed length
            signals.
        fixed_len (int): The fixed length of the extracted signals.
        chunk_overlap (int): The overlap between chunks in case the input signal is chunked into multiple
            signal chunks.

    Returns:
        The number of waveforms added to the list.
    """
    if preproc_type == "crop_or_pad":
        signal.pad_or_crop(fixed_len, in_place=True)  # Pads or crops signal on the sides
        X.append(signal.get_signal())
        return 1
    elif preproc_type == "chunk":
        chunks = signal.chunk(chunk_length=fixed_len, chunk_overlap=chunk_overlap)
        X.extend(chunks)
        return len(chunks)
    elif preproc_type == "high-energy":
        X.append(signal.get_higher_energy_window(window_size=fixed_len))
        return 1


def read_crema_dataset(dataset_path, preproc_type, fixed_len, chunk_overlap=0):
    """ Reads the CREMA-D dataset from the hard disk into numpy arrays.

    Args:
        dataset_path (str): The path to the directory with the wav files.
        preproc_type (str): Code for the type of the preprocessing to be used.
        fixed_len (int): Size of the preprocessed samples.
        chunk_overlap (int): Overlap between chunks when samples are broken down to chunks.

    Returns:
        A 4 element tuple with three numpy arrays: i) The preprocessed input data, ii) Their labels, iii) The actors
            corresponding to each utterance and iv) The sampling rate of the utterances, assuming all of them
            have the same sampling rate.
    """
    X = []
    y = []
    actors = []
    print("Converting wav files to fixed size numpy arrays and creating dataset...")
    for file_name in tqdm(os.listdir(dataset_path)):
        if not file_name.endswith('.wav'):
            continue

        full_file_path = os.path.join(dataset_path, file_name)
        audio, sampling_rate = librosa.load(full_file_path, sr=None, dtype=np.float64)

        signal = SignalWrapper(audio)
        if not signal.meets_amplitude_requirement(1e-4):
            tqdm.write("Minimum amplitude requirement not met - File name: {}".format(file_name))
            continue

        no_waveforms = add_fixed_length_waveforms(X, signal, preproc_type, fixed_len, chunk_overlap)

        emotion_label, actor = get_cremad_file_meta_data(file_name)
        y.extend([emotion_label for _ in range(no_waveforms)])
        actors.extend([actor for _ in range(no_waveforms)])

    X = np.stack(X)
    y = np.stack(y)
    actors = np.stack(actors)

    return X, y, actors, sampling_rate


def read_speech_commands_dataset(dataset_path, preproc_type, fixed_len, chunk_overlap=0):
    """ Reads the SpeechCommands dataset from the hard disk into numpy arrays.

    Args:
        dataset_path (str): The path to the directory with the SpeechCommands dataset.
        preproc_type (str): Code for the type of the preprocessing to be used.
        fixed_len (int): Size of the preprocessed samples.
        chunk_overlap (int): Overlap between chunks when samples are broken down to chunks.

    Returns:
        A 4 element tuple with three numpy arrays: i) The preprocessed input data, ii) Their labels, iii) The actors
            corresponding to each utterance and iv) The sampling rate of the utterances, assuming all of them
            have the same sampling rate.
    """
    X = []
    y = []
    actors = []
    print("Converting wav files to fixed size numpy arrays and creating dataset...")
    for class_dir_name in tqdm(os.listdir(dataset_path)):
        class_dir_path = os.path.join(dataset_path, class_dir_name)
        for file_name in tqdm(os.listdir(class_dir_path)):
            if not file_name.endswith('.wav'):
                continue
            full_file_path = os.path.join(class_dir_path, file_name)
            audio, sampling_rate = librosa.load(full_file_path, sr=None, dtype=np.float64)
            signal = SignalWrapper(audio)
            if not signal.meets_amplitude_requirement(1e-4):
                tqdm.write("Minimum amplitude requirement not met - File name: {}".format(file_name))
                continue

            no_waveforms = add_fixed_length_waveforms(X, signal, preproc_type, fixed_len, chunk_overlap)
            
            actor, repeated_id = get_speech_commands_file_meta_data(file_name)
            y.extend([emotion_label for _ in range(no_waveforms)])
            actors.extend([actor for _ in range(no_waveforms)])

    X = np.stack(X)
    y = np.stack(y)
    actors = np.stack(actors)
    return X, y, actors, sampling_rate


def exctract_mfcc_features(X, y, actors, fft_window_length, fft_hop_size, sampling_rate, batch_size,
                           input_dir_path, labels_dir_path, actors_dir_path, preproc_filename,
                           numcep=20, nfilt=40, window=np.hamming, fbank_feats=False):
    """ Extracts and saves the MFCC features from the given audio vectors.

    Args:
        X: A 2D numpy array with the audio vectors.
        y: The labels for the audio vectors in X.
        actors: The actor identifiers for the audio vectors in X.
        fft_window_length: The window length to be used for the FFT.
        fft_hop_size: The hop size to be used for the FTT.
        sampling_rate: The sampling rate of the audio vectors in X.
        batch_size: The batch size for splitting the features into different files.
        input_dir_path: The directory where the features should be saved.
        labels_dir_path: The directory where the feature labels should be saved.
        actors_dir_path: The directory where the actor ids should be saved.
        preproc_filename: The file name prefix to be used for the saved files.
        numcep: The number of MFCC coefficients to extract from the audio vectors.
        nfilt: The number of filter banks (FBANKs) to use for extracting MFCCs.
        window: The type of window to use, if any, during the windowing stage (e.g. Hamming window).
        fbank_feats: Set to True if you only want to extract the mel-filterbank energy (FBANK) features
            instead of MFCC features.
    Returns:
        None
    """
    N = int(sampling_rate / fft_hop_size)  # Number of frames for each audio
    fft_window_length = fft_window_length / sampling_rate  # Express window length in seconds
    fft_hop_length = fft_hop_size / sampling_rate  # Express hop size in seconds

    y_batch = []
    actors_batch = []
    if not fbank_feats:
        spectrograms = np.zeros([batch_size, N, numcep], dtype=np.float64)
    else:
        spectrograms = np.zeros([batch_size, N, nfilt], dtype=np.float64)
    for index, x in tqdm(enumerate(X)):
        if not fbank_feats:
            spectrogram = mfcc(x, samplerate=sampling_rate, winlen=fft_window_length, winstep=fft_hop_length,
                               numcep=numcep, nfilt=nfilt, winfunc=window)
        else:
            spectrogram, energy = fbank(x, samplerate=sampling_rate, winlen=fft_window_length, winstep=fft_hop_length,
                                   nfilt=nfilt, winfunc=window)
        spectrograms[index % batch_size] = spectrogram
        y_batch.append(y[index])
        actors_batch.append(actors[index])

        if index % batch_size == batch_size - 1:
            try:
                batch_no = int(index / batch_size)
                file_path = os.path.join(input_dir_path, preproc_filename) + "_" + str(batch_no) + ".npy"
                label_file_path = os.path.join(labels_dir_path, preproc_filename) + "_" + str(batch_no) + "_labels.npy"
                actors_file_path = os.path.join(actors_dir_path, preproc_filename) + "_" + str(batch_no) + "_actors.npy"
                np.save(file_path, spectrograms)  # Save input data
                np.save(label_file_path, np.stack(y_batch))  # Save labels
                np.save(actors_file_path, np.stack(actors_batch))  # Save actors
                y_batch = []
                actors_batch = []
                tqdm.write("Batch no. " + str(batch_no) + " saved at " + file_path + ".")
            except Exception as e:
                print(e)

    # Save last batch
    last_batch_samples = index % batch_size + 1
    if last_batch_samples > 0:
        try:
            batch_no = int(index / batch_size)

            spectrograms = spectrograms[:last_batch_samples]
            file_path = os.path.join(input_dir_path, preproc_filename) + "_" + str(batch_no) + ".npy"
            label_file_path = os.path.join(labels_dir_path, preproc_filename) + "_" + str(batch_no) + "_labels.npy"
            actors_file_path = os.path.join(actors_dir_path, preproc_filename) + "_" + str(batch_no) + "_actors.npy"
            np.save(file_path, spectrograms)  # Save input data
            np.save(label_file_path, np.stack(y_batch))  # Save labels
            np.save(actors_file_path, np.stack(actors_batch))  # Save actors
            tqdm.write("Batch no. " + str(batch_no) + " saved at " + file_path + ".")
        except Exception as e:
            print(e)


def extract_tifgan_features(X, y, actors, fft_hop_size, fft_window_length, batch_size, input_dir_path,
                            labels_dir_path, actors_dir_path, preproc_filename):
    """ Extracts and saves the TiF-GAN features from the given audio vectors.

    Args:
        X: A 2D numpy array with the audio vectors.
        y: The labels for the audio vectors in X.
        actors: The actor identifiers for the audio vectors in X.
        fft_hop_size: The hop size to be used for the FTT.
        fft_window_length: The window length to be used for the FFT.
        batch_size: The batch size for splitting the features into different files.
        input_dir_path: The directory where the features should be saved.
        labels_dir_path: The directory where the feature labels should be saved.
        actors_dir_path: The directory where the actor ids should be saved.
        preproc_filename: The file name prefix to be used for the saved files.

    Returns:
        None.
    """
    clip_below = -10
    ltfatpy.gabphasegrad = modgabphasegrad
    anStftWrapper = LTFATStft()
    N = int(X.shape[1] / fft_hop_size)
    spectrograms = np.zeros([batch_size, int(fft_window_length // 2 + 1), N], dtype=np.float64)
    tgrads = np.zeros([batch_size, int(fft_window_length // 2 + 1), N], dtype=np.float64)
    fgrads = np.zeros([batch_size, int(fft_window_length // 2 + 1), N], dtype=np.float64)
    y_batch = []
    actors_batch = []

    for index, x in tqdm(enumerate(X)):
        realDGT = anStftWrapper.oneSidedStft(signal=x, windowLength=fft_window_length, hopSize=fft_hop_size)
        spectrogram = anStftWrapper.logMagFromRealDGT(realDGT, clipBelow=np.e ** clip_below, normalize=True)
        spectrograms[index % batch_size] = spectrogram
        tgradreal, fgradreal = ltfatpy.gabphasegrad('phase', np.angle(realDGT), fft_hop_size,
                                                    fft_window_length)
        tgrads[index % batch_size] = tgradreal[:, :N] / 64
        fgrads[index % batch_size] = fgradreal[:, :N] / 256
        y_batch.append(y[index])
        actors_batch.append(actors[index])
        if index % batch_size == batch_size - 1:
            try:
                batch_no = int(index / batch_size)
                # Save input data
                file_path = os.path.join(input_dir_path, preproc_filename) + "_" + str(batch_no) + ".npz"
                shifted_spectrograms = spectrograms / (-clip_below / 2) + 1
                np.savez_compressed(file_path, logspecs=shifted_spectrograms, tgrad=tgrads, fgrad=fgrads)
                # Save labels
                label_file_path = os.path.join(labels_dir_path, preproc_filename) + "_" + str(batch_no) + "_labels.npy"
                np.save(label_file_path, np.stack(y_batch))
                # Save actor ids
                actors_file_path = os.path.join(actors_dir_path, preproc_filename) + "_" + str(batch_no) + "_actors.npy"
                np.save(actors_file_path, np.stack(actors_batch))
                y_batch = []
                actors_batch = []
                tqdm.write("Batch no. " + str(batch_no) + " saved at " + file_path + ".")
            except Exception as e:
                print(e)

    # Save last batch
    last_batch_samples = index % batch_size + 1
    if last_batch_samples > 0:
        try:
            batch_no = int(index / batch_size)

            spectrograms = spectrograms[:last_batch_samples]
            tgrads = tgrads[:last_batch_samples]
            fgrads = fgrads[:last_batch_samples]
            # Save input data
            file_path = os.path.join(input_dir_path, preproc_filename) + "_" \
                        + str(batch_no) + ".npz"
            shifted_spectrograms = spectrograms / (-clip_below / 2) + 1
            np.savez_compressed(file_path, logspecs=shifted_spectrograms, tgrad=tgrads, fgrad=fgrads)
            # Save labels
            label_file_path = os.path.join(labels_dir_path, preproc_filename) + "_" + str(batch_no) + "_labels.npy"
            np.save(label_file_path, np.stack(y_batch))
            # Save actor ids
            actors_file_path = os.path.join(actors_dir_path, preproc_filename) + "_" + str(batch_no) + "_actors.npy"
            np.save(actors_file_path, np.stack(actors_batch))
            tqdm.write("Batch no. " + str(batch_no) + " saved at " + file_path + ".")
        except Exception as e:
            print(e)
