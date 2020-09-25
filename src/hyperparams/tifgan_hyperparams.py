import tensorflow as tf
import os

from gantools import blocks


def get_hyperparams(result_path=None, name=None, bidirectional=True):
    """ Defines the dictionary of the parameters for the bi-directional TiF-GAN.

        This function creates and returns a dictionary with the parameters for the
        design of the bi-directional TiF-GAN architecture (hyperparameters of for
        each layer) as well as any training hyperparameter (e.g. initializers,
        learning rate, etc.).

    Args:
        result_path (str): The directory where checkpoints and event summaries are saved.
        name (str): The name of the model.
        bidirectional (bool): If set to False then the hyperparameters for a simple TiFGAN
            will be returned rather than a BiTiFGAN.

    Returns:
        The dictionary with the bi-directional tif-gan hyperparameters.
    """
    bn = False
    md = 64
    latent_dim = 100
    input_shape = [256, 128, 1]
    batch_size = 64
    downscale = 1

    # all parameters
    params = dict()
    params['net'] = dict()  # All the parameters for the model
    params['net']['generator'] = get_generator_hyperparams(latent_dim, md, bn)
    params['net']['discriminator'] = get_discriminator_hyperparams(md, bn, bidirectional)
    if bidirectional:
        params['net']['encoder'] = get_encoder_hyperparams(latent_dim, md, bn, input_shape)
    params['net']['prior_distribution'] = 'gaussian'
    params['net']['shape'] = input_shape  # Shape of the image
    params['net']['gamma_gp'] = 10  # Gradient penalty
    params['net']['fs'] = 16000 // downscale

    params['optimization'] = get_optimization_hyperparams(batch_size, bidirectional)
    params['input_shape'] = input_shape
    params['md'] = 64
    params['bn'] = bn
    params['summary_every'] = 100  # Tensorboard summaries every ** iterations
    params['print_every'] = 100  # Console summaries every ** iterations
    params['save_every'] = 2000  # Save the model every ** iterations
    params['summary_dir'] = os.path.join(result_path, name + '_summary/')
    params['save_dir'] = os.path.join(result_path, name + '_checkpoints/')
    params['Nstats'] = 500
    return params


def get_optimization_hyperparams(batch_size, bidirectional):
    """ Gets the optimization hyperparameters for the GAN training.

    Gets a dictionary with the optimization hyperparameters such as the type of the
    optimizer to be used for the discriminator and the generator training as well as
    the hyperparameters for each optimizer such as the learning rate, etc.

    Args:
        batch_size (int): Batch size to be used for training.
        bidirectional (bool): If set to False then the hyperparameters for a simple TiFGAN
            will be returned rather than a BiTiFGAN.

    Returns:
        A dictionary with the optimization hyperparameters.
    """
    params_optimization = dict()
    params_optimization['batch_size'] = batch_size
    params_optimization['epoch'] = 10000
    params_optimization['n_critic'] = 5
    params_optimization["clip_grads"] = True
    params_optimization['generator'] = dict()
    params_optimization['generator']['optimizer'] = 'adam'
    params_optimization['generator']['kwargs'] = {'beta1': 0.5, 'beta2': 0.9}
    params_optimization['generator']['learning_rate'] = 1e-4
    params_optimization['discriminator'] = dict()
    params_optimization['discriminator']['optimizer'] = 'adam'
    params_optimization['discriminator']['kwargs'] = {'beta1': 0.5, 'beta2': 0.9}
    params_optimization['discriminator']['learning_rate'] = 1e-4
    if bidirectional:
        params_optimization['encoder'] = dict()
        params_optimization['encoder']['optimizer'] = 'adam'
        params_optimization['encoder']['kwargs'] = {'beta1': 0.5, 'beta2': 0.9}
        params_optimization['encoder']['learning_rate'] = 1e-4
    return params_optimization


def get_generator_hyperparams(latent_dim, md, bn):
    """ Gets a dictionary with the generator hyperparameters.

    Args:
        latent_dim: Dimensions in the latent space.
        md: Determine the number of filters per convolutional layer.
        bn: Whether to use batch normalization or not.

    Returns:
        A dictionary determining the generator hyperparameters for its architecture design.
    """
    params_generator = dict()
    params_generator['stride'] = [2, 2, 2, 2, 2]
    params_generator['latent_dim'] = latent_dim
    params_generator['consistency_contribution'] = 0
    params_generator['nfilter'] = [8 * md, 4 * md, 2 * md, md, 1]
    params_generator['shape'] = [[12, 3], [12, 3], [12, 3], [12, 3], [12, 3]]
    params_generator['batch_norm'] = [bn, bn, bn, bn]
    params_generator['full'] = [256 * md]
    params_generator['summary'] = True
    params_generator['non_lin'] = tf.nn.tanh
    params_generator['activation'] = tf.nn.relu
    params_generator['data_size'] = 2
    params_generator['spectral_norm'] = True
    params_generator['in_conv_shape'] = [8, 4]
    return params_generator


def get_discriminator_hyperparams(md, bn, bidirectional):
    """ Gets a dictionary with the discriminator hyperparameters.

    Args:
        md: Determine the number of filters per convolutional layer.
        bn: Whether to use batch normalization or not.
        bidirectional (bool): If set to False then the hyperparameters for a simple TiFGAN
            will be returned rather than a BiTiFGAN.

    Returns:
        A dictionary determining the discriminator hyperparameters for its architecture design.
    """
    params_discriminator = dict()
    params_discriminator['stride'] = [2, 2, 2, 2, 2]
    params_discriminator['nfilter'] = [md, 2 * md, 4 * md, 8 * md, 16 * md]
    params_discriminator['shape'] = [[12, 3], [12, 3], [12, 3], [12, 3], [12, 3]]
    params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
    if bidirectional:
        params_discriminator['latent_full'] = [50]  # 256
        params_discriminator['latent_activation'] = blocks.lrelu
    params_discriminator['full'] = []  # TODO: 516
    params_discriminator['minibatch_reg'] = False
    params_discriminator['summary'] = True
    params_discriminator['data_size'] = 2
    params_discriminator['apply_phaseshuffle'] = True
    params_discriminator['spectral_norm'] = True
    params_discriminator['activation'] = blocks.lrelu
    return params_discriminator


def get_encoder_hyperparams(latent_dim, md, bn, input_shape):
    """ Gets a dictionary with the encoder hyperparameters.

    Args:
        input_shape: A list with the shape of the input image.
        latent_dim: Dimensions in the latent space.
        md: Determine the number of filters per convolutional layer.
        bn: Whether to use batch normalization or not.

    Returns:
        A dictionary determining the generator hyperparameters for its architecture design.
    """
    params_encoder = dict()
    params_encoder['input_shape'] = input_shape  # Shape of the image
    params_encoder['stride'] = [2, 2, 2, 2, 2]
    params_encoder['nfilter'] = [md, 2 * md, 4 * md, 8 * md, 16 * md]
    params_encoder['shape'] = [[12, 3], [12, 3], [12, 3], [12, 3], [12, 3]]
    params_encoder['batch_norm'] = [bn, bn, bn, bn, bn]
    params_encoder['full'] = []
    params_encoder['summary'] = True
    params_encoder['data_size'] = 2
    params_encoder['apply_phaseshuffle'] = True
    params_encoder['spectral_norm'] = True
    params_encoder['activation'] = blocks.lrelu
    params_encoder['latent_dim'] = latent_dim  # Dimensionality of the latent representation
    return params_encoder
