# python imports
import os
import logging
import numpy as np

# project imports
from lab2im.utils import utils
from lab2im.image_generator import ImageGenerator


# -------------------------------------------------- full brain ----------------------------------------------------

# path training labels directory (can also be path of a single image) and result folder
paths = '../data_example/brain_label_map.nii.gz'
result_folder = '../data_example/generated_images'

# general parameters
n_examples = 10
batchsize = 1
crop = 112  # crop produced image to this size
output_divisible_by_n = 16  # forces output images to have a shape divisible by 16

# materials to load
path_generation_labels = '../data_example/generation_labels.npy'
path_segmentation_labels = '../data_example/segmentation_labels.npy'
path_generation_classes = '../data_example/generation_classes.npy'
path_prior_means = '../data_example/prior_means.npy'
path_prior_stds = '../data_example/prior_stds.npy'

########################################################################################################

# load label list, classes list and intensity ranges if necessary
generation_label_list, generation_neutral_labels = utils.get_list_labels(path_generation_labels, FS_sort=True)
if path_segmentation_labels is not None:
    segmentation_label_list, _ = utils.get_list_labels(path_segmentation_labels, FS_sort=True)
else:
    segmentation_label_list = generation_label_list

# instantiate BrainGenerator object
logging.getLogger('tensorflow').disabled = True
brain_generator = ImageGenerator(labels_dir=paths,
                                 generation_labels=generation_label_list,
                                 output_labels=segmentation_label_list,
                                 generation_classes=path_generation_classes,
                                 prior_means=path_prior_means,
                                 prior_stds=path_prior_stds,
                                 prior_distributions='normal',
                                 output_shape=crop,
                                 output_div_by_n=output_divisible_by_n)

if not os.path.exists(os.path.join(result_folder)):
    os.mkdir(result_folder)

for n in range(n_examples):

    # generate new image and corresponding labels
    im, lab = brain_generator.generate_image()

    # save image
    for b in range(batchsize):
        utils.save_volume(np.squeeze(im[b, ...]), brain_generator.aff, brain_generator.header,
                          os.path.join(result_folder, 'minibatch_{}_image_{}.nii.gz'.format(n, b)))
        utils.save_volume(np.squeeze(lab[b, ...]), brain_generator.aff, brain_generator.header,
                          os.path.join(result_folder, 'minibatch_{}_labels_{}.nii.gz'.format(n, b)))
