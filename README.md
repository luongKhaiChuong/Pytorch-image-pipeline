# Pytorch-image-pipeline
An (almost) complete pipeline for basic image preprocessing and training. Currently it only accept dataset in the folder type, but still, it is kinda convention.
It works best on cuda and cpu, because the tpu part is still not optimized. But still, please enjoy it and hopefully, it can save you some time.

# Overview:
This repository provides a standardized model training class for pretrained models in PyTorch. It incorporates data preprocessing, training, progress tracking and evaluation while maintaining flexibility for different architectures and datasets. It supports TPU, CUDA and normal CPU.
These the default parameters of the library:

# Default parameters:
Pipeliner(data_path=None, image_extensions=(".jpg", ".png", ".jpeg"), device = 'cpu', split_ratio=(0.8, 0.1, 0.1), 
                 resized=None, cropped=None, is_augmented=False, horizontal_flip=False, p_horizontal=0.5, 
                 vertical_flip=False, p_vertical=0.5, jitter=False, jit_brightness=0.2, jit_contrast=0.2,
                 jit_saturation=0.2, jit_hue=0.1, rotation=0, gaussian_blur=False, blur_kernel=(5, 5), normalize=True,
                 nor_mean=[0.485, 0.456, 0.406], nor_std=[0.229, 0.224, 0.225], criterion="CrossEntropyLoss", batch_size=32, epochs=10,
                 optimizer=lambda params: optim.Adam(params, lr=0.001), scheduler=None, model_name=None, model=None, weights=None, 
                 random_state=42)
# Parameter description:
## Data handling and preprocessing:
data_path: A directory string of the dataset 

image_extensions: A tuple consisting of the extension of the images

device: Current machine used to train. Got "cpu", "cuda" and "tpu"

split_ratio: A tuple representing the dataset split ratios for training, validation, and testing

resized: A tuple (width, height) defining the target image size after resizing. If None, no resizing is applied

cropped: A tuple (width, height) specifying the crop size. If None, no cropping is applied

normalize: A boolean indicating whether to apply normalization to images

nor_mean: A list of three float values representing the mean normalization values for each channel. The default ver used ImageNet parameters

nor_std: A list of three float values representing the standard deviation normalization values for each channel. The default ver used ImageNet parameters
## Data augmentation: 
is_augmented: A boolean indicating whether data augmentation is applied

horizontal_flip: A boolean enabling horizontal flipping of images

p_horizontal: A float (0-1) representing the probability of applying horizontal flipping

vertical_flip: A boolean enabling vertical flipping of images

p_vertical: A float (0-1) representing the probability of applying vertical flipping

jitter: A boolean enabling color jittering

jit_brightness: A float controlling brightness jitter intensity

jit_contrast: A float controlling contrast jitter intensity

jit_saturation: A float controlling saturation jitter intensity

jit_hue: A float controlling hue jitter intensity

rotation: An integer specifying the maximum degree for random rotations

gaussian_blur: A boolean enabling Gaussian blurring

blur_kernel: A tuple (height, width) defining the kernel size for Gaussian blurring
## Training parameters:
criterion: A string specifying the loss function

batch_size: An integer defining the number of samples per batch

epochs: An integer specifying the number of training epochs

optimizer: A callable function that takes model parameters and returns an optimizer instance (default: optim.Adam(params, lr=0.001))

scheduler: A learning rate scheduler
## Model config:
model_name: A string representing the predefined model architecture name (if applicable)

model: A PyTorch model instance used for training

weights: A path or a pre-trained model weight file to load before training
## Miscellaneous:
random_state: An integer used to set a fixed seed for reproducibility 

# Functions: 
load_dataset(dog_barplot=32): A pipeline that automatically read your folders, extract the labels and turn them into the DataLoader form. The parameter dog_barplot is just a convention for degree of gradient for the bar plot I used to visualize the distribution of the dataset, so please dont bother. It does these things:
  - Load the data from the folders.
  - Preprocess the data (including basic transformation like normalization and resize, and augmentations)
  - Visualize the images before and after the transformations. Note: Since it will print some of the image in ALL the categories, so it might overwhelmed you if the    amount of categories exceed 10, so maybe you might want to turn them off.
  - Split the dataset into train, test, val sets. Note: I used stratified split for the sake of uniform across the datasets, which takes a lot of computational     resource, so you may want to rewrite it.
  - 
build_model(): A function that automatically build the model based on the name of the model you provided, or use the model that you assigned. It also prints the architecture of the model.

train(): A function for training the model and also have progress bar visualization using tqdm. It returns the model with the best validation loss.

report(): A function for model evaluation. Consists of confusion matrix, lineplots for loss and accuracy, and f1 score.


Feel free to contribute, suggest improvements, or just enjoy using it!
