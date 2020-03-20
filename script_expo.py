import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow 
# Change to root path
#if os.path.basename(os.getcwd()) != 'PConv-Keras':
    #os.chdir('..')

from libs.pconv_model import PConvUnet
from libs.util import MaskGenerator, ImageChunker

##%load_ext autoreload
##%autoreload 2

# SETTINGS
SAMPLE_IMAGE = '/content/PConv-Keras/data/mickey.jpg.jpg'
BATCH_SIZE = 4


# Image samplings
crops = [
    [512, 512],[300, 200], [512, 200], [800, 200],
    [300, 512], [512, 512], [800, 512],
    [300, 1200], [512, 1200] ,[2000,2000],
]

# Setup the figure
_, axes = plt.subplots(3, 3, figsize=(15, 15))

# Set random seed
np.random.seed(7)

# Lists for saving images and masks
imgs, masks = [], []

# Plot images
for crop, ax in zip(crops, axes.flatten()):
    
    # Load image
    im = Image.open(SAMPLE_IMAGE).resize((2048, 2048))
    
    # Crop image
    h, w = im.height, im.width
    left = np.random.randint(0, w - crop[1])
    right = left + crop[1]
    upper = np.random.randint(0, h - crop[0])
    lower = upper + crop[0]
    im = im.crop((left, upper, right, lower))

    # Create masked array
    im = np.array(im) / 255
    mask_gen = MaskGenerator(*crop)
    mask = mask_gen._generate_mask()
    im[mask==0] = 1
    
    # Store for prediction
    imgs.append(im)
    masks.append(mask)

    # Show image
    ax.imshow(im)
    ax.set_title("{}x{}".format(crop[0], crop[1]))


from libs.pconv_model import PConvUnet
model = PConvUnet(vgg_weights=None, inference_only=True)
model.load(r"/content/pconv_imagenet.26-1.07.h5", train_bn=False)
chunker = ImageChunker(512, 512, 30)

def plot_images(images, s=5):
    _, axes = plt.subplots(1, len(images), figsize=(s*len(images), s))
    if len(images) == 1:
        axes = [axes]
    for img, ax in zip(images, axes):
        ax.imshow(img)
    plt.show()
    
for img, mask in zip(imgs, masks):
    print("Image with size: {}".format(img.shape))

    # Process sample
    chunked_images = chunker.dimension_preprocess(deepcopy(img))
    chunked_masks = chunker.dimension_preprocess(deepcopy(mask))
    pred_imgs = model.predict([chunked_images, chunked_masks])
    reconstructed_image = chunker.dimension_postprocess(pred_imgs, img)

    # Plot results

    plot_images(chunked_images)
    plot_images(pred_imgs)
plot_images([img, reconstructed_image], s=5)
plot_images(pred_imgs)