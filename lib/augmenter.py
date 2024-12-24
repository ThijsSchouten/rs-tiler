import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt

def augment(img, msk):
    # Check if the sizes of the image and mask are correct
    assert img.shape == (512, 512, 3)
    assert msk.shape == (512, 512)

    # Define the separate components of the augmentation sequence
    spatial_seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 50% chances to horizontally flip images
        iaa.Flipud(0.5),  # 50% chances to vertically flip images
    ])
    
    color_seq = iaa.Sequential([
        iaa.Multiply((0.5, 1.5)),  # change brightness, here between 50% and 150%
        iaa.AddToHueAndSaturation((-30, 30)),  # change hue and saturation between -30 and 30
        iaa.Grayscale(0.5),  # convert images to grayscale with a 50% probability
    ])

    # imgaug expects a list of images, so we convert the image and mask to a list of one image
    img_list = [img]
    msk_list = [msk]

    # Make the spatial transformations deterministic
    spatial_seq_det = spatial_seq.to_deterministic()

    # Apply the color augmentations
    aug_img_list = color_seq(images=img_list)

    # Apply the deterministic spatial augmentations
    aug_img_list = spatial_seq_det(images=aug_img_list)
    aug_msk_list = spatial_seq_det(images=msk_list)

    # The augmenters return lists of images, so we need to take the first image of each list
    aug_img = aug_img_list[0]
    aug_msk = aug_msk_list[0]

    return aug_img, aug_msk


def save_images(filename, img, msk, aug_img=None, aug_msk=None):
    if aug_img is None and aug_msk is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img)
        axs[0].set_title('Original Image')
        axs[1].imshow(msk, cmap='gray')
        axs[1].set_title('Original Mask')
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(img)
        axs[0, 0].set_title('Original Image')
        axs[0, 1].imshow(msk, cmap='gray')
        axs[0, 1].set_title('Original Mask')
        axs[1, 0].imshow(aug_img)
        axs[1, 0].set_title('Augmented Image')
        axs[1, 1].imshow(aug_msk, cmap='gray')
        axs[1, 1].set_title('Augmented Mask')

    # Remove axis
    for ax in axs.flat:
        ax.axis('off')

    # Save the figure
    plt.savefig(filename)
    plt.close()
