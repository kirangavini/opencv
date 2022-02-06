"""
=====================================
Image Registration
=====================================
In this example, we use phase cross-correlation to identify the
relative shift between two similar-sized images.
The ``phase_cross_correlation`` function uses cross-correlation in
Fourier space, optionally employing an upsampled matrix-multiplication
DFT to achieve arbitrary subpixel precision [1]_.
.. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
       "Efficient subpixel image registration algorithms," Optics Letters 33,
       156-158 (2008). :DOI:`10.1364/OL.33.000156`
"""
import numpy as np
import os
import matplotlib.pyplot as plt

from skimage import data, io, color
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def subpixel_translation(new_template, template):
    # Make output dir
    if not os.path.isdir('Translation_NRMSE'):
        os.mkdir('Translation_NRMSE')

    path = 'Translation_NRMSE'
    translation = np.zeros([len(template), 2])
    for i in range(len(template)):
        new_t = new_template[i]
        t = template[i]
        shift, error, diffphase = phase_cross_correlation(new_t, t, upsample_factor = 100)
        translation[i, :] = np.array(shift)
        new_path = os.path.join(path, str(i+1) + '_point.txt')
        append_new_line(new_path, ','.join([str(i) for i in shift]))
        # translation.append(shift)
        # translation = np.array(translation)
        # print(translation)

    return translation

"""
if __name__ == '__main__':
    image_name = 'Humen.jpg'
    image = io.imread(image_name)
    image = color.rgb2gray(image)
    shift = (-12.9425, 13.8421)
    # The shift corresponds to the pixel offset relative to the reference image
    offset_image = fourier_shift(np.fft.fftn(image), shift)
    offset_image = np.fft.ifftn(offset_image)
    print("Known offset (y, x): {}".format(shift))

    # pixel precision first
    shift, error, diffphase = phase_cross_correlation(image, offset_image)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1, adjustable='box')
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box')
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')

    ax2.imshow(offset_image.real, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Offset image')

    # Show the output of a cross-correlation to show what the algorithm is
    # doing behind the scenes
    image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Cross-correlation")

    plt.show()

    print("Detected pixel offset (y, x): {}".format(shift))

    # subpixel precision
    shift, error, diffphase = register_translation(image, offset_image, 100)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1, adjustable='box')
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box')
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')

    ax2.imshow(offset_image.real, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Offset image')

    # Calculate the upsampled DFT, again to show what the algorithm is doing
    # behind the scenes.  Constants correspond to calculated values in routine.
    # See source code for details.
    cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Supersampled XC sub-area")


    plt.show()

    print("Detected subpixel offset (y, x): {}".format(shift))
"""