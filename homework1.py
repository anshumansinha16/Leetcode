import numpy as np
import imageio
from matplotlib import pyplot as plt
import sys
import os


def mykmeans(pixels, K):
    """
    Your goal of this assignment is implementing your own K-means.

    Input:
        pixels: data set. Each row contains one data point. For image
        dataset, it contains 3 columns, each column corresponding to Red,
        Green, and Blue component.

        K: the number of desired clusters. Too high value of K may result in
        empty cluster error. Then, you need to reduce it.

    Output:
        class: the class assignment of each data point in pixels. The
        assignment should be 0, 1, 2, etc. For K = 5, for example, each cell
        of class should be either 0, 1, 2, 3, or 4. The output should be a
        column vector with size(pixels, 1) elements.

        centroid: the location of K centroids in your result. With images,
        each centroid corresponds to the representative color of each
        cluster. The output should be a matrix with K rows and
        3 columns. The range of values should be [0, 255].
    """
    raise NotImplementedError

def mykmedoids(pixels, K):
    """
    Your goal of this assignment is implementing your own K-medoids.
    Please refer to the instructions carefully, and we encourage you to
    consult with other resources about this algorithm on the web.

    Input:
        pixels: data set. Each row contains one data point. For image
        dataset, it contains 3 columns, each column corresponding to Red,
        Green, and Blue component.

        K: the number of desired clusters. Too high value of K may result in
        empty cluster error. Then, you need to reduce it.

    Output:
        class: the class assignment of each data point in pixels. The
        assignment should be 0, 1, 2, etc. For K = 5, for example, each cell
        of class should be either 0, 1, 2, 3, or 4. The output should be a
        column vector with size(pixels, 1) elements.

        centroid: the location of K centroids in your result. With images,
        each centroid corresponds to the representative color of each
        cluster. The output should be a matrix with K rows and
        3 columns. The range of values should be [0, 255].
    """
    raise NotImplementedError

def main():
    if(len(sys.argv) < 2):
        print("Please supply an image file")
        return
    print(1)

    image_file_name = sys.argv[1]
    K = 5 if len(sys.argv) == 2 else int(sys.argv[2])
    print(image_file_name, K)
    print('hello1')
    
    im = np.asarray(imageio.imread(image_file_name))

    fig, axs = plt.subplots(1, 2)

    print('hello2')

    classes, centers = mykmedoids(im, K)
    print(classes, centers)
    new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
    imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
    axs[0].imshow(new_im)
    axs[0].set_title('K-medoids')

    classes, centers = mykmeans(im, K)
    print(classes, centers)
    new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
    imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
    axs[1].imshow(new_im)
    axs[1].set_title('K-means')

    plt.show()

if __name__ == '__main__':
    main()
    print(1)
