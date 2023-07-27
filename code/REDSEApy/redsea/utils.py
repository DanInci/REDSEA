import glob
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt


def is_member(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value


def load_images(tiff_path, channels):
    #### this part is reading all tif files
    # read in the image and transform into a 'countsNoNoise' matrix
    files = glob.glob(tiff_path + '/*.tif*')  # read all the tiff/tif files under the folder

    if "tiff" in files[0]:
        end = ".tiff"
    else:
        end = ".tif"

    array_list = []
    for channel in channels:
        t = plt.imread(tiff_path + '/' + str(channel) + end)
        array_list.append(t)

    counts_no_noise = np.stack(array_list, axis=2)  # count matrices in the image
    ##### for this folder the matrix is read

    return counts_no_noise


def extract_cell_information(newLmod, counts_no_noise):
    n_cells = np.max(newLmod)  # how many labels/cells
    n_channels = counts_no_noise.shape[2]  # how many channels

    ### make empty container matrices
    data = np.zeros((n_cells, n_channels))
    data_scale_size = np.zeros((n_cells, n_channels))
    cell_sizes = np.zeros((n_cells, 1))

    # get the regional props for all the labels
    stats = skimage.measure.regionprops(newLmod)

    # this part extract counts data from the whole cell regions, for each individual cells etc
    for i in range(n_cells):  # for each cell (label)
        label_counts = [counts_no_noise[coord[0], coord[1], :] for coord in stats[i].coords]  # all channel count for this cell
        data[i, 0:n_channels] = np.sum(label_counts, axis=0)  # sum the counts for this cell
        data_scale_size[i, 0:n_channels] = np.sum(label_counts, axis=0) / stats[i].area  # scaled by size
        cell_sizes[i] = stats[i].area  # cell sizes

    return data, data_scale_size, cell_sizes
