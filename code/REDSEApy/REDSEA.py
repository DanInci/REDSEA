import os

import pandas as pd
from scipy.io import loadmat
from redsea.utils import *
from redsea.MIBIboundary_compensation_wholeCellSA import MIBIboundary_compensation_wholeCellSA
from redsea.MIBIboundary_compensation_boundarySA import MIBIboundary_compensation_boundarySA


# these code is just entire translation of REDSEA matlab v1.0

def run_boundarySA_compensation(channels_path, tiff_path, masks_path, norm_channels,
                     REDSEAChecker=1, element_shape=2, element_size=2):

    return run_compensation(channels_path, tiff_path, masks_path, norm_channels,
                     REDSEAChecker=REDSEAChecker, element_shape=element_shape, element_size=element_size, boundary_mod=2)


def run_wholCellSA_compensation(channels_path, tiff_path, masks_path, norm_channels, REDSEAChecker=1):
    return run_compensation(channels_path, tiff_path, masks_path, norm_channels,
                            REDSEAChecker=REDSEAChecker, boundary_mod=1)


def run_compensation(channels_path, tiff_path, mask_path, norm_channels,
                     boundary_mod=2, REDSEAChecker=1, element_shape=2, element_size=2):

    channels_csv = pd.read_csv(channels_path)  # read the channels csv
    channels = channels_csv['Label']  # only get the label column
    channels_indexes = np.where(np.isin(channels, channels))[0]  # channel indexes

    #### should be inside the function
    channel_norm_indexes = is_member(norm_channels, channels)
    channel_norm_identity = np.zeros((len(channels), 1))

    # make a flag for compensation
    for i in range(len(channel_norm_indexes)):
        channel_norm_identity[channel_norm_indexes[i]] = 1

    # load images
    counts_no_noise = load_images(tiff_path, channels)

    # Define the boundary region
    newLmod = loadmat(mask_path)['newLmod']
    n_labels = np.max(newLmod)  # how many labels/cells

    # this part extract counts data from the whole cell regions, for each individual cells etc
    data, data_scale_size, cell_sizes = extract_cell_information(newLmod, counts_no_noise)

    # now we start the boundary compensation part of code in MATLAB function:
    if boundary_mod == 1:
        # MIBIboundary_compensation_wholeCellSA(newLmod,MIBIdata,channelNormIdentity,REDSEAChecker)
        data_compensated = MIBIboundary_compensation_wholeCellSA(newLmod, data, channel_norm_identity, REDSEAChecker)
    elif boundary_mod == 2:
        # MIBIboundary_compensation_boundarySA(newLmod,data,countsNoNoise,channelNormIdentity,elementShape,elementSize,REDSEAChecker)
        data_compensated = MIBIboundary_compensation_boundarySA(newLmod, data, counts_no_noise, channel_norm_identity,
                                                                element_shape, element_size, REDSEAChecker)
    else:
        raise NotImplementedError(
            "boundary_mod can only be set to 1 (whole cell compensation) or 2 (boundary compensation)")

    # scale compensated data by size
    data_compensated_scale_size = data_compensated / cell_sizes

    # some last steps
    ############ SKIP THE POSITIVE NUCLEAR IDENTITY FILTER
    ############ SHOULD ADD by user's choice

    label_identity = np.ones(n_labels)  # this part is the skipped line (positive nuclear identity filter)
    sum_data_scale_size_in_channels = np.sum(data_scale_size[:, channels_indexes], axis=1)  # add all the channels
    label_identity[sum_data_scale_size_in_channels < 0.1] = 2  # remove the cells that does not have info in channels

    # the function should return 4 variables
    data_cells = data[label_identity == 1, :]
    data_scale_size_cells = data_scale_size[label_identity == 1, :]
    data_compensated_cells = data_compensated[label_identity == 1, :]
    data_compensated_scale_size_cells = data_compensated_scale_size[label_identity == 1, :]

    # create the final matrix's (4 types of them)

    labels_vector = np.where(label_identity == 1)
    labels_vector = [item + 1 for item in labels_vector]  # python indexing difference need to add 1

    # get cell sizes
    cell_sizes_vector = cell_sizes[label_identity == 1]
    cell_sizes_vector = [item for sublist in cell_sizes_vector for item in sublist]  # flat the list

    # Cell Metadata
    cells_metadata_df = pd.DataFrame({'cell_label': labels_vector[0].tolist(), 'cell_size': cell_sizes_vector})

    # Original Data
    data_df = pd.DataFrame(data_cells)
    data_df.columns = channels
    data_full_df = pd.concat((cells_metadata_df, data_df), axis=1)

    # Original Data Scaled by Cell Sizes
    data_scale_size_df = pd.DataFrame(data_scale_size_cells)
    data_scale_size_df.columns = channels
    data_scale_size_full_df = pd.concat((cells_metadata_df, data_scale_size_df), axis=1)

    # Compensated Data
    data_compensated_df = pd.DataFrame(data_compensated_cells)
    data_compensated_df.columns = channels
    data_compensated_full_df = pd.concat((cells_metadata_df, data_compensated_df), axis=1)

    # Compensated Data Scaled by Cell Sizes
    data_compensated_scale_size_df = pd.DataFrame(data_compensated_scale_size_cells)
    data_compensated_scale_size_df.columns = channels
    data_compensated_scale_size_full_df = pd.concat((cells_metadata_df, data_compensated_scale_size_df), axis=1)

    return data_full_df, data_scale_size_full_df, data_compensated_full_df, data_compensated_scale_size_full_df


if __name__ == "__main__":
    EXPERIMENT_PATH = '../sampleData_MIBI'
    channels_path = os.path.join(EXPERIMENT_PATH, 'channels.csv')
    tiff_path = os.path.join(EXPERIMENT_PATH, 'images/Point1')
    masks_path = os.path.join(EXPERIMENT_PATH, 'masks/Point1')

    # parameters for compensation
    boundary_mod = 2  # 2 means boundary
    REDSEAChecker = 1  # 1 means subtract+reinforce
    element_shape = 2  # star, 1 == square size
    element_size = 2  # star or square extension size

    # select which channel to normalize
    norm_channels = ['CD4', 'CD56', 'CD21 (CR2)', 'CD163', 'CD68', 'CD3', 'CD20', 'CD8a']

    compensation = run_compensation(
        channels_path, tiff_path, masks_path, norm_channels,
        boundary_mod=boundary_mod, REDSEAChecker=REDSEAChecker, element_shape=element_shape, element_size=element_size
    )
    data_full_df, data_scale_size_full_df, data_compensated_full_df, data_compensated_scale_size_full_df = compensation

    print(data_full_df.head())

    print(data_scale_size_full_df.head())

    print(data_compensated_full_df.head())

    print(data_compensated_scale_size_full_df.head())
