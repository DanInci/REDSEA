import numpy as np
import skimage.measure
import skimage.morphology
from tqdm import tqdm

# 1 newLmod (segmentation mask matrix)
# data
# counts_no_noise (counts matrix, row, col, channel)
# channel_norm_identity (plots, leave out for now)
# element_shape, element_size
# REDSEAChecker: subtract, reinforce selection
def MIBIboundary_compensation_boundarySA(newLmod, data, counts_no_noise, channel_norm_identity, element_shape, element_size, REDSEAChecker):
    n_cells = np.max(newLmod)
    n_channels = len(channel_norm_identity)

    # compute cell-2-cell contact matrix
    cell_pair_map = create_cell_contact_matrix(newLmod)

    ###############
    cell_boundary_total = np.sum(cell_pair_map, axis=0)  # count the boundary
    ############### this step might cause error in ark version, double check with YH

    # divide to get fraction
    cell_boundary_total_matrix = np.tile(cell_boundary_total, (n_cells, 1))
    # cellBoundaryTotalMatrix = repmat(cellBoundaryTotal',[1 cellNum]);
    cell_pair_norm = REDSEAChecker * np.identity(n_cells) - cell_pair_map / cell_boundary_total_matrix
    cell_pair_norm = np.transpose(cell_pair_norm)  # this is a weird bug in python, need to transpose

    # compute signals from pixels along the boundary of cells
    MIBIdataNearEdge1 = compute_pixels_near_edge(newLmod, counts_no_noise, n_cells, n_channels, element_shape, element_size)

    # perform signal compensation on the cell boundaries
    MIBIdataNorm2 = np.transpose(np.dot(np.transpose(MIBIdataNearEdge1), cell_pair_norm))
    # this is boundary signal subtracted by cell neighbor boundary
    MIBIdataNorm2 = MIBIdataNorm2 + data  # reinforce onto the whole cell signal (original signal)
    MIBIdataNorm2[MIBIdataNorm2 < 0] = 0  # clear out the negative ones

    # flip the channel_norm_identity for calculation
    rev_channel_norm_identity = np.ones_like(channel_norm_identity) - channel_norm_identity

    # composite the normalized channels with non-normalized channels
    # MIBIdataNorm2 is the matrix to return
    MIBIdataNorm2 = data * np.transpose(np.tile(rev_channel_norm_identity, (1, n_cells))) + \
                    MIBIdataNorm2 * np.transpose(np.tile(channel_norm_identity, (1, n_cells)))

    return MIBIdataNorm2


# this function is for computing cell-2-cell contact matrix
def create_cell_contact_matrix(newLmod):
    [n_row, n_col] = newLmod.shape
    n_cells = np.max(newLmod)
    cell_pair_map = np.zeros((n_cells, n_cells))  # cell-cell shared perimeter matrix container

    ## need to add border to the segmentation mask (newLmod in this case)
    newLmod_border = np.pad(newLmod, pad_width=1, mode='constant', constant_values=0)

    # start looping the mask and produce the cell-cell contact matrix
    for i in range(n_row):
        for j in range(n_col):
            if newLmod[i, j] == 0:
                temp_matrix = newLmod_border[i:i + 3, j:j + 3]  # the 3x3 window, xy shifted +1 due to border
                temp_factors = np.unique(temp_matrix)  # unique
                temp_factors = temp_factors - 1  # minus one for python index

                if len(temp_factors) == 3:  # means only two cells
                    cell_pair_map[temp_factors[1], temp_factors[2]] = cell_pair_map[temp_factors[1], temp_factors[2]] + 1  # count zero

                elif len(temp_factors) == 4:  # means three cells, three pairs
                    cell_pair_map[temp_factors[1], temp_factors[2]] = cell_pair_map[temp_factors[1], temp_factors[2]] + 1  # count zero
                    cell_pair_map[temp_factors[1], temp_factors[3]] = cell_pair_map[temp_factors[1], temp_factors[3]] + 1  # count zero
                    cell_pair_map[temp_factors[2], temp_factors[3]] = cell_pair_map[temp_factors[2], temp_factors[3]] + 1  # count zero

                elif len(temp_factors) == 5:  # means four cells, 6 pairs
                    cell_pair_map[temp_factors[1], temp_factors[2]] = cell_pair_map[temp_factors[1], temp_factors[2]] + 1  # count zero
                    cell_pair_map[temp_factors[1], temp_factors[3]] = cell_pair_map[temp_factors[1], temp_factors[3]] + 1  # count zero
                    cell_pair_map[temp_factors[1], temp_factors[4]] = cell_pair_map[temp_factors[1], temp_factors[4]] + 1  # count zero

                    cell_pair_map[temp_factors[2], temp_factors[3]] = cell_pair_map[temp_factors[2], temp_factors[3]] + 1  # count zero
                    cell_pair_map[temp_factors[2], temp_factors[4]] = cell_pair_map[temp_factors[2], temp_factors[4]] + 1  # count zero

                    cell_pair_map[temp_factors[3], temp_factors[4]] = cell_pair_map[temp_factors[3], temp_factors[4]] + 1  # count zero

    # formatting of the cell maps

    # double direction
    cell_pair_map = cell_pair_map + np.transpose(cell_pair_map)

    return cell_pair_map


# this function is for computing signals from pixels along the boundary of cells
def compute_pixels_near_edge(newLmod, counts_no_noise, n_cells, n_channels, element_shape, element_size):
    [n_row, n_col] = newLmod.shape
    MIBIdataNearEdge1 = np.zeros((n_cells, n_channels))

    def is_inside(x, y):
        return 0 <= x < n_row and 0 <= y < n_col

    ###### pre-calculated shape
    if element_shape == 1:  # square
        square = skimage.morphology.square(2 * element_size + 1)
        square_loc = np.where(square == 1)
    elif element_shape == 2:  # diamond
        diam = skimage.morphology.diamond(element_size)  # create diamond shape based on element_size
        diam_loc = np.where(diam == 1)
    else:
        raise NotImplementedError("Error elementShape Value not recognized.")
    ############

    # start the boundary region selection and count extraction
    for i in tqdm(range(n_cells), desc='Running boundarySA cell compensation ...'):
        label = i + 1  # python problem
        [temp_row, temp_col] = np.where(newLmod == label)

        # sequence in row not col, should not affect the code
        for j in range(len(temp_row)):
            ini_point = [temp_row[j] - element_size, temp_col[j] - element_size]  # corrected top-left point, can be outside image

            if element_shape == 1:  # square
                square_loc_ini_x = [item + ini_point[0] for item in square_loc[0]]
                square_loc_ini_y = [item + ini_point[1] for item in square_loc[1]]
                shape_coords = list(zip(square_loc_ini_x, square_loc_ini_y))

            elif element_shape == 2:  # diamond
                diam_loc_ini_x = [item + ini_point[0] for item in diam_loc[0]]
                diam_loc_ini_y = [item + ini_point[1] for item in diam_loc[1]]
                shape_coords = list(zip(diam_loc_ini_x, diam_loc_ini_y))

            else:
                raise NotImplementedError("Element shape not recognized")

            # finish add to ini point
            filtered_shape_coords = list(filter(lambda x: is_inside(x[0], x[1]), shape_coords))  # make sure not check outside coordinates
            label_in_shape = [newLmod[filtered_shape_coords[k][0], filtered_shape_coords[k][1]] for k in range(len(filtered_shape_coords))]

            if 0 in label_in_shape:
                MIBIdataNearEdge1[i, :] = MIBIdataNearEdge1[i, :] + counts_no_noise[temp_row[j], temp_col[j], :]

    return MIBIdataNearEdge1
