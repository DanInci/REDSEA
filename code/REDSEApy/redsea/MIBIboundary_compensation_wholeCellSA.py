import numpy as np


# 1 newLmod (segmentation mask matrix)
# MIBIdata
# channel_norm_identity (plots, leave out for now)
# REDSEAChecker: subtract, reinforce selection
def MIBIboundary_compensation_wholeCellSA(newLmod, MIBIdata, channel_norm_identity, REDSEAChecker):
    n_cells = np.max(newLmod)

    # compute cell-2-cell contact matrix
    cell_pair_map = create_cell_contact_matrix(newLmod)

    ###############
    cell_boundary_total = np.sum(cell_pair_map, axis=0)  # count the boundary
    ############### this step might cause error in ark version, double check with YH

    # divide to get fraction
    cell_boundary_total_matrix = np.tile(cell_boundary_total, (n_cells, 1))
    # cellBoundaryTotalMatrix = repmat(cellBoundaryTotal',[1 cellNum]);
    cell_pair_norm = (REDSEAChecker + 1) * np.identity(n_cells) - cell_pair_map / cell_boundary_total_matrix
    cell_pair_norm = np.transpose(cell_pair_norm)  # this is a weird bug in python, need to transpose

    # perform signal compensation on the whole cell
    MIBIdataNorm1 = np.transpose(np.dot(np.transpose(MIBIdata), cell_pair_norm))
    MIBIdataNorm1[MIBIdataNorm1 < 0] = 0  # clear out the negative ones

    # flip the channel_norm_identity for calculation
    rev_channel_norm_identity = np.ones_like(channel_norm_identity) - channel_norm_identity

    # composite the normalized channels with non-normalized channels
    # MIBIdataNorm is the matrix to return
    MIBIdataNorm1 = MIBIdata * np.transpose(np.tile(rev_channel_norm_identity, (1, n_cells))) + \
                    MIBIdataNorm1 * np.transpose(np.tile(channel_norm_identity, (1, n_cells)))

    return MIBIdataNorm1


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
