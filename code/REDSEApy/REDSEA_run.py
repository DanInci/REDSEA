import argparse
import os
import json
import numpy as np
import pandas as pd

from REDSEA import run_compensation


def create_compensation_stats(data_compensated_df, boundary_mod, element_shape, element_size, image_id=None):
    data_compensated_cells_df = data_compensated_df[data_compensated_df["cell_compensated_area"] != 0]

    return {
        'image_id': image_id if image_id else 'Overall',
        'total_cells': data_compensated_df.shape[0],
        'cells_compensated': data_compensated_cells_df.shape[0],
        'compensation_size': element_size,
        'compensation_shape': 'square' if element_shape == 1 else 'diamond',
        'compensation_mode': 'boundary' if boundary_mod == 2 else 'whole cell',
        'mean_cell_area': np.mean(data_compensated_df['cell_size']),
        'mean_cell_compensated_area': np.mean(data_compensated_cells_df["cell_compensated_area"]),
        'cell_percent_compensated': data_compensated_cells_df.shape[0] / data_compensated_df.shape[0] * 100,
        'pixel_percent_compensated': np.mean(data_compensated_cells_df["cell_compensated_area"] / data_compensated_cells_df["cell_size"]) * 100
    }


def main():
    parser = argparse.ArgumentParser(description='REDSEA')
    parser.add_argument('--base_path', type=str, required=True,
                        help='configuration_path')
    args = parser.parse_args()

    config_path = os.path.join(args.base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # define base directories for experiment
    data_dir = config['input_dir']
    output_dir = config['output_dir']
    channels_path = os.path.join(data_dir, 'channels.csv')
    image_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')

    # parameters for compensation
    boundary_mod_list = config['boundary_mod']  # 2 means boundary
    REDSEAChecker_list = config['REDSEAChecker']  # 1 means subtract+ reinforce
    element_shape_list = config['element_shape']  # star, 1 == square size
    element_size_list = config['element_size']  # star or square extension size
    norm_channels = config['norm_channels']

    compensation_stats_list = []
    for boundary_mod in boundary_mod_list:
        for REDSEAChecker in REDSEAChecker_list:
            for element_shape in element_shape_list:
                for element_size in element_size_list:
                    print("\n===============================================")
                    print("Running REDSEA for the following configuration:")
                    print(f"Boundary Mod: {boundary_mod} ({'whole cell compensation' if boundary_mod == 1 else 'boundary compensation' if boundary_mod == 2 else '???'})")
                    print(f"REDSEAChecker: {REDSEAChecker} ({'subtraction and reinforcement' if REDSEAChecker == 1 else 'only substraction' if REDSEAChecker == 0 else '???'})")
                    print(f"Element Shape: {element_shape} ({'square' if element_shape == 1 else 'diamond' if element_shape == 2 else '???'})")
                    print(f"Element Size: {element_size}px")
                    print("===============================================\n")
                    params_dir = f'BM={boundary_mod}_RC={REDSEAChecker}_Shape={element_shape}_Size={element_size}'

                    aggregated_data_df = pd.DataFrame()
                    aggregated_data_scaled_df = pd.DataFrame()
                    aggregated_data_compensated_df = pd.DataFrame()
                    aggregated_data_compensated_scaled_df = pd.DataFrame()

                    images = np.sort([f for f in os.listdir(image_dir)])
                    for image in images:
                        tiff_path = os.path.join(image_dir, image)
                        mask_path = os.path.join(masks_dir, f'{image}.mat')

                        print(f'Doing cell compensation for {image}...')
                        compensation = run_compensation(channels_path, tiff_path, mask_path, norm_channels,
                                         boundary_mod, REDSEAChecker, element_shape, element_size)

                        data_df, data_scaled_df, data_compensated_df, data_compensated_scaled_df = compensation

                        compensation_stats = create_compensation_stats(data_compensated_df, boundary_mod, element_shape, element_size, image_id=image)
                        compensation_stats_list.append(compensation_stats)
                        print('{}: There were {}/{} ({:.2f}%) cells that suffered compensation.'.format(image, compensation_stats['cells_compensated'], compensation_stats['total_cells'], compensation_stats['cell_percent_compensated']))
                        print('{}: On average, for cells that were compensated, {:.2f}% of cell area was affected during compensation.'.format(image, compensation_stats['pixel_percent_compensated']))

                        # output compensation results
                        compensation_output_dir = os.path.join(output_dir, image, params_dir)
                        if not os.path.exists(compensation_output_dir):
                            os.makedirs(compensation_output_dir)

                        data_df.to_csv(os.path.join(compensation_output_dir, 'data.csv'), index=False)
                        data_scaled_df.to_csv(os.path.join(compensation_output_dir, 'data_scaled.csv'), index=False)
                        data_compensated_df.to_csv(os.path.join(compensation_output_dir, 'data_compensated.csv'), index=False)
                        data_compensated_scaled_df.to_csv(os.path.join(compensation_output_dir, 'data_compensated_scaled.csv'), index=False)

                        data_df['image_name'] = image
                        data_scaled_df['image_name'] = image
                        data_compensated_df['image_name'] = image
                        data_compensated_scaled_df['image_name'] = image

                        aggregated_data_df = pd.concat([aggregated_data_df, data_df], axis=0)
                        aggregated_data_scaled_df = pd.concat([aggregated_data_scaled_df, data_scaled_df], axis=0)
                        aggregated_data_compensated_df = pd.concat([aggregated_data_compensated_df, data_compensated_df], axis=0)
                        aggregated_data_compensated_scaled_df = pd.concat([aggregated_data_compensated_scaled_df, data_compensated_scaled_df], axis=0)

                    # output aggregated compensation results
                    if aggregated_data_df.shape[0] > 0:
                        aggregated_output_dir = os.path.join(output_dir, 'aggregated', params_dir)
                        if not os.path.exists(aggregated_output_dir):
                            os.makedirs(aggregated_output_dir)

                        aggregated_data_df.to_csv(os.path.join(aggregated_output_dir, 'data.csv'), index=False)
                        aggregated_data_scaled_df.to_csv(os.path.join(aggregated_output_dir, 'data_scaled.csv'), index=False)
                        aggregated_data_compensated_df.to_csv(os.path.join(aggregated_output_dir, 'data_compensated.csv'), index=False)
                        aggregated_data_compensated_scaled_df.to_csv(os.path.join(aggregated_output_dir, 'data_compensated_scaled.csv'), index=False)

                        aggregated_compensation_stats = create_compensation_stats(aggregated_data_compensated_df, boundary_mod, element_shape, element_size)
                        compensation_stats_list.append(aggregated_compensation_stats)
                        print('Overall aggregated compensation statistics for all images...')
                        print('There were {}/{} ({:.2f}%) cells that suffered compensation.'.format(aggregated_compensation_stats['cells_compensated'], aggregated_compensation_stats['total_cells'], aggregated_compensation_stats['cell_percent_compensated']))
                        print('On average, for cells that were compensated, {:.2f}% of cell area was affected during compensation.'.format(aggregated_compensation_stats['pixel_percent_compensated']))

    compensation_stats_df = pd.DataFrame(compensation_stats_list)
    compensation_stats_df.to_csv(os.path.join(output_dir, 'compensation_stats.csv'), index=False)


if __name__ == '__main__':
    main()
