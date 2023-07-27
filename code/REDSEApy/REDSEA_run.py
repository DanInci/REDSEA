import argparse
import os
import json
import numpy as np
from REDSEA import run_compensation

def main():
    parser = argparse.ArgumentParser(description='STELLAR')
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
    boundary_mod = config['boundary_mod']  # 2 means boundary
    REDSEAChecker = config['REDSEAChecker']  # 1 means subtract+ reinforce
    element_shape = config['element_shape']  # star, 1 == square size
    element_size = config['element_size']  # star or square extension size
    norm_channels = config['norm_channels']

    images = np.sort([f for f in os.listdir(image_dir)])
    for image in images:
        tiff_path = os.path.join(image_dir, image)
        mask_path = os.path.join(masks_dir, f'{image}.mat')

        print(f'Doing cell compensation for {image}...')
        compensation = run_compensation(channels_path, tiff_path, mask_path, norm_channels,
                         boundary_mod, REDSEAChecker, element_shape, element_size)

        data_full_df, data_scale_size_full_df, data_compensated_full_df, data_compensated_scale_size_full_df = compensation

        # output compensation results
        compensation_output_dir = os.path.join(output_dir, image)
        if not os.path.exists(compensation_output_dir):
            os.makedirs(compensation_output_dir)

        data_full_df.to_csv(os.path.join(compensation_output_dir, 'data.csv'), index=False)
        data_scale_size_full_df.to_csv(os.path.join(compensation_output_dir, 'data_scaled.csv'), index=False)
        data_compensated_full_df.to_csv(os.path.join(compensation_output_dir, 'data_compensated.csv'), index=False)
        data_compensated_scale_size_full_df.to_csv(os.path.join(compensation_output_dir, 'data_compensated_scaled.csv'), index=False)


if __name__ == '__main__':
    main()
