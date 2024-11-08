import logging
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import sys

def analyze_board():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = BrainFlowInputParams()
    params.ip_address = '192.168.229.255'
    params.ip_port = 3132
    params.timeout = 15

    try:
        # First get available presets
        presets = BoardShim.get_board_presets(BoardIds.EMOTIBIT_BOARD)
        print(f"\nAvailable presets: {presets}")

        # Get board description
        descr = BoardShim.get_board_descr(BoardIds.EMOTIBIT_BOARD)
        print(f"\nBoard description: {descr}")

        # Connect to board and analyze channels
        board_shim = BoardShim(BoardIds.EMOTIBIT_BOARD, params)
        board_shim.prepare_session()
        board_shim.start_stream(65536)

        # Wait a bit to collect data
        time.sleep(5)

        # Get data
        data = board_shim.get_current_board_data(125)  # 5 seconds at 25Hz

        # Analyze each channel
        num_channels = data.shape[0]
        print(f"\nFound {num_channels} channels")

        for i in range(num_channels):
            channel_data = data[i]

            # Basic statistics
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)

            # Check if channel contains non-zero data
            is_active = not np.allclose(channel_data, 0)

            print(f"\nChannel {i}:")
            print(f"  Active: {is_active}")
            print(f"  Mean: {mean:.4f}")
            print(f"  Std Dev: {std:.4f}")
            print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
            print(f"  First few values: {channel_data[:5]}")

            # Try to identify the type of data
            if is_active:
                if np.all(channel_data.astype(int) == channel_data):
                    print("  Appears to be integer data")
                if 0 <= mean <= 1:
                    print("  Appears to be normalized data")
                if -1 <= mean <= 1 and -1 <= min_val <= 1 and -1 <= max_val <= 1:
                    print("  Could be normalized IMU data")
                if mean > 1000:
                    print("  Could be timestamp data")

        # Print all available methods for board configuration
        print("\nAvailable board methods:")
        for method in dir(board_shim):
            if not method.startswith('_'):  # Skip private methods
                print(f"  {method}")

    except BaseException as e:
        logging.warning(f'Exception: {str(e)}', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()

if __name__ == '__main__':
    analyze_board()
