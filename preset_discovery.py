import argparse
import logging
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import sys

def analyze_preset(preset):
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = BrainFlowInputParams()
    params.ip_address = '192.168.229.255'
    params.ip_port = 3132
    params.timeout = 15

    try:
        board_shim = BoardShim(BoardIds.EMOTIBIT_BOARD, params)
        board_shim.prepare_session()

        # Set the preset before starting stream
        print(f"\nTesting preset {preset}")
        board_shim.set_board_preset(preset)

        board_shim.start_stream(65536)

        # Wait a bit to collect data
        import time
        time.sleep(5)

        # Get data
        data = board_shim.get_current_board_data(125)  # 5 seconds at 25Hz

        # Analyze each channel
        num_channels = data.shape[0]
        print(f"Found {num_channels} channels")

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

    except BaseException as e:
        logging.warning(f'Exception: {str(e)}', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()

def main():
    if len(sys.argv) > 1:
        preset = int(sys.argv[1])
    else:
        preset = 0  # default preset

    analyze_preset(preset)

if __name__ == '__main__':
    main()
