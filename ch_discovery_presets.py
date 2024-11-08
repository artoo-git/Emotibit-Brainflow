import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
import numpy as np
import time

def get_preset_description(preset):
    """Return description of what data types are expected in each preset"""
    if preset == BrainFlowPresets.DEFAULT_PRESET:
        return "IMU data (accelerometer, gyroscope, magnetometer)"
    elif preset == BrainFlowPresets.AUXILIARY_PRESET:
        return "PPG data"
    elif preset == BrainFlowPresets.ANCILLARY_PRESET:
        return "EDA and temperature data"
    return "Unknown preset"

def analyze_channels_for_preset(board_shim, preset):
    """Analyze channels for a specific preset configuration"""
    # Get data
    data = board_shim.get_current_board_data(125)  # 5 seconds at 25Hz
    num_channels = data.shape[0]

    preset_desc = get_preset_description(preset)
    print(f"\nAnalyzing {num_channels} channels for preset: {preset}")
    print(f"Expected data types: {preset_desc}")

    channel_info = []
    for i in range(num_channels):
        channel_data = data[i]

        # Basic statistics
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)

        # Check if channel contains non-zero data
        is_active = not np.allclose(channel_data, 0)

        channel_info.append({
            'channel': i,
            'active': is_active,
            'mean': mean,
            'std': std,
            'range': (min_val, max_val),
            'sample_values': channel_data[:5]
        })

        print(f"\nChannel {i}:")
        print(f"  Active: {is_active}")
        print(f"  Mean: {mean:.4f}")
        print(f"  Std Dev: {std:.4f}")
        print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  First few values: {channel_data[:5]}")

        # Data type analysis based on preset
        if is_active:
            if preset == BrainFlowPresets.DEFAULT_PRESET:
                if -1 <= mean <= 1 and -1 <= min_val <= 1 and -1 <= max_val <= 1:
                    print("  Likely normalized IMU data")
                elif mean > 1000:
                    print("  Could be timestamp data")

            elif preset == BrainFlowPresets.AUXILIARY_PRESET:
                if mean > 0:
                    print("  Likely PPG data")

            elif preset == BrainFlowPresets.ANCILLARY_PRESET:
                if 20 <= mean <= 40:  # typical temperature range in Celsius
                    print("  Likely temperature data")
                elif mean >= 0 and std > 0:
                    print("  Could be EDA data")

    return channel_info

def analyze_channels():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = BrainFlowInputParams()
    params.ip_address = '192.168.229.255'
    params.ip_port = 3132
    params.timeout = 15

    try:
        board_id = BoardIds.EMOTIBIT_BOARD
        presets = [
            BrainFlowPresets.DEFAULT_PRESET,    # IMU data
            BrainFlowPresets.AUXILIARY_PRESET,  # PPG data
            BrainFlowPresets.ANCILLARY_PRESET  # EDA and temperature
        ]

        # Store results for each preset
        preset_results = {}

        for preset in presets:
            print(f"\n{'='*50}")
            print(f"Testing preset: {preset}")
            print(f"Expected data: {get_preset_description(preset)}")
            print(f"{'='*50}")

            # Initialize board with current preset
            params.preset = preset
            board_shim = BoardShim(board_id, params)
            board_shim.prepare_session()
            board_shim.start_stream(65536)

            # Wait to collect data
            time.sleep(5)

            # Analyze channels for this preset
            preset_results[preset] = analyze_channels_for_preset(board_shim, preset)

            # Clean up
            board_shim.stop_stream()
            board_shim.release_session()

            # Wait between presets
            time.sleep(2)

        # Print summary
        print("\nSummary of Channel Analysis Across Presets:")
        for preset, channels in preset_results.items():
            active_channels = sum(1 for ch in channels if ch['active'])
            print(f"\nPreset {preset} ({get_preset_description(preset)}):")
            print(f"  Total Channels: {len(channels)}")
            print(f"  Active Channels: {active_channels}")

        # Print board description
        print("\nBoard Description:")
        print(BoardShim.get_board_descr(board_id))

    except BaseException as e:
        logging.warning(f'Exception: {str(e)}', exc_info=True)
    finally:
        logging.info('End')

if __name__ == '__main__':
    analyze_channels()
