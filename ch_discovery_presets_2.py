import logging
from pprint import pprint
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowPresets

def print_board_info():
    # Enable logging
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    # Get board info for each preset
    presets = {
        "DEFAULT": BrainFlowPresets.DEFAULT_PRESET,
        "AUXILIARY": BrainFlowPresets.AUXILIARY_PRESET,
        "ANCILLARY": BrainFlowPresets.ANCILLARY_PRESET
    }

    board_id = BoardIds.EMOTIBIT_BOARD

    print("\nEmotiBit Board Information:")
    print("=" * 50)

    # Get general board description
    print("\nGeneral Board Description:")
    print("-" * 30)
    board_descr = BoardShim.get_board_descr(board_id)
    pprint(board_descr)

    # Print info for each preset
    for preset_name, preset in presets.items():
        print(f"\nPreset: {preset_name}")
        print("-" * 30)
        try:
            # Get channels for this preset
            channels = BoardShim.get_board_descr(board_id, preset)
            pprint(channels)
        except Exception as e:
            print(f"Error getting info for {preset_name} preset: {str(e)}")

if __name__ == "__main__":
    print_board_info()
