import logging
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, DetrendOperations
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer
import numpy as np
import json

class EmotibitVisualizer:
    def __init__(self):
        self.current_preset = 0
        self.window_size = 200  # Increased window size
        self.auto_switch = True

        # Initialize data buffers for each preset
        self.preset_data = {
            'DEFAULT': np.zeros((12, self.window_size)),
            'AUXILIARY': np.zeros((12, self.window_size)),
            'ANCILLARY': np.zeros((12, self.window_size))
        }

        self.setup_board()
        self.setup_gui()

    def setup_board(self):
        params = BrainFlowInputParams()
        params.ip_address = '192.168.229.255'
        params.ip_port = 3132
        params.timeout = 15

        self.board = BoardShim(BoardIds.EMOTIBIT_BOARD, params)
        self.board.prepare_session()

        # Channel mapping with colors and names
        self.channels_map = {
            'DEFAULT': {
                'Accelerometer': {
                    'channels': [1,2,3], 
                    'detrend': True,
                    'colors': ['r', 'g', 'b'],
                    'names': ['X', 'Y', 'Z']
                },
                'Gyroscope': {
                    'channels': [4,5,6], 
                    'detrend': True,
                    'colors': ['r', 'g', 'b'],
                    'names': ['X', 'Y', 'Z']
                },
                'Magnetometer': {
                    'channels': [7,8,9], 
                    'detrend': True,
                    'colors': ['r', 'g', 'b'],
                    'names': ['X', 'Y', 'Z']
                }
            },
            'AUXILIARY': {
                'PPG': {
                    'channels': [1,2,3], 
                    'detrend': True,
                    'colors': ['r', 'g', 'b'],
                    'names': ['Red', 'IR', 'Green']
                }
            },
            'ANCILLARY': {
                'Biometrics': {
                    'channels': [1,2], 
                    'detrend': False,
                    'colors': ['y', 'c'],
                    'names': ['EDA', 'Temp']
                }
            }
        }

        self.board.start_stream(65536)
        self.switch_preset('DEFAULT')

    def switch_preset(self, preset_name):
        preset_map = {
            'DEFAULT': BrainFlowPresets.DEFAULT_PRESET,
            'AUXILIARY': BrainFlowPresets.AUXILIARY_PRESET,
            'ANCILLARY': BrainFlowPresets.ANCILLARY_PRESET
        }

        try:
            preset_json = json.dumps({'preset': str(int(preset_map[preset_name]))})
            self.board.config_board(preset_json)
        except Exception as e:
            print(f"Error switching preset: {e}")

    def setup_gui(self):
        self.app = QApplication([])

        # Main window
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout()

        # Graph widget
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground('w')  # White background

        # Control buttons
        self.button_layout = QHBoxLayout()
        self.auto_switch_button = QPushButton('Auto Switch: ON')
        self.auto_switch_button.clicked.connect(self.toggle_auto_switch)
        self.button_layout.addWidget(self.auto_switch_button)

        for preset in self.channels_map.keys():
            btn = QPushButton(preset)
            btn.clicked.connect(lambda checked, p=preset: self.manual_switch_preset(p))
            self.button_layout.addWidget(btn)

        self.main_layout.addWidget(self.win)
        self.main_layout.addLayout(self.button_layout)

        self.main_widget.setLayout(self.main_layout)
        self.main_widget.resize(1200, 800)

        # Setup plots
        self.plots = {}
        self.curves = {}

        row = 0
        for preset in self.channels_map:
            for sensor, info in self.channels_map[preset].items():
                p = self.win.addPlot(row=row, col=0)
                p.showGrid(x=True, y=True)
                p.setLabel('left', sensor)
                p.setLabel('bottom', 'Samples')
                p.addLegend()

                self.curves[f"{preset}_{sensor}"] = []
                for idx, name in enumerate(info['names']):
                    curve = p.plot(pen=info['colors'][idx], name=f"{name}")
                    self.curves[f"{preset}_{sensor}"].append(curve)

                row += 1

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)  # 50Hz update

        self.preset_timer = QTimer()
        self.preset_timer.timeout.connect(self.cycle_preset)
        self.preset_timer.start(1000)  # 1Hz preset switch

    def toggle_auto_switch(self):
        self.auto_switch = not self.auto_switch
        self.auto_switch_button.setText(f'Auto Switch: {"ON" if self.auto_switch else "OFF"}')

    def manual_switch_preset(self, preset):
        self.auto_switch = False
        self.auto_switch_button.setText('Auto Switch: OFF')
        self.switch_preset(preset)

    def cycle_preset(self):
        if not self.auto_switch:
            return

        presets = list(self.channels_map.keys())
        self.current_preset = (self.current_preset + 1) % len(presets)
        self.switch_preset(presets[self.current_preset])

        try:
            data = self.board.get_current_board_data(self.window_size)
            if data.size > 0:
                self.preset_data[presets[self.current_preset]] = data
        except Exception as e:
            print(f"Error getting data: {e}")

    def update(self):
        for preset in self.channels_map:
            data = self.preset_data[preset]
            if data is None or np.all(data == 0):
                continue

            for sensor, info in self.channels_map[preset].items():
                curves = self.curves[f"{preset}_{sensor}"]
                for idx, channel in enumerate(info['channels']):
                    if channel < data.shape[0]:
                        channel_data = data[channel]
                        if info['detrend'] and not np.allclose(channel_data, 0):
                            DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
                        curves[idx].setData(channel_data)

    def cleanup(self):
        if self.board.is_prepared():
            self.board.release_session()

    def run(self):
        try:
            self.main_widget.show()
            self.app.exec_()
        finally:
            self.cleanup()

def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    visualizer = EmotibitVisualizer()
    visualizer.run()

if __name__ == '__main__':
    main()
