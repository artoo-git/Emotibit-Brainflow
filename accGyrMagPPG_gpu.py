import logging
import numpy as np
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, DetrendOperations
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtOpenGL import QGLFormat
import json

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.current_preset = 'DEFAULT'

        # Configure OpenGL format
        fmt = QGLFormat()
        fmt.setSampleBuffers(True)
        fmt.setSwapInterval(1)
        QGLFormat.setDefaultFormat(fmt)

        pg.setConfigOptions(
            antialias=True,
            useOpenGL=True,
            enableExperimental=True
        )

        # Channel configurations for different presets
        self.preset_configs = {
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
                'PPG_IR': {
                    'channels': [1], 
                    'detrend': True,
                    'colors': ['r'],
                    'names': ['IR']
                },
                'PPG_Red': {
                    'channels': [2], 
                    'detrend': True,
                    'colors': ['darkred'],
                    'names': ['Red']
                },
                'PPG_Green': {
                    'channels': [3], 
                    'detrend': True,
                    'colors': ['g'],
                    'names': ['Green']
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

        self.sensors = self.preset_configs['DEFAULT']
        self.window_size = 200
        self.update_speed_ms = 20  # 50Hz update

        # Setup GUI
        self.app = QApplication([])
        self.main_window = QWidget()
        self.main_window.setWindowTitle('EmotiBit Data Viewer')
        self.layout = QVBoxLayout()

        # Create button layout
        self.button_layout = QHBoxLayout()

        # Create preset buttons
        self.default_button = QPushButton('IMU (DEFAULT)')
        self.auxiliary_button = QPushButton('PPG (AUXILIARY)')
        self.ancillary_button = QPushButton('EDA (ANCILLARY)')

        self.default_button.clicked.connect(lambda: self.change_preset('DEFAULT'))
        self.auxiliary_button.clicked.connect(lambda: self.change_preset('AUXILIARY'))
        self.ancillary_button.clicked.connect(lambda: self.change_preset('ANCILLARY'))

        self.button_layout.addWidget(self.default_button)
        self.button_layout.addWidget(self.auxiliary_button)
        self.button_layout.addWidget(self.ancillary_button)

        self.layout.addLayout(self.button_layout)

        # Create plot widget
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground('w')
        self.layout.addWidget(self.win)

        self.main_window.setLayout(self.layout)
        self.main_window.resize(1600, 1000)

        self._init_timeseries()

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

        self.main_window.show()
        self.app.exec_()

    def change_preset(self, preset_name):
        preset_map = {
            'DEFAULT': BrainFlowPresets.DEFAULT_PRESET,
            'AUXILIARY': BrainFlowPresets.AUXILIARY_PRESET,
            'ANCILLARY': BrainFlowPresets.ANCILLARY_PRESET
        }

        try:
            preset_json = json.dumps({'preset': str(int(preset_map[preset_name]))})
            self.board_shim.config_board(preset_json)
            self.current_preset = preset_name
            self.sensors = self.preset_configs[preset_name]
            self.win.clear()
            self._init_timeseries()
            print(f"Changed to preset {preset_name}")
        except Exception as e:
            print(f"Error changing preset: {e}")

    def _init_timeseries(self):
        self.plots = {}
        self.curves = {}

        for row, (sensor_name, sensor_info) in enumerate(self.sensors.items()):
            p = self.win.addPlot(row=row, col=0)
            p.setDownsampling(auto=True, mode='peak')
            p.setClipToView(True)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel('left', sensor_name)
            p.setLabel('bottom', 'Samples' if row == len(self.sensors)-1 else '')
            p.getAxis('left').setPen('k')
            p.getAxis('bottom').setPen('k')
            p.addLegend()

            vb = p.getViewBox()
            vb.setAspectLocked(False)
            vb.enableAutoRange(axis='y')

            self.curves[sensor_name] = []
            for name, color in zip(sensor_info['names'], sensor_info['colors']):
                curve = p.plot(
                    pen=pg.mkPen(color=color, width=1.5),
                    name=f'{name}',
                    antialias=True,
                    skipFiniteCheck=True
                )
                self.curves[sensor_name].append(curve)

            self.plots[sensor_name] = p

            if row < len(self.sensors)-1:
                self.win.nextRow()

    def update(self):
        try:
            data = self.board_shim.get_current_board_data(self.window_size)
            if data.size > 0:
                for sensor_name, sensor_info in self.sensors.items():
                    for idx, channel in enumerate(sensor_info['channels']):
                        if channel < data.shape[0]:
                            channel_data = data[channel]
                            if len(channel_data) > 0:
                                if sensor_info['detrend'] and not np.allclose(channel_data, 0):
                                    DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
                                self.curves[sensor_name][idx].setData(channel_data)
        except Exception as e:
            print(f"Update error: {e}")

def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = BrainFlowInputParams()
    params.ip_address = '192.168.229.255'
    params.ip_port = 3132
    params.timeout = 15

    board_shim = None
    try:
        board_shim = BoardShim(BoardIds.EMOTIBIT_BOARD, params)
        board_shim.prepare_session()
        board_shim.config_board('{"preset":"0"}')  # Start with DEFAULT preset
        board_shim.start_stream(65536)

        Graph(board_shim)

    except Exception as e:
        logging.error(f'Error: {str(e)}', exc_info=True)
    finally:
        if board_shim and board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()

if __name__ == '__main__':
    main()
