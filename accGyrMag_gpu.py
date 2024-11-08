import logging
import numpy as np
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from PyQt5.QtOpenGL import QGLFormat

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim

        # Configure OpenGL format
        fmt = QGLFormat()
        fmt.setSampleBuffers(True)
        fmt.setSwapInterval(1)
        QGLFormat.setDefaultFormat(fmt)

        # Enable hardware acceleration
        pg.setConfigOptions(
            antialias=True,
            useOpenGL=True,
            enableExperimental=True
        )

        # Channel configuration grouped by sensor (using channels 1-9)
        self.sensors = {
            'Accelerometer': {
                'channels': [1, 2, 3],
                'colors': ['r', 'g', 'b'],
                'names': ['X', 'Y', 'Z']
            },
            'Gyroscope': {
                'channels': [4, 5, 6],
                'colors': ['r', 'g', 'b'],
                'names': ['X', 'Y', 'Z']
            },
            'Magnetometer': {
                'channels': [7, 8, 9],
                'colors': ['r', 'g', 'b'],
                'names': ['X', 'Y', 'Z']
            }
        }

        # Display configuration
        self.sampling_rate = 100
        self.update_speed_ms = 16  # ~60 FPS
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        print(f"Window size: {self.window_size}s at {self.sampling_rate}Hz")

        # Setup GUI
        self.app = QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title='EmotiBit IMU Data (GPU Accelerated)')
        self.win.resize(1200, 800)
        self.win.setBackground('w')

        self._init_timeseries()

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

        self.win.show()
        self.app.exec_()

    def _init_timeseries(self):
        self.plots = {}
        self.curves = {}

        for row, (sensor_name, sensor_info) in enumerate(self.sensors.items()):
            # Create plot for this sensor
            p = self.win.addPlot(row=row, col=0)
            p.setDownsampling(auto=True, mode='peak')
            p.setClipToView(True)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel('left', sensor_name)
            p.setLabel('bottom', 'Time (s)' if row == len(self.sensors)-1 else '')
            p.getAxis('left').setPen('k')
            p.getAxis('bottom').setPen('k')
            p.addLegend()

            # Enable hardware acceleration for ViewBox
            vb = p.getViewBox()
            vb.setAspectLocked(False)
            vb.enableAutoRange(axis='y')

            # Create GPU-accelerated curves
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
            data = self.board_shim.get_current_board_data(self.num_points)
            if data.size > 0:
                for sensor_name, sensor_info in self.sensors.items():
                    for idx, channel in enumerate(sensor_info['channels']):
                        if channel < data.shape[0]:
                            channel_data = data[channel]
                            if len(channel_data) > 0:
                                if not np.allclose(channel_data, 0):
                                    DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
                                time_axis = np.linspace(0, self.window_size, len(channel_data))
                                self.curves[sensor_name][idx].setData(
                                    time_axis,
                                    channel_data,
                                    connect='finite'
                                )
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
        board_shim.config_board('{"preset":"0"}')
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
