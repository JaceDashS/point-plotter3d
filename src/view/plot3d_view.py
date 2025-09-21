from PyQt6 import QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QColor
import numpy as np

class Plot3DView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._gl = gl.GLViewWidget()
        self._gl.opts["distance"] = 30  # 초기 카메라 줌 거리
        self._gl.setBackgroundColor("k")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._gl)

        # 좌표축/그리드 초기화
        self._init_axes_and_grid()

        # 포인트 산포 아이템
        self._scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=8, color=(0.1, 0.2, 0.8, 1.0))
        self._gl.addItem(self._scatter)

        # 내부 상태
        self._points = np.zeros((0, 3))

    def _init_axes_and_grid(self):
        # 그리드 (XY, XZ, YZ)
        grid_xy = gl.GLGridItem()
        grid_xy.setSize(20, 20)
        grid_xy.setSpacing(1, 1)
        grid_xy.translate(0, 0, 0)
        self._gl.addItem(grid_xy)

        grid_xz = gl.GLGridItem()
        grid_xz.setSize(20, 20)
        grid_xz.setSpacing(1, 1)
        grid_xz.rotate(90, 1, 0, 0)  # XZ 평면
        self._gl.addItem(grid_xz)

        grid_yz = gl.GLGridItem()
        grid_yz.setSize(20, 20)
        grid_yz.setSpacing(1, 1)
        grid_yz.rotate(90, 0, 1, 0)  # YZ 평면
        self._gl.addItem(grid_yz)

        # 축 라인 (X:빨강, Y:초록, Z:파랑)
        def _axis_line(start, end, color):
            plt = gl.GLLinePlotItem(pos=np.array([start, end]), color=color, width=2, antialias=True)
            self._gl.addItem(plt)

        _axis_line([-10, 0, 0], [10, 0, 0], (1, 0, 0, 1))
        _axis_line([0, -10, 0], [0, 10, 0], (0, 0.7, 0, 1))
        _axis_line([0, 0, -10], [0, 0, 10], (0, 0, 1, 1))

    # 컨트롤러에서 호출
    def reset_view(self):
        self._gl.opts["distance"] = 30
        self._gl.opts["elevation"] = 30
        self._gl.opts["azimuth"] = -45

    def clear_points(self):
        self._points = np.zeros((0, 3))
        self._scatter.setData(pos=self._points)

    def add_point(self, x: float, y: float, z: float):
        p = np.array([[x, y, z]])
        self._points = np.vstack([self._points, p])
        self._scatter.setData(pos=self._points)

    def set_points(self, points_np: np.ndarray):
        """모델 전체 동기화용 (Nx3)"""
        self._points = np.array(points_np, dtype=float)
        self._scatter.setData(pos=self._points)
