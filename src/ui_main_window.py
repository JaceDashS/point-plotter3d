from PyQt6 import QtWidgets, QtCore, QtGui
from view.plot3d_view import Plot3DView
from view.controls_panel import ControlsPanel
from controller.main_controller import MainController
from model.points_model import PointsModel


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Point Plotter 3D")
        self.resize(1100, 700)

        # 중앙 위젯 + 레이아웃
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 왼쪽: 3D 뷰
        self.plot_view = Plot3DView(self)

        # 오른쪽: 컨트롤 패널
        self.controls = ControlsPanel(self)

        # 레이아웃 배치
        layout.addWidget(self.plot_view, stretch=5)
        layout.addWidget(self.controls, stretch=2)

        # 상태바
        self.status = self.statusBar()
        self.status.showMessage("Ready")

        # 모델 & 컨트롤러
        self.model = PointsModel()
        self.controller = MainController(
            model=self.model,
            view=self.plot_view,
            panel=self.controls,
            status=self.status
        )

        # 단축키: Enter로 Add, Ctrl+R Reset, Ctrl+L Clear
        add_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Return"), self)
        add_shortcut.activated.connect(self.controls.emit_add)

        reset_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        reset_shortcut.activated.connect(self.controls.emit_reset)

        clear_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+L"), self)
        clear_shortcut.activated.connect(self.controls.emit_clear)
