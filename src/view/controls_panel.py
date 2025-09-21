from PyQt6 import QtWidgets, QtCore

class ControlsPanel(QtWidgets.QGroupBox):
    addRequested = QtCore.pyqtSignal(str)
    clearRequested = QtCore.pyqtSignal()
    resetRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Controls", parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        # 안내 레이블
        hint = QtWidgets.QLabel("Enter 3D point coordinates:")
        layout.addWidget(hint)

        # X, Y, Z 입력 필드들
        input_layout = QtWidgets.QGridLayout()
        
        # X 좌표
        input_layout.addWidget(QtWidgets.QLabel("X:"), 0, 0)
        self.input_x = QtWidgets.QLineEdit()
        self.input_x.setPlaceholderText("0.0")
        input_layout.addWidget(self.input_x, 0, 1)
        
        # Y 좌표
        input_layout.addWidget(QtWidgets.QLabel("Y:"), 1, 0)
        self.input_y = QtWidgets.QLineEdit()
        self.input_y.setPlaceholderText("0.0")
        input_layout.addWidget(self.input_y, 1, 1)
        
        # Z 좌표
        input_layout.addWidget(QtWidgets.QLabel("Z:"), 2, 0)
        self.input_z = QtWidgets.QLineEdit()
        self.input_z.setPlaceholderText("0.0")
        input_layout.addWidget(self.input_z, 2, 1)
        
        layout.addLayout(input_layout)
        
        # Tab 키 순서 설정 (X → Y → Z → Add 버튼)
        self.input_x.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.input_y.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.input_z.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        # 버튼 영역
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add Point (Enter)")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_reset = QtWidgets.QPushButton("Reset View")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.btn_reset)
        layout.addLayout(btn_row)

        # 포인트 리스트(선택사항): 간단히 보여주기
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.list_widget, stretch=1)

        layout.addStretch(1)

        # 시그널 연결 (컨트롤러에 위임)
        self.btn_add.clicked.connect(self._emit_add_from_button)
        self.btn_clear.clicked.connect(self.clearRequested.emit)
        self.btn_reset.clicked.connect(self.resetRequested.emit)

        # Enter 입력으로도 Add (모든 입력 필드에서)
        self.input_x.returnPressed.connect(self._emit_add_from_button)
        self.input_y.returnPressed.connect(self._emit_add_from_button)
        self.input_z.returnPressed.connect(self._emit_add_from_button)

    # 외부(메인윈도우 단축키)에서 재사용할 수 있게 노출
    def emit_add(self):
        self._emit_add_from_button()

    def emit_clear(self):
        self.clearRequested.emit()

    def emit_reset(self):
        self.resetRequested.emit()

    def _emit_add_from_button(self):
        x = self.input_x.text().strip()
        y = self.input_y.text().strip()
        z = self.input_z.text().strip()
        text = f"{x},{y},{z}"
        self.addRequested.emit(text)

    # 리스트 관리용 (컨트롤러에서 호출)
    def add_list_item(self, x, y, z):
        self.list_widget.addItem(f"({x}, {y}, {z})")

    def clear_list(self):
        self.list_widget.clear()
    
    def clear_input(self):
        """입력 필드들을 비웁니다."""
        self.input_x.clear()
        self.input_y.clear()
        self.input_z.clear()
        self.input_x.setFocus()  # X 필드에 포커스 설정