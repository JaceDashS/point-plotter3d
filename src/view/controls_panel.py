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
        hint = QtWidgets.QLabel("Enter 3D point as text:  x,y,z\n(e.g., 1.2, -3, 4)")
        layout.addWidget(hint)

        # 입력창
        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText("x,y,z")
        layout.addWidget(self.input_edit)

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

        # Enter 입력으로도 Add
        self.input_edit.returnPressed.connect(self._emit_add_from_button)

    # 외부(메인윈도우 단축키)에서 재사용할 수 있게 노출
    def emit_add(self):
        self._emit_add_from_button()

    def emit_clear(self):
        self.clearRequested.emit()

    def emit_reset(self):
        self.resetRequested.emit()

    def _emit_add_from_button(self):
        text = self.input_edit.text().strip()
        self.addRequested.emit(text)

    # 리스트 관리용 (컨트롤러에서 호출)
    def add_list_item(self, x, y, z):
        self.list_widget.addItem(f"({x}, {y}, {z})")

    def clear_list(self):
        self.list_widget.clear()
