from PyQt6 import QtWidgets
from model.points_model import PointsModel
from view.plot3d_view import Plot3DView
from view.controls_panel import ControlsPanel
from utils.parser import parse_xyz
from utils.validators import validate_xyz

class MainController:
    def __init__(self,
                 model: PointsModel,
                 view: Plot3DView,
                 panel: ControlsPanel,
                 status: QtWidgets.QStatusBar):
        self.model = model
        self.view = view
        self.panel = panel
        self.status = status

        # 패널 시그널 → 컨트롤러 슬롯
        panel.addRequested.connect(self._on_add_text)
        panel.clearRequested.connect(self._on_clear)
        panel.resetRequested.connect(self._on_reset)

        # 초기 상태
        self._on_reset()  # 카메라 초기화 메시지는 표시하지 않음

    # --- 슬롯들 ---
    def _on_add_text(self, text: str):
        try:
            p = parse_xyz(text)
            validate_xyz(p)
        except Exception as e:
            self._set_status(str(e), error=True)
            return

        # 모델 업데이트
        self.model.add(*p)

        # 뷰 업데이트
        self.view.add_point(*p)

        # 패널 리스트 업데이트
        self.panel.add_list_item(*p)

        # 입력 필드 비우기
        self.panel.clear_input()

        self._set_status(f"Added point {p}", error=False)

    def _on_clear(self):
        self.model.clear()
        self.view.clear_points()
        self.panel.clear_list()
        self._set_status("Cleared all points.", error=False)

    def _on_reset(self):
        self.view.reset_view()
        # 모델이나 리스트는 그대로 두고 카메라만 초기화
        self._set_status("Reset view.", error=False)

    def _set_status(self, msg: str, error: bool = False):
        self.status.showMessage(msg)
        if error:
            # 간단한 팝업도 함께 (선택)
            QtWidgets.QMessageBox.warning(self.view, "Invalid Input", msg)
