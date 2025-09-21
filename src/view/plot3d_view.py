from PyQt6 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QColor, QVector3D
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

        # 화살표 아이템들을 저장할 리스트
        self._arrows = []
        
        # 그리드 아이템들을 저장할 리스트
        self._grid_items = []
        
        # 텍스트 아이템들을 저장할 리스트
        self._text_items = []
        
        # 색상 팔레트 (HSL 색상 공간에서 균등하게 분포)
        self._color_palette = self._generate_color_palette(20)

        # 내부 상태
        self._points = np.zeros((0, 3))

        # 좌표축/그리드 초기화
        self._init_axes_and_grid()
        
        # 마우스 이벤트 설정
        self._gl.mousePressEvent = self._mouse_press_event
        self._gl.mouseMoveEvent = self._mouse_move_event
        
        # 툴팁을 위한 변수
        self._tooltip = None

    def _init_axes_and_grid(self):
        # 그리드 (XY, XZ, YZ) - 기본 크기로 초기화
        self._create_grids(20, 1)

        # 축 라인 (X:빨강, Y:초록, Z:파랑)
        def _axis_line(start, end, color):
            plt = gl.GLLinePlotItem(pos=np.array([start, end]), color=color, width=2, antialias=True)
            self._gl.addItem(plt)

        _axis_line([-10, 0, 0], [10, 0, 0], (1, 0, 0, 1))
        _axis_line([0, -10, 0], [0, 10, 0], (0, 0.7, 0, 1))
        _axis_line([0, 0, -10], [0, 0, 10], (0, 0, 1, 1))

    def _generate_color_palette(self, num_colors: int):
        """HSL 색상 공간에서 균등하게 분포된 색상 팔레트를 생성합니다."""
        import colorsys
        
        colors = []
        for i in range(num_colors):
            # HSL 색상 공간에서 색상 생성
            # Hue: 0~1 (색조), Saturation: 0.7~1.0 (채도), Lightness: 0.5~0.8 (명도)
            hue = i / num_colors  # 0~1 사이의 균등 분포
            saturation = 0.7 + (i % 3) * 0.1  # 0.7, 0.8, 0.9 순환
            lightness = 0.5 + (i % 2) * 0.2   # 0.5, 0.7 순환
            
            # HSL을 RGB로 변환
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            
            # RGB 값을 0~1 범위로 변환하고 알파 채널 추가
            color = (rgb[0], rgb[1], rgb[2], 1.0)
            colors.append(color)
        
        return colors

    def _create_grids(self, size, spacing):
        """그리드를 생성합니다."""
        # 기존 그리드 제거
        self._clear_grids()
        
        # 새로운 그리드 생성 (XY, XZ, YZ)
        grid_xy = gl.GLGridItem()
        grid_xy.setSize(size, size)
        grid_xy.setSpacing(spacing, spacing)
        grid_xy.translate(0, 0, 0)
        self._gl.addItem(grid_xy)
        self._grid_items.append(grid_xy)

        grid_xz = gl.GLGridItem()
        grid_xz.setSize(size, size)
        grid_xz.setSpacing(spacing, spacing)
        grid_xz.rotate(90, 1, 0, 0)  # XZ 평면
        self._gl.addItem(grid_xz)
        self._grid_items.append(grid_xz)

        grid_yz = gl.GLGridItem()
        grid_yz.setSize(size, size)
        grid_yz.setSpacing(spacing, spacing)
        grid_yz.rotate(90, 0, 1, 0)  # YZ 평면
        self._gl.addItem(grid_yz)
        self._grid_items.append(grid_yz)

    def _clear_grids(self):
        """기존 그리드를 제거합니다."""
        for grid in self._grid_items:
            self._gl.removeItem(grid)
        self._grid_items.clear()

    def _update_grids_for_data(self):
        """데이터 범위에 맞게 그리드를 업데이트합니다."""
        bounds = self._calculate_bounds()
        if bounds is None:
            return
        
        # 데이터 범위 계산
        range_x = bounds['max_x'] - bounds['min_x']
        range_y = bounds['max_y'] - bounds['min_y']
        range_z = bounds['max_z'] - bounds['min_z']
        
        # 최대 범위 찾기
        max_range = max(range_x, range_y, range_z)
        
        # 최소 범위 보장
        if max_range < 1.0:
            max_range = 10.0
        
        # 그리드 크기와 간격 계산
        # 그리드 크기는 데이터 범위의 2배 + 여백
        grid_size = int(max_range * 2.5)
        # 그리드 간격은 데이터 범위에 비례하되 적당한 크기 유지
        grid_spacing = max(1, int(max_range / 10))
        
        # 그리드 업데이트
        self._create_grids(grid_size, grid_spacing)

    def _create_coordinate_marker(self, x: float, y: float, z: float, color=(1, 1, 1, 1)):
        """좌표 표시를 위한 마커를 생성합니다."""
        # 화살표 끝점에 작은 구체를 표시
        marker = gl.GLScatterPlotItem(
            pos=np.array([[x, y, z]]), 
            size=8, 
            color=color,
            pxMode=False  # 3D 공간 단위로 크기 설정
        )
        return marker

    def _clear_text_labels(self):
        """모든 좌표 마커를 제거합니다."""
        for marker in self._text_items:
            self._gl.removeItem(marker)
        self._text_items.clear()

    def _update_text_labels(self):
        """호버링을 위한 포인트 정보를 준비합니다."""
        # 이 메서드는 현재는 호버링 기능을 위해 비워둡니다
        pass

    def _mouse_press_event(self, event):
        """마우스 클릭 이벤트 처리"""
        # 기본 마우스 이벤트 처리
        gl.GLViewWidget.mousePressEvent(self._gl, event)

    def _mouse_move_event(self, event):
        """마우스 이동 이벤트 처리 - 호버링 감지"""
        # 기본 마우스 이벤트 처리
        gl.GLViewWidget.mouseMoveEvent(self._gl, event)
        
        # 마우스 위치에서 가장 가까운 포인트 찾기
        closest_point = self._find_closest_point(event.pos())
        
        if closest_point is not None:
            # 툴팁 표시
            self._show_tooltip(event.pos(), closest_point)
        else:
            # 툴팁 숨기기
            self._hide_tooltip()

    def _find_closest_point(self, mouse_pos):
        """마우스 위치에서 가장 가까운 포인트를 찾습니다."""
        if len(self._points) == 0:
            return None
        
        # 마우스 위치를 3D 좌표로 변환 (간단한 근사)
        # 실제로는 더 복잡한 변환이 필요하지만, 여기서는 간단히 처리
        min_distance = float('inf')
        closest_point = None
        closest_index = -1
        
        for i, point in enumerate(self._points):
            # 2D 화면 좌표로 투영 (간단한 근사)
            # 실제로는 OpenGL의 투영 행렬을 사용해야 함
            screen_x = point[0] * 10 + 400  # 간단한 스케일링
            screen_y = point[1] * 10 + 300
            
            distance = ((mouse_pos.x() - screen_x) ** 2 + (mouse_pos.y() - screen_y) ** 2) ** 0.5
            
            if distance < min_distance and distance < 50:  # 50픽셀 내
                min_distance = distance
                closest_point = point
                closest_index = i
        
        return (closest_point, closest_index) if closest_point is not None else None

    def _show_tooltip(self, mouse_pos, point_info):
        """툴팁을 표시합니다."""
        point, index = point_info
        text = f"Point {index + 1}: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})"
        
        # 툴팁이 이미 있다면 업데이트
        if self._tooltip is None:
            self._tooltip = QtWidgets.QLabel(text, self)
            self._tooltip.setStyleSheet("""
                QLabel {
                    background-color: rgba(0, 0, 0, 180);
                    color: white;
                    padding: 5px;
                    border-radius: 3px;
                    font-size: 12px;
                }
            """)
            self._tooltip.setWindowFlags(QtCore.Qt.WindowType.ToolTip)
        
        self._tooltip.setText(text)
        self._tooltip.move(mouse_pos.x() + 10, mouse_pos.y() - 30)
        self._tooltip.show()

    def _hide_tooltip(self):
        """툴팁을 숨깁니다."""
        if self._tooltip is not None:
            self._tooltip.hide()

    def _calculate_bounds(self):
        """모든 포인트의 경계를 계산합니다."""
        if len(self._points) == 0:
            return None
        
        # 모든 포인트의 min/max 계산
        min_x = np.min(self._points[:, 0])
        max_x = np.max(self._points[:, 0])
        min_y = np.min(self._points[:, 1])
        max_y = np.max(self._points[:, 1])
        min_z = np.min(self._points[:, 2])
        max_z = np.max(self._points[:, 2])
        
        return {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'min_z': min_z, 'max_z': max_z
        }

    def _auto_zoom_to_fit(self):
        """모든 포인트가 보이도록 줌 레벨을 자동으로 조정합니다."""
        bounds = self._calculate_bounds()
        if bounds is None:
            return
        
        # 원점에서 가장 먼 포인트까지의 거리 계산
        max_distance = 0
        for point in self._points:
            distance = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
            max_distance = max(max_distance, distance)
        
        # 최소 거리 보장 (너무 작으면 기본값 사용)
        if max_distance < 1.0:
            max_distance = 10.0
        
        # 여백 추가 (30% 여백)
        margin = max_distance * 0.3
        view_distance = max_distance + margin
        
        # 뷰 거리 설정 (원점 기준)
        self._gl.opts['distance'] = view_distance * 2.0
        
        # 카메라 중심은 항상 원점 유지
        self._gl.opts['center'] = QVector3D(0, 0, 0)
        
        # 그리드도 데이터 범위에 맞게 업데이트
        self._update_grids_for_data()
        
        # 좌표 마커도 업데이트
        self._update_text_labels()

    def _create_arrow(self, start_x: float, start_y: float, start_z: float, 
                     end_x: float, end_y: float, end_z: float, 
                     color=(0.1, 0.2, 0.8, 1.0)):
        """시작점에서 끝점으로 향하는 화살표를 생성합니다."""
        # 화살표의 몸체 (시작점에서 끝점까지)
        arrow_body = gl.GLLinePlotItem(
            pos=np.array([[start_x, start_y, start_z], [end_x, end_y, end_z]]), 
            color=color, 
            width=3, 
            antialias=True
        )
        
        # 화살표 머리 (끝점에서 약간 뒤로)
        dx = end_x - start_x
        dy = end_y - start_y
        dz = end_z - start_z
        arrow_length = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if arrow_length > 0:
            # 화살표 머리의 크기 (전체 길이의 10%)
            head_size = max(0.5, arrow_length * 0.1)
            
            # 화살표 머리 방향 벡터
            direction = np.array([dx, dy, dz]) / arrow_length
            
            # 화살표 머리의 끝점들 (4개 방향)
            head_points = []
            for i in range(4):
                angle = i * np.pi / 2
                # 화살표 머리를 위한 수직 벡터들
                if abs(direction[2]) < 0.9:  # Z축과 평행하지 않은 경우
                    perp1 = np.array([0, 0, 1])
                else:
                    perp1 = np.array([1, 0, 0])
                
                perp1 = perp1 - np.dot(perp1, direction) * direction
                perp1 = perp1 / np.linalg.norm(perp1)
                perp2 = np.cross(direction, perp1)
                
                head_point = np.array([end_x, end_y, end_z]) - direction * head_size + \
                            (perp1 * np.cos(angle) + perp2 * np.sin(angle)) * head_size * 0.3
                head_points.append(head_point)
            
            # 화살표 머리 그리기
            for head_point in head_points:
                arrow_head = gl.GLLinePlotItem(
                    pos=np.array([[end_x, end_y, end_z], head_point]), 
                    color=color, 
                    width=2, 
                    antialias=True
                )
                self._gl.addItem(arrow_head)
        
        self._gl.addItem(arrow_body)
        return arrow_body

    # 컨트롤러에서 호출
    def reset_view(self):
        self._gl.opts["distance"] = 30
        self._gl.opts["elevation"] = 30
        self._gl.opts["azimuth"] = -45

    def clear_points(self):
        """모든 화살표를 제거합니다."""
        for arrow in self._arrows:
            self._gl.removeItem(arrow)
        self._arrows.clear()
        
        # 텍스트 라벨도 제거
        self._clear_text_labels()
        
        self._points = np.zeros((0, 3))
        
        # 뷰를 초기 상태로 리셋
        self.reset_view()
        
        # 그리드도 초기 상태로 리셋
        self._create_grids(20, 1)

    def add_point(self, x: float, y: float, z: float):
        """새로운 화살표를 추가합니다."""
        p = np.array([[x, y, z]])
        self._points = np.vstack([self._points, p])
        
        # 색상 팔레트에서 색상 선택 (화살표 개수에 따라)
        color_index = len(self._arrows) % len(self._color_palette)
        color = self._color_palette[color_index]
        
        # 시작점 결정: 첫 번째 화살표는 원점에서, 나머지는 이전 포인트에서
        if len(self._points) == 1:
            # 첫 번째 화살표는 원점에서 시작
            start_x, start_y, start_z = 0.0, 0.0, 0.0
        else:
            # 나머지 화살표는 이전 포인트에서 시작
            prev_point = self._points[-2]  # 현재 포인트 바로 이전
            start_x, start_y, start_z = prev_point[0], prev_point[1], prev_point[2]
        
        # 새로운 화살표 생성
        arrow = self._create_arrow(start_x, start_y, start_z, x, y, z, color)
        self._arrows.append(arrow)
        
        # 자동 줌으로 모든 포인트가 보이도록 조정
        self._auto_zoom_to_fit()
        
        # 좌표 마커 업데이트
        self._update_text_labels()

    def set_points(self, points_np: np.ndarray):
        """모델 전체 동기화용 (Nx3) - 모든 화살표를 새로 그립니다."""
        # 기존 화살표들 제거
        self.clear_points()
        
        # 새로운 포인트들로 화살표 생성
        self._points = np.array(points_np, dtype=float)
        for i, point in enumerate(self._points):
            if len(point) == 3:
                # 색상 팔레트에서 색상 선택
                color_index = i % len(self._color_palette)
                color = self._color_palette[color_index]
                
                # 시작점 결정: 첫 번째 화살표는 원점에서, 나머지는 이전 포인트에서
                if i == 0:
                    # 첫 번째 화살표는 원점에서 시작
                    start_x, start_y, start_z = 0.0, 0.0, 0.0
                else:
                    # 나머지 화살표는 이전 포인트에서 시작
                    prev_point = self._points[i-1]
                    start_x, start_y, start_z = prev_point[0], prev_point[1], prev_point[2]
                
                arrow = self._create_arrow(start_x, start_y, start_z, 
                                         point[0], point[1], point[2], color)
                self._arrows.append(arrow)
        
        # 자동 줌으로 모든 포인트가 보이도록 조정
        self._auto_zoom_to_fit()
        
        # 텍스트 라벨 업데이트
        self._update_text_labels()
