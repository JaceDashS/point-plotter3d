from PyQt6 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QVector3D
import numpy as np
import math


class Plot3DView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._gl = gl.GLViewWidget()
        self._gl.opts["distance"] = 30  # 초기 카메라 줌 거리
        self._gl.setBackgroundColor("k")
        self._gl.setMouseTracking(True)  # 호버링을 위해 필요

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._gl)

        # GL 아이템 컨테이너
        self._arrows = []          # 모든 화살표 GL 아이템(몸체+머리) 평면 리스트
        self._arrow_groups = []    # 각 화살표 묶음 {"items":[...], "color":(r,g,b,a)}
        self._hover_idx = None     # 하이라이트 중인 화살표 data_index
        self._grid_items = []      # 그리드 GL 아이템
        self._axis_items = []      # 축 GL 아이템
        self._text_items = []      # 텍스트/마커 GL 아이템

        # 호버 툴팁
        self._tooltip = None

        # 색상 팔레트
        self._color_palette = self._generate_color_palette(20)

        # 데이터 및 피킹 세그먼트
        self._points = np.zeros((0, 3))
        # (start_xyz, end_xyz, data_index) — data_index는 "끝점" 인덱스
        self._segments = []

        # 좌표축/그리드 초기화
        self._init_axes_and_grid()

        # 마우스 이벤트 훅
        self._gl.mousePressEvent = self._mouse_press_event
        self._gl.mouseMoveEvent  = self._mouse_move_event

    # -------------------------------
    # 초기 렌더 구성
    # -------------------------------
    def _init_axes_and_grid(self):
        init_size, init_spacing = 20, 1
        self._create_grids(init_size, init_spacing)
        self._create_axes(init_size)  # 축 최초 생성

    def _generate_color_palette(self, num_colors: int):
        import colorsys
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            saturation = 0.7 + (i % 3) * 0.1
            lightness = 0.5 + (i % 2) * 0.2
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append((r, g, b, 1.0))
        return colors

    # -------------------------------
    # 그리드
    # -------------------------------
    def _create_grids(self, size, spacing):
        self._clear_grids()

        grid_xy = gl.GLGridItem()
        grid_xy.setSize(size, size)
        grid_xy.setSpacing(spacing, spacing)
        grid_xy.translate(0, 0, 0)
        self._gl.addItem(grid_xy)
        self._grid_items.append(grid_xy)

        grid_xz = gl.GLGridItem()
        grid_xz.setSize(size, size)
        grid_xz.setSpacing(spacing, spacing)
        grid_xz.rotate(90, 1, 0, 0)
        self._gl.addItem(grid_xz)
        self._grid_items.append(grid_xz)

        grid_yz = gl.GLGridItem()
        grid_yz.setSize(size, size)
        grid_yz.setSpacing(spacing, spacing)
        grid_yz.rotate(90, 0, 1, 0)
        self._gl.addItem(grid_yz)
        self._grid_items.append(grid_yz)

    def _clear_grids(self):
        for g in self._grid_items:
            self._gl.removeItem(g)
        self._grid_items.clear()

    # -------------------------------
    # 축 (Axes)
    # -------------------------------
    def _clear_axes(self):
        for it in self._axis_items:
            self._gl.removeItem(it)
        self._axis_items.clear()

    def _create_axes(self, size):
        self._clear_axes()
        half = float(size) / 2.0

        def _axis_line(start, end, color):
            item = gl.GLLinePlotItem(pos=np.array([start, end]), color=color, width=2, antialias=True)
            self._gl.addItem(item)
            self._axis_items.append(item)

        # X(빨강), Y(초록), Z(파랑)
        _axis_line([-half, 0, 0], [half, 0, 0], (1, 0, 0, 1))
        _axis_line([0, -half, 0], [0, half, 0], (0, 0.7, 0, 1))
        _axis_line([0, 0, -half], [0, 0, half], (0, 0, 1, 1))

    # -------------------------------
    # 그리드/축 업데이트
    # -------------------------------
    def _update_grids_for_data(self):
        """첫 점부터 확장되도록 원점 기준 최대 절대값(extent)으로 크기 계산"""
        if len(self._points) == 0:
            return

        min_x = float(np.min(self._points[:, 0])); max_x = float(np.max(self._points[:, 0]))
        min_y = float(np.min(self._points[:, 1])); max_y = float(np.max(self._points[:, 1]))
        min_z = float(np.min(self._points[:, 2])); max_z = float(np.max(self._points[:, 2]))

        extent = max(
            abs(min_x), abs(max_x),
            abs(min_y), abs(max_y),
            abs(min_z), abs(max_z)
        )

        half_range = max(10.0, extent * 1.2)              # 여유 20%
        grid_size  = int(math.ceil(half_range * 2.0))     # 한 변 길이
        grid_spacing = max(1, int(max(1.0, half_range / 10.0)))

        self._create_grids(grid_size, grid_spacing)
        self._create_axes(grid_size)

    # -------------------------------
    # 텍스트 라벨 (현재 미사용)
    # -------------------------------
    def _clear_text_labels(self):
        for t in self._text_items:
            self._gl.removeItem(t)
        self._text_items.clear()

    def _update_text_labels(self):
        pass

    # -------------------------------
    # 마우스 이벤트 (픽셀기반 피킹)
    # -------------------------------
    def _mouse_press_event(self, ev: QtGui.QMouseEvent):
        gl.GLViewWidget.mousePressEvent(self._gl, ev)

    def _mouse_move_event(self, ev: QtGui.QMouseEvent):
        # 기본 동작(회전/팬) 유지
        gl.GLViewWidget.mouseMoveEvent(self._gl, ev)

        if not self._segments:
            self._hide_tooltip()
            self._set_hover_group(None)
            return

        # 마우스 물리 픽셀 좌표
        try:
            dpr = float(self._gl.devicePixelRatioF())
        except Exception:
            dpr = 1.0
        mx_px = float(ev.position().x() * dpr)
        my_px = float(ev.position().y() * dpr)

        PIX_THRESH = 10.0  # 픽셀 기준 임계값

        best_dist_px, best_idx = 1e9, None
        for (p0, p1, data_idx) in self._segments:
            dpx = self._screen_segment_distance(self._gl, p0, p1, mx_px, my_px)
            if dpx < best_dist_px:
                best_dist_px, best_idx = dpx, data_idx

        if best_idx is not None and best_dist_px <= PIX_THRESH:
            dp = self._points[best_idx]
            self._show_tooltip(ev.pos(), f"({dp[0]:.6g}, {dp[1]:.6g}, {dp[2]:.6g})")
            self._set_hover_group(best_idx)
        else:
            self._hide_tooltip()
            self._set_hover_group(None)

    # -------------------------------
    # 하이라이트
    # -------------------------------
    def _lighter_color(self, color, gain=1.35):
        r, g, b, a = color
        return (min(r * gain, 1.0), min(g * gain, 1.0), min(b * gain, 1.0), a)

    def _apply_group_color(self, group_idx, color):
        if group_idx is None or not (0 <= group_idx < len(self._arrow_groups)):
            return
        for item in self._arrow_groups[group_idx]["items"]:
            try:
                item.setData(color=color)
            except Exception:
                try:
                    item.color = color
                    item.update()
                except Exception:
                    pass

    def _set_hover_group(self, new_idx):
        # 이전 하이라이트 복원
        if self._hover_idx is not None and 0 <= self._hover_idx < len(self._arrow_groups):
            base = self._arrow_groups[self._hover_idx]["color"]
            self._apply_group_color(self._hover_idx, base)
        # 새 하이라이트
        self._hover_idx = new_idx
        if new_idx is not None and 0 <= new_idx < len(self._arrow_groups):
            base = self._arrow_groups[new_idx]["color"]
            self._apply_group_color(new_idx, self._lighter_color(base, gain=1.45))

    # -------------------------------
    # 툴팁
    # -------------------------------
    def _show_tooltip(self, mouse_pos: QtCore.QPointF, text: str):
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
        gp = self._gl.mapToGlobal(QtCore.QPoint(int(mouse_pos.x()), int(mouse_pos.y())))
        self._tooltip.move(gp.x() + 12, gp.y() - 30)
        self._tooltip.show()

    def _hide_tooltip(self):
        if self._tooltip is not None:
            self._tooltip.hide()

    # -------------------------------
    # 카메라/투영/피킹 유틸
    # -------------------------------
    def _center_to_np(self, c) -> np.ndarray:
        if isinstance(c, QVector3D):
            return np.array([c.x(), c.y(), c.z()], float)
        try:
            a, b, d = c
            return np.array([float(a), float(b), float(d)], float)
        except Exception:
            return np.array([0.0, 0.0, 0.0], float)

    def _compute_camera_axes(self, glv: gl.GLViewWidget):
        """실제 카메라 위치 기반 전방/우/상 벡터 계산"""
        cam_qv = glv.cameraPosition()
        cam = np.array([cam_qv.x(), cam_qv.y(), cam_qv.z()], float)
        center = self._center_to_np(glv.opts['center'])

        fwd = center - cam
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 0 else np.array([0.0, 0.0, -1.0], float)

        up_world = np.array([0.0, 0.0, 1.0], float)
        right = np.cross(fwd, up_world)
        rn = np.linalg.norm(right)
        if rn < 1e-9:
            right = np.cross(fwd, np.array([1.0, 0.0, 0.0], float))
            rn = np.linalg.norm(right)
            if rn < 1e-9:
                right = np.array([1.0, 0.0, 0.0], float)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        up = up / np.linalg.norm(up)
        return fwd, right, up

    def _project_point_to_pixels(self, glv, p3):
        # 1) 위젯 크기
        try:
            dpr = float(glv.devicePixelRatioF())
        except Exception:
            dpr = 1.0
        w = int(round(max(1, glv.width()) * dpr))
        h = int(round(max(1, glv.height()) * dpr))
        aspect = w / max(1.0, float(h))

        # 2) 카메라 축
        fwd, right, up = self._compute_camera_axes(glv)
        cam_qv = glv.cameraPosition()
        cam = np.array([cam_qv.x(), cam_qv.y(), cam_qv.z()], float)

        v = p3 - cam
        z_cam = float(np.dot(v, fwd))
        if z_cam <= 1e-6:
            return None
        x_cam = float(np.dot(v, right))
        y_cam = float(np.dot(v, up))

        # 3) 수평 FOV 사용 (GLViewWidget의 fov는 horizontal)
        fov_deg = float(glv.opts.get('fov', 60.0))
        tan_half_x = math.tan(math.radians(fov_deg) * 0.5)
        tan_half_y = tan_half_x / aspect   # ★ 핵심 수정: 세로는 가로FOV/종횡비

        ndc_x = (x_cam / z_cam) / tan_half_x
        ndc_y = (y_cam / z_cam) / tan_half_y

        px = (ndc_x * 0.5 + 0.5) * w
        py = (1.0 - (ndc_y * 0.5 + 0.5)) * h
        return (px, py, z_cam)

    def _screen_segment_distance(self, glv, p0, p1, mx_px, my_px):
        """화면 픽셀 공간에서 점과 3D 선분의 최소거리(px)"""
        a = self._project_point_to_pixels(glv, p0)
        b = self._project_point_to_pixels(glv, p1)
        if a is None or b is None:
            return float('inf')

        ax, ay, _ = a; bx, by, _ = b
        vx, vy = bx - ax, by - ay
        wx, wy = mx_px - ax, my_px - ay
        seg_len2 = vx*vx + vy*vy
        if seg_len2 <= 1e-9:
            return math.hypot(wx, wy)
        t = max(0.0, min(1.0, (wx*vx + wy*vy) / seg_len2))
        cx, cy = ax + vx * t, ay + vy * t
        
        # 기본 거리 계산
        base_distance = math.hypot(mx_px - cx, my_px - cy)
        
        # 끝점 근처에서 호버링을 더 쉽게 하기 위해 가중치 적용
        # t가 0.7 이상 (끝점에 가까울 때) 거리를 줄임
        if t >= 0.7:
            # 끝점 근처에서는 호버링 영역을 2배 확대
            return base_distance * 0.5
        else:
            # 시작점 근처에서는 호버링을 어렵게 함
            return base_distance * 1.5

    # (보조: 현재 미사용이지만 남겨둠)
    def _ray_from_mouse(self, glv: gl.GLViewWidget, ev: QtGui.QMouseEvent):
        try:
            dpr = float(self.devicePixelRatioF())
        except Exception:
            dpr = 1.0
        w = int(round(max(1, glv.width()) * dpr))
        h = int(round(max(1, glv.height()) * dpr))
        mx = float(ev.position().x() * dpr)
        my = float(ev.position().y() * dpr)

        ndc_x = (mx / w - 0.5) * 2.0
        ndc_y = (1.0 - my / h - 0.5) * 2.0

        fov = float(glv.opts.get('fov', 60.0))
        aspect = w / max(1.0, float(h))
        tan_half = math.tan(math.radians(fov) * 0.5)
        sx = ndc_x * tan_half * aspect
        sy = ndc_y * tan_half

        fwd, right, up = self._compute_camera_axes(glv)
        cam_qv = glv.cameraPosition()
        cam_pos = np.array([cam_qv.x(), cam_qv.y(), cam_qv.z()], float)

        dir_world = fwd + right * sx + up * sy
        n = np.linalg.norm(dir_world)
        dir_world = dir_world / n if n > 0 else fwd
        cam_dist = float(glv.opts.get('distance', 10.0))
        return cam_pos, dir_world, cam_dist

    @staticmethod
    def _segment_distance(p0, p1, o, d):
        """선분 p0->p1 과 레이 (o + s d)의 최소거리 (보조)"""
        v = p1 - p0
        w0 = p0 - o
        a = float(np.dot(v, v))
        b = float(np.dot(v, d))
        c = float(np.dot(d, d))
        d0 = float(np.dot(v, w0))
        e0 = float(np.dot(d, w0))
        denom = a * c - b * b
        if denom < 1e-12:
            t = 0.0
            s = max(0.0, -e0 / max(1e-12, c))
        else:
            t = (b * e0 - c * d0) / denom
            s = (a * e0 - b * d0) / denom
            t = min(1.0, max(0.0, t))
            s = max(0.0, s)
        cp_seg = p0 + v * t
        cp_ray = o + d * s
        return float(np.linalg.norm(cp_seg - cp_ray))

    # -------------------------------
    # 뷰/경계/줌
    # -------------------------------
    def _calculate_bounds(self):
        if len(self._points) == 0:
            return None
        min_x = np.min(self._points[:, 0]); max_x = np.max(self._points[:, 0])
        min_y = np.min(self._points[:, 1]); max_y = np.max(self._points[:, 1])
        min_z = np.min(self._points[:, 2]); max_z = np.max(self._points[:, 2])
        return {
            'min_x': float(min_x), 'max_x': float(max_x),
            'min_y': float(min_y), 'max_y': float(max_y),
            'min_z': float(min_z), 'max_z': float(max_z)
        }

    def _auto_zoom_to_fit(self):
        b = self._calculate_bounds()
        if b is None:
            return
        maxd = 0.0
        for p in self._points:
            maxd = max(maxd, float(np.linalg.norm(p)))
        if maxd < 1.0:
            maxd = 10.0
        margin = maxd * 0.3
        view_distance = maxd + margin
        self._gl.opts['distance'] = view_distance * 2.0
        self._gl.opts['center'] = QVector3D(0, 0, 0)
        self._update_grids_for_data()

    def reset_view(self):
        self._gl.opts["distance"] = 30
        self._gl.opts["elevation"] = 30
        self._gl.opts["azimuth"] = -45

    # -------------------------------
    # 화살표 생성/갱신/클리어
    # -------------------------------
    def _create_arrow(self, sx, sy, sz, ex, ey, ez, color=(0.1, 0.2, 0.8, 1.0)):
        """
        시작점에서 끝점으로 향하는 화살표를 생성.
        반환: [GLLinePlotItem(몸체), GLLinePlotItem(머리1), ...]
        """
        items = []

        # 몸체
        body = gl.GLLinePlotItem(
            pos=np.array([[sx, sy, sz], [ex, ey, ez]]),
            color=color, width=3, antialias=True
        )
        self._gl.addItem(body)
        items.append(body)

        # 머리
        dx, dy, dz = ex - sx, ey - sy, ez - sz
        L = float(np.sqrt(dx*dx + dy*dy + dz*dz))
        if L > 0:
            head = max(0.5, L * 0.1)
            d = np.array([dx, dy, dz], float) / L
            perp = np.array([0, 0, 1], float) if abs(d[2]) < 0.9 else np.array([1, 0, 0], float)
            perp = perp - np.dot(perp, d) * d
            perp = perp / np.linalg.norm(perp)
            perp2 = np.cross(d, perp)
            for i in range(4):
                ang = i * np.pi / 2
                hp = np.array([ex, ey, ez], float) - d * head + (perp * np.cos(ang) + perp2 * np.sin(ang)) * head * 0.3
                tip = gl.GLLinePlotItem(pos=np.array([[ex, ey, ez], hp]), color=color, width=2, antialias=True)
                self._gl.addItem(tip)
                items.append(tip)

        return items

    def clear_points(self):
        """모든 화살표/보조표시 제거 후 초기 상태로 복귀."""
        # 모든 화살표(GL 아이템) 제거
        for it in self._arrows:
            try:
                self._gl.removeItem(it)
            except Exception:
                pass
        self._arrows.clear()

        # 그룹/하이라이트 초기화
        self._arrow_groups.clear()
        self._hover_idx = None

        # 텍스트/마커 제거
        self._clear_text_labels()

        # 툴팁/세그먼트/데이터 초기화
        if self._tooltip is not None:
            self._tooltip.hide()
            self._tooltip = None
        self._segments.clear()
        self._points = np.zeros((0, 3))

        # 뷰/그리드/축 초기화
        self.reset_view()
        self._gl.opts['center'] = QVector3D(0, 0, 0)
        self._create_grids(20, 1)
        self._create_axes(20)

    def add_point(self, x: float, y: float, z: float):
        """새로운 화살표(이전점→현재점)를 추가. 첫 점은 원점→현재점."""
        p = np.array([[x, y, z]], float)
        self._points = np.vstack([self._points, p])

        color = self._color_palette[len(self._arrows) % len(self._color_palette)]

        if len(self._points) == 1:
            sx, sy, sz = 0.0, 0.0, 0.0
        else:
            prev = self._points[-2]
            sx, sy, sz = prev[0], prev[1], prev[2]

        items = self._create_arrow(sx, sy, sz, x, y, z, color)
        self._arrows.extend(items)
        self._arrow_groups.append({"items": items, "color": color})

        # 피킹 세그먼트(끝점 = 방금 추가된 데이터 인덱스)
        self._segments.append((np.array([sx, sy, sz], float), np.array([x, y, z], float), len(self._points) - 1))

        self._auto_zoom_to_fit()

    def set_points(self, points_np: np.ndarray):
        """모델 전체 동기화용 (Nx3) - 모든 화살표를 새로 그림."""
        self.clear_points()
        self._points = np.array(points_np, float)
        for i, p in enumerate(self._points):
            if len(p) == 3:
                color = self._color_palette[i % len(self._color_palette)]
                if i == 0:
                    sx, sy, sz = 0.0, 0.0, 0.0
                else:
                    prev = self._points[i - 1]
                    sx, sy, sz = prev[0], prev[1], prev[2]
                items = self._create_arrow(sx, sy, sz, p[0], p[1], p[2], color)
                self._arrows.extend(items)
                self._arrow_groups.append({"items": items, "color": color})
                self._segments.append((np.array([sx, sy, sz], float), np.array([p[0], p[1], p[2]], float), i))
        self._auto_zoom_to_fit()
