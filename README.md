# Point Plotter 3D

A simple **practice project** for visualizing and interacting with 3D points and arrows using [PyQt6](https://pypi.org/project/PyQt6/) and [pyqtgraph](http://www.pyqtgraph.org/).  
This repository is for learning purposes and **does not contain the built executable** in the root.  
You can download the Windows `.exe` build from Google Drive:

👉 **Download (Windows .exe):** https://drive.google.com/file/d/1s21M414e9zD_ttY-jvzdzKSmp-1p-1c0/view?usp=drive_link

---

## ✨ Features

- Plot 3D points and connect them with arrow vectors
- Auto-scaling grid and axes (expand dynamically with data)
- Hover detection with tooltip and highlight
- Camera controls (rotate, pan, zoom)
- Reset view and auto-zoom-to-fit
- Packaged build available as `.exe` (see link above)

---

## 📂 Project Structure

```
point-plotter3d/
├─ assets/
│  └─ app.ico
├─ src/
│  ├─ controller/
│  │  └─ main_controller.py
│  ├─ model/
│  │  └─ points_model.py
│  ├─ utils/
│  │  ├─ parser.py
│  │  └─ validators.py
│  ├─ view/
│  │  ├─ controls_panel.py
│  │  └─ plot3d_view.py
│  ├─ app_main.py
│  └─ ui_main_window.py
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt
```

---

## 🛠️ Requirements

Create a virtual environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyQt6
- pyqtgraph
- numpy

---

## ▶️ Run (Source)

```bash
python src/app_main.py
```

---

## 🧱 Build (Windows, optional)

If you want to build your own `.exe` locally (instead of using the Drive link), use **PyInstaller**:

**PowerShell (multi-line with backticks):**

```powershell
pyinstaller `
  --onefile `
  --windowed `
  --name "3dPlotter" `
  --icon "assets\app.ico" `
  --add-data "assets\app.ico;assets" `
  src\app_main.py
```

The executable will be placed under `dist/3dPlotter.exe`.

> **Note:** In one-file mode, `--add-data` paths are extracted to a temporary folder at runtime.  
> On macOS/Linux, use a colon (`:`) instead of semicolon (`;`) in `--add-data`.

---

## 🎯 Purpose

This project is primarily for **practice and learning**:
- Exploring PyQt6 GUI/event handling
- Understanding 3D rendering with pyqtgraph
- Experimenting with packaging via PyInstaller

Not intended for production use.

---

## 📜 License

See [LICENSE](LICENSE). Feel free to fork and experiment.
