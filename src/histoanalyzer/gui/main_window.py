from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QProcess, QSettings, Qt, QUrl
from PySide6.QtGui import QAction, QBrush, QColor, QDesktopServices, QDragEnterEvent, QDropEvent, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QSplitter, QStatusBar, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QToolBar, QVBoxLayout, QWidget,
)

from histoanalyzer import __version__
from histoanalyzer.job import JobConfig, SUPPORTED_IMAGE_SUFFIXES, discover_images
from histoanalyzer.resources import bundled_classifier_paths
from histoanalyzer.worker import DONE_PREFIX, PROGRESS_PREFIX, RESULT_PREFIX

IMAGE_FILTER = (
    "Supported images (*.tif *.tiff *.svs *.ndpi *.mrxs *.scn *.vms *.vmu *.bif *.png *.jpg *.jpeg *.bmp);;"
    "TIFF images (*.tif *.tiff);;Whole-slide images (*.svs *.ndpi *.mrxs *.scn *.bif);;All files (*)"
)


def resource_path(relative: str) -> Path:
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidates = [base / relative, base / "histoanalyzer" / relative]
    else:
        package_root = Path(__file__).resolve().parents[1]
        repository_root = Path(__file__).resolve().parents[3]
        candidates = [package_root / relative, repository_root / relative]
    return next((path for path in candidates if path.exists()), candidates[0])


class ImagePreview(QScrollArea):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.title = title
        self.path: Optional[Path] = None
        self.label = QLabel(f"{title}\n\nNo preview available")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(480, 360)
        self.label.setStyleSheet("QLabel { background: #171b20; color: #aeb7c2; border: 1px solid #303842; }")
        self.setWidget(self.label)
        self.setWidgetResizable(True)

    def set_image(self, path: Path) -> None:
        self.path = path if path.exists() else None
        if not self.path:
            self.label.setText(f"{self.title}\n\nPreview not generated for this mode")
            self.label.setPixmap(QPixmap())
            return
        pixmap = QPixmap(str(self.path))
        if pixmap.isNull():
            self.label.setText(f"Could not load:\n{self.path}")
            return
        viewport_size = self.viewport().size()
        target = viewport_size.expandedTo(self.label.minimumSize())
        self.label.setPixmap(pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.label.setText("")
        self.label.setToolTip(str(self.path))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.path and self.path.exists():
            self.set_image(self.path)


class PathPicker(QWidget):
    def __init__(self, dialog: str = "file", filter_text: str = "All files (*)", parent=None) -> None:
        super().__init__(parent)
        self.dialog = dialog
        self.filter_text = filter_text
        self.edit = QLineEdit()
        self.button = QPushButton("Browse…")
        self.button.clicked.connect(self.browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:  # noqa: N802
        self.edit.setText(value)

    def browse(self) -> None:
        start = self.text() or str(Path.home())
        if self.dialog == "directory":
            value = QFileDialog.getExistingDirectory(self, "Select folder", start)
        elif self.dialog == "save":
            value, _ = QFileDialog.getSaveFileName(self, "Select output file", start, self.filter_text)
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Select file", start, self.filter_text)
        if value:
            self.setText(value)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"HistoAnalyzer {__version__}")
        self.resize(1540, 940)
        self.setAcceptDrops(True)
        self.settings = QSettings("HistoAnalyzer", "HistoAnalyzer")
        self.process: Optional[QProcess] = None
        self.job_file: Optional[Path] = None
        self.current_output: Optional[Path] = None
        self._stdout_buffer = ""
        self._build_ui()
        self._restore_settings()
        self._apply_style()

    def _build_ui(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        add_action = QAction("Add images", self)
        add_action.triggered.connect(self.add_images)
        add_folder_action = QAction("Add folder", self)
        add_folder_action.triggered.connect(self.add_folder)
        open_action = QAction("Open results", self)
        open_action.triggered.connect(self.open_results)
        toolbar.addAction(add_action)
        toolbar.addAction(add_folder_action)
        toolbar.addSeparator()
        toolbar.addAction(open_action)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_inputs_panel())
        splitter.addWidget(self._build_settings_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setSizes([390, 440, 710])
        splitter.setStretchFactor(2, 1)
        self.setCentralWidget(splitter)

        status = QStatusBar()
        self.setStatusBar(status)
        self.status_label = QLabel("Ready")
        status.addWidget(self.status_label, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(260)
        status.addPermanentWidget(self.progress)

    def _build_inputs_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("Input queue")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        mode_box = QGroupBox("Workflow")
        mode_layout = QFormLayout(mode_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Baseline + nuclei preview", "baseline")
        self.mode_combo.addItem("Predict Tumor / Stroma / Other", "predict")
        self.mode_combo.addItem("Train compartment model", "train")
        self.mode_combo.addItem("Train model, then predict", "train_predict")
        self.mode_combo.currentIndexChanged.connect(self._update_mode_ui)
        mode_layout.addRow("Mode", self.mode_combo)
        layout.addWidget(mode_box)

        self.image_table = QTableWidget(0, 3)
        self.image_table.setHorizontalHeaderLabels(["Image", "Annotation", "Status"])
        self.image_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.image_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.image_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.image_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.image_table.setAlternatingRowColors(True)
        layout.addWidget(self.image_table, 1)

        buttons = QHBoxLayout()
        add = QPushButton("Add images")
        add.clicked.connect(self.add_images)
        folder = QPushButton("Add folder")
        folder.clicked.connect(self.add_folder)
        remove = QPushButton("Remove")
        remove.clicked.connect(self.remove_selected)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear_images)
        buttons.addWidget(add)
        buttons.addWidget(folder)
        buttons.addWidget(remove)
        buttons.addWidget(clear)
        layout.addLayout(buttons)

        output_box = QGroupBox("Output")
        output_layout = QFormLayout(output_box)
        self.output_root = PathPicker("directory")
        output_layout.addRow("Results folder", self.output_root)
        layout.addWidget(output_box)

        self.run_button = QPushButton("Run analysis")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.run_analysis)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_button, 2)
        run_row.addWidget(self.cancel_button, 1)
        layout.addLayout(run_row)
        return panel

    def _build_settings_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("Analysis configuration")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        classifiers = QGroupBox("QuPath classifiers")
        form = QFormLayout(classifiers)
        self.tissue_picker = PathPicker("file", "JSON classifier (*.json)")
        self.anthra_picker = PathPicker("file", "JSON classifier (*.json)")
        self.dab_picker = PathPicker("file", "JSON classifier (*.json)")
        form.addRow("Tissue", self.tissue_picker)
        form.addRow("Anthracosis", self.anthra_picker)
        form.addRow("DAB threshold", self.dab_picker)
        defaults_row = QHBoxLayout()
        restore_defaults = QPushButton("Use bundled defaults")
        restore_defaults.setToolTip("Restore the three classifiers distributed with HistoAnalyzer")
        restore_defaults.clicked.connect(self._set_bundled_classifier_defaults)
        defaults_note = QLabel("Included with HistoAnalyzer")
        defaults_note.setObjectName("helpText")
        defaults_row.addWidget(restore_defaults)
        defaults_row.addWidget(defaults_note, 1)
        form.addRow(defaults_row)
        layout.addWidget(classifiers)

        compartments = QGroupBox("Compartment model")
        self.compartment_box = compartments
        form = QFormLayout(compartments)
        self.annotation_folder = PathPicker("directory")
        self.annotation_folder.edit.textChanged.connect(self._refresh_annotation_cells)
        self.compartment_model = PathPicker("file", "Random Forest model (*.joblib)")
        self.model_output = PathPicker("save", "Random Forest model (*.joblib)")
        form.addRow("Annotation folder", self.annotation_folder)
        form.addRow("Existing model", self.compartment_model)
        form.addRow("Model output", self.model_output)
        note = QLabel("Training pairs each image with <image>_compartments.geojson containing Tumor, Stroma and Other polygons.")
        note.setWordWrap(True)
        note.setObjectName("helpText")
        form.addRow(note)
        layout.addWidget(compartments)

        nuclei = QGroupBox("Nuclei segmentation")
        form = QFormLayout(nuclei)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["instanseg", "watershed"])
        self.instanseg_model = QComboBox()
        self.instanseg_model.setEditable(True)
        self.instanseg_model.addItems(["brightfield_nuclei", "fluorescence_nuclei_and_cells"])
        self.instanseg_input = QComboBox()
        self.instanseg_input.addItems(["rgb", "hematoxylin"])
        self.device_combo = QComboBox()
        self.device_combo.setEditable(True)
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.0, 20.0)
        self.pixel_size.setDecimals(4)
        self.pixel_size.setSpecialValueText("Use metadata")
        self.pixel_fallback = QDoubleSpinBox()
        self.pixel_fallback.setRange(0.01, 20.0)
        self.pixel_fallback.setDecimals(4)
        self.pixel_fallback.setValue(0.5)
        self.instanseg_tile = QSpinBox()
        self.instanseg_tile.setRange(128, 4096)
        self.instanseg_tile.setSingleStep(128)
        self.instanseg_tile.setValue(512)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 32)
        self.batch_size.setValue(1)
        self.min_area = QSpinBox()
        self.min_area.setRange(1, 10000)
        self.min_area.setValue(1)
        self.fallback_checkbox = QCheckBox("Fall back to watershed if InstanSeg fails")
        self.fallback_checkbox.setChecked(True)
        form.addRow("Backend", self.backend_combo)
        form.addRow("InstanSeg model", self.instanseg_model)
        form.addRow("Model input", self.instanseg_input)
        form.addRow("Device", self.device_combo)
        form.addRow("Pixel size override (µm/px)", self.pixel_size)
        form.addRow("Fallback resolution (µm/px)", self.pixel_fallback)
        form.addRow("InstanSeg tile", self.instanseg_tile)
        form.addRow("Batch size", self.batch_size)
        form.addRow("Minimum nucleus area (px)", self.min_area)
        form.addRow(self.fallback_checkbox)
        layout.addWidget(nuclei)

        nucleus_classes = QGroupBox("Nucleus classification and tissue graph")
        form = QFormLayout(nucleus_classes)
        self.enable_nucleus_classes = QCheckBox("Classify all CleanTissue nuclei")
        self.enable_nucleus_classes.setChecked(True)
        self.nucleus_classifier_model = PathPicker("file", "Nucleus classifier (*.joblib)")
        self.nucleus_classifier_model.setToolTip(
            "Optional trained predict_proba model. Leave empty to use the built-in morphology classifier."
        )
        self.nucleus_class_tile = QSpinBox()
        self.nucleus_class_tile.setRange(256, 4096)
        self.nucleus_class_tile.setSingleStep(256)
        self.nucleus_class_tile.setValue(1024)
        self.nucleus_class_halo = QSpinBox()
        self.nucleus_class_halo.setRange(0, 512)
        self.nucleus_class_halo.setValue(64)
        self.nucleus_graph_k = QSpinBox()
        self.nucleus_graph_k.setRange(1, 30)
        self.nucleus_graph_k.setValue(6)
        self.nucleus_graph_radius = QDoubleSpinBox()
        self.nucleus_graph_radius.setRange(5.0, 200.0)
        self.nucleus_graph_radius.setValue(25.0)
        self.nucleus_graph_radius.setSuffix(" µm")
        self.nucleus_region_size = QDoubleSpinBox()
        self.nucleus_region_size.setRange(30.0, 1000.0)
        self.nucleus_region_size.setValue(120.0)
        self.nucleus_region_size.setSuffix(" µm")
        note = QLabel(
            "Outputs include class probabilities, entropy/margin uncertainty, a color per class, "
            "a k-nearest-neighbour graph, and Tumour/Stroma/Immune/Vascular-rich regions. "
            "Built-in probabilities are morphology compatibility scores and require validation."
        )
        note.setWordWrap(True)
        note.setObjectName("helpText")
        form.addRow(self.enable_nucleus_classes)
        form.addRow("Optional trained model", self.nucleus_classifier_model)
        form.addRow("Classification tile", self.nucleus_class_tile)
        form.addRow("Tile halo", self.nucleus_class_halo)
        form.addRow("Graph neighbours (k)", self.nucleus_graph_k)
        form.addRow("Graph radius", self.nucleus_graph_radius)
        form.addRow("Tissue-region window", self.nucleus_region_size)
        form.addRow(note)
        layout.addWidget(nucleus_classes)

        basic = QGroupBox("Processing")
        form = QFormLayout(basic)
        self.ink_dilation = QSpinBox()
        self.ink_dilation.setRange(0, 100)
        self.ink_dilation.setValue(5)
        self.analysis_tile = QSpinBox()
        self.analysis_tile.setRange(0, 8192)
        self.analysis_tile.setSpecialValueText("Classifier default")
        self.preview_side = QSpinBox()
        self.preview_side.setRange(800, 8000)
        self.preview_side.setValue(2500)
        self.nuclei_preview_side = QSpinBox()
        self.nuclei_preview_side.setRange(512, 8192)
        self.nuclei_preview_side.setValue(2048)
        self.save_masks = QCheckBox("Save intermediate mask TIFFs")
        self.keep_work = QCheckBox("Keep temporary work masks")
        form.addRow("Anthracosis dilation (px)", self.ink_dilation)
        form.addRow("Analysis tile size", self.analysis_tile)
        form.addRow("Whole-image preview size", self.preview_side)
        form.addRow("Nuclei preview size", self.nuclei_preview_side)
        form.addRow(self.save_masks)
        form.addRow(self.keep_work)
        layout.addWidget(basic)

        advanced = QGroupBox("Tumor / Stroma regional classifier")
        form = QFormLayout(advanced)
        self.region_size = QDoubleSpinBox()
        self.region_size.setRange(20.0, 1000.0)
        self.region_size.setValue(160.0)
        self.region_size.setSuffix(" µm")
        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setRange(0.0, 1.0)
        self.min_confidence.setSingleStep(0.01)
        self.min_confidence.setValue(0.52)
        self.rf_trees = QSpinBox()
        self.rf_trees.setRange(50, 5000)
        self.rf_trees.setValue(500)
        self.min_regions = QSpinBox()
        self.min_regions.setRange(1, 10000)
        self.min_regions.setValue(20)
        form.addRow("Regional window", self.region_size)
        form.addRow("Minimum confidence", self.min_confidence)
        form.addRow("Random Forest trees", self.rf_trees)
        form.addRow("Minimum regions/class", self.min_regions)
        layout.addWidget(advanced)

        layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    def _build_results_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("Results and quality control")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        self.result_tabs = QTabWidget()
        self.stage_preview = ImagePreview("Pipeline stages")
        self.nuclei_preview = ImagePreview("Nuclei segmentation")
        self.nucleus_class_preview = ImagePreview("Nucleus classes and uncertainty")
        self.tissue_region_preview = ImagePreview("Graph-derived tissue regions")
        self.compartment_preview = ImagePreview("Tumor / Stroma / Other")
        self.dab_preview = ImagePreview("Compartment-specific DAB")
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.NoWrap)
        self.result_tabs.addTab(self.stage_preview, "Stages")
        self.result_tabs.addTab(self.nuclei_preview, "Nuclei")
        self.result_tabs.addTab(self.nucleus_class_preview, "Nucleus classes")
        self.result_tabs.addTab(self.tissue_region_preview, "Tissue regions")
        self.result_tabs.addTab(self.compartment_preview, "Compartments")
        self.result_tabs.addTab(self.dab_preview, "DAB")
        self.result_tabs.addTab(self.log_view, "Log")
        layout.addWidget(self.result_tabs, 1)
        open_row = QHBoxLayout()
        self.open_folder_button = QPushButton("Open current result folder")
        self.open_folder_button.clicked.connect(self.open_results)
        self.open_folder_button.setEnabled(False)
        self.copy_log_button = QPushButton("Copy log")
        self.copy_log_button.clicked.connect(lambda: QApplication.clipboard().setText(self.log_view.toPlainText()))
        open_row.addWidget(self.open_folder_button)
        open_row.addStretch()
        open_row.addWidget(self.copy_log_button)
        layout.addLayout(open_row)
        return panel

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f6f8; color: #202830; font-size: 12px; }
            QLabel#panelTitle { font-size: 18px; font-weight: 650; padding: 5px 0 9px 0; }
            QLabel#helpText { color: #697784; font-size: 11px; }
            QGroupBox { font-weight: 600; border: 1px solid #ccd3da; border-radius: 7px; margin-top: 12px; padding-top: 10px; background: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget { background: #ffffff; border: 1px solid #c7d0d9; border-radius: 4px; padding: 4px; }
            QPushButton { background: #e8edf2; border: 1px solid #c3ccd5; border-radius: 5px; padding: 7px 10px; }
            QPushButton:hover { background: #dce5ec; }
            QPushButton#primaryButton { background: #1769aa; color: white; border: none; font-weight: 650; padding: 10px; }
            QPushButton#primaryButton:hover { background: #0f5b98; }
            QPushButton:disabled { color: #8b949e; background: #e8eaed; }
            QTabWidget::pane { border: 1px solid #c7d0d9; background: #ffffff; }
            QTabBar::tab { padding: 8px 13px; background: #e9edf1; }
            QTabBar::tab:selected { background: #ffffff; font-weight: 600; }
            QProgressBar { border: 1px solid #bfc8d1; border-radius: 5px; text-align: center; background: #ffffff; }
            QProgressBar::chunk { background: #2b80c5; border-radius: 4px; }
        """)

    def _set_bundled_classifier_defaults(self) -> None:
        bundled = bundled_classifier_paths()
        self.tissue_picker.setText(str(bundled.tissue))
        self.anthra_picker.setText(str(bundled.anthra))
        self.dab_picker.setText(str(bundled.dab))

    def _restore_settings(self) -> None:
        bundled = bundled_classifier_paths()
        defaults = {
            "tissue": str(bundled.tissue), "anthra": str(bundled.anthra),
            "dab": str(bundled.dab), "output": "",
            "annotations": "", "model": "", "model_output": "",
        }
        self.tissue_picker.setText(self.settings.value("paths/tissue", defaults["tissue"]) or defaults["tissue"])
        self.anthra_picker.setText(self.settings.value("paths/anthra", defaults["anthra"]) or defaults["anthra"])
        self.dab_picker.setText(self.settings.value("paths/dab", defaults["dab"]) or defaults["dab"])
        self.output_root.setText(self.settings.value("paths/output", defaults["output"]))
        self.annotation_folder.setText(self.settings.value("paths/annotations", defaults["annotations"]))
        self.compartment_model.setText(self.settings.value("paths/model", defaults["model"]))
        self.model_output.setText(self.settings.value("paths/model_output", defaults["model_output"]))
        self.nucleus_classifier_model.setText(self.settings.value("paths/nucleus_model", "") or "")
        self._update_mode_ui()

    def _save_settings(self) -> None:
        self.settings.setValue("paths/tissue", self.tissue_picker.text())
        self.settings.setValue("paths/anthra", self.anthra_picker.text())
        self.settings.setValue("paths/dab", self.dab_picker.text())
        self.settings.setValue("paths/output", self.output_root.text())
        self.settings.setValue("paths/annotations", self.annotation_folder.text())
        self.settings.setValue("paths/model", self.compartment_model.text())
        self.settings.setValue("paths/model_output", self.model_output.text())
        self.settings.setValue("paths/nucleus_model", self.nucleus_classifier_model.text())

    def _refresh_annotation_cells(self) -> None:
        for row in range(self.image_table.rowCount()):
            self._update_annotation_cell(row)

    def _update_mode_ui(self) -> None:
        mode = self.mode_combo.currentData()
        training = mode in {"train", "train_predict"}
        prediction = mode in {"predict", "train_predict"}
        self.annotation_folder.setEnabled(training)
        self.model_output.setEnabled(training)
        self.compartment_model.setEnabled(mode == "predict")
        for row in range(self.image_table.rowCount()):
            self._update_annotation_cell(row)

    def _update_annotation_cell(self, row: int) -> None:
        image_item = self.image_table.item(row, 0)
        if not image_item:
            return
        image = Path(image_item.data(Qt.UserRole))
        mode = self.mode_combo.currentData()
        if mode not in {"train", "train_predict"}:
            value = "—"
        else:
            folder = Path(self.annotation_folder.text()) if self.annotation_folder.text() else image.parent
            candidates = [folder / f"{self._safe_stem(image)}_compartments.geojson", folder / f"{self._safe_stem(image)}.geojson"]
            match = next((p for p in candidates if p.exists()), None)
            value = match.name if match else "Missing"
        item = self.image_table.item(row, 1) or QTableWidgetItem()
        item.setText(value)
        item.setForeground(QBrush(QColor("#197447") if value not in {"Missing", "—"} else QColor("#b42318") if value == "Missing" else QColor("#7a8692")))
        self.image_table.setItem(row, 1, item)

    @staticmethod
    def _safe_stem(path: Path) -> str:
        lower = path.name.lower()
        for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
            if lower.endswith(suffix):
                return path.name[: -len(suffix)]
        return path.stem

    def add_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select images", str(Path.home()), IMAGE_FILTER)
        self._append_images(paths)

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select image folder", str(Path.home()))
        if not folder:
            return
        recursive = QMessageBox.question(
            self, "Search subfolders?", "Include images in subfolders?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        ) == QMessageBox.Yes
        self._append_images(discover_images(folder, recursive=recursive))

    def _append_images(self, paths: List[str]) -> None:
        existing = {
            str(Path(self.image_table.item(row, 0).data(Qt.UserRole)).resolve())
            for row in range(self.image_table.rowCount())
        }
        for raw in paths:
            path = Path(raw)
            if not path.exists() or not any(path.name.lower().endswith(s) for s in SUPPORTED_IMAGE_SUFFIXES):
                continue
            resolved = str(path.resolve())
            if resolved in existing:
                continue
            row = self.image_table.rowCount()
            self.image_table.insertRow(row)
            item = QTableWidgetItem(path.name)
            item.setData(Qt.UserRole, resolved)
            item.setToolTip(resolved)
            self.image_table.setItem(row, 0, item)
            self.image_table.setItem(row, 1, QTableWidgetItem("—"))
            self.image_table.setItem(row, 2, QTableWidgetItem("Queued"))
            self._update_annotation_cell(row)
            existing.add(resolved)
        if paths and not self.output_root.text():
            first = Path(paths[0])
            self.output_root.setText(str(first.parent / "HistoAnalyzer_results"))

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.image_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.image_table.removeRow(row)

    def clear_images(self) -> None:
        self.image_table.setRowCount(0)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        files: List[str] = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                files.extend(discover_images(path, recursive=True))
            else:
                files.append(str(path))
        self._append_images(files)

    def _images(self) -> List[str]:
        return [str(self.image_table.item(row, 0).data(Qt.UserRole)) for row in range(self.image_table.rowCount())]

    def build_job(self) -> JobConfig:
        override = self.pixel_size.value() if self.pixel_size.value() > 0 else None
        model_output = self.model_output.text()
        if self.mode_combo.currentData() in {"train", "train_predict"} and not model_output:
            base = Path(self.output_root.text() or Path(self._images()[0]).parent)
            model_output = str(base / "TumorStromaRF_InstanSeg.joblib")
        return JobConfig(
            mode=str(self.mode_combo.currentData()), images=self._images(),
            output_root=self.output_root.text(), tissue_classifier=self.tissue_picker.text(),
            anthra_classifier=self.anthra_picker.text(), dab_classifier=self.dab_picker.text(),
            compartment_model=self.compartment_model.text(), model_output=model_output,
            annotation_folder=self.annotation_folder.text(), ink_dilation=self.ink_dilation.value(),
            tile_size=self.analysis_tile.value() or None, preview_max_side=self.preview_side.value(),
            nuclei_preview_size=self.nuclei_preview_side.value(), save_mask_tiffs=self.save_masks.isChecked(),
            keep_work_masks=self.keep_work.isChecked(), nuclei_backend=self.backend_combo.currentText(),
            instanseg_model=self.instanseg_model.currentText().strip(), instanseg_input=self.instanseg_input.currentText(),
            instanseg_device=self.device_combo.currentText().strip(), instanseg_tile_size=self.instanseg_tile.value(),
            instanseg_batch_size=self.batch_size.value(), pixel_size_um=override,
            pixel_size_fallback_um=self.pixel_fallback.value(), instanseg_min_area_px=self.min_area.value(),
            instanseg_fallback_watershed=self.fallback_checkbox.isChecked(),
            enable_nucleus_classification=self.enable_nucleus_classes.isChecked(),
            nucleus_classifier_model=self.nucleus_classifier_model.text(),
            nucleus_classification_tile_size=self.nucleus_class_tile.value(),
            nucleus_classification_halo_px=self.nucleus_class_halo.value(),
            nucleus_graph_k=self.nucleus_graph_k.value(),
            nucleus_graph_radius_um=self.nucleus_graph_radius.value(),
            nucleus_tissue_region_size_um=self.nucleus_region_size.value(),
            region_size_um=self.region_size.value(),
            min_compartment_confidence=self.min_confidence.value(), rf_trees=self.rf_trees.value(),
            min_training_regions_per_class=self.min_regions.value(),
        )

    def run_analysis(self) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            return
        try:
            job = self.build_job()
            job.validate()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot start", str(exc))
            return
        self._save_settings()
        temp_dir = Path(tempfile.mkdtemp(prefix="histoanalyzer_gui_"))
        self.job_file = job.to_json(temp_dir / "job.json")
        self.log_view.clear()
        self.result_tabs.setCurrentWidget(self.log_view)
        self.progress.setValue(0)
        self.status_label.setText("Starting analysis…")
        for row in range(self.image_table.rowCount()):
            self.image_table.item(row, 2).setText("Queued")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        env = self.process.processEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)

        if getattr(sys, "frozen", False):
            program = sys.executable
            arguments = ["--worker", str(self.job_file)]
        else:
            program = sys.executable
            arguments = ["-m", "histoanalyzer", "--worker", str(self.job_file)]
        self._append_log(f"Launching: {program} {' '.join(arguments)}")
        self.process.start(program, arguments)

    def cancel_analysis(self) -> None:
        if not self.process or self.process.state() == QProcess.NotRunning:
            return
        if QMessageBox.question(self, "Cancel analysis", "Stop the current batch?", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self._append_log("Cancellation requested…")
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()
        self.status_label.setText("Cancelled")

    def _read_process_output(self) -> None:
        if not self.process:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        self._stdout_buffer += text
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            self._handle_output_line(line.rstrip())

    def _handle_output_line(self, line: str) -> None:
        if not line:
            return
        if line.startswith(PROGRESS_PREFIX):
            payload = json.loads(line[len(PROGRESS_PREFIX):])
            current, total = int(payload.get("current", 0)), max(1, int(payload.get("total", 1)))
            self.progress.setValue(round(current / total * 100))
            self.status_label.setText(str(payload.get("message", "Processing")))
            image = payload.get("image")
            if image:
                self._set_status_for_image(image, "Processing" if current < total else "Completed")
            return
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line[len(RESULT_PREFIX):])
            output = payload.get("output")
            if output:
                self.current_output = Path(output)
                self.open_folder_button.setEnabled(True)
                self._load_previews(self.current_output)
            image = payload.get("image")
            if image:
                self._set_status_for_image(image, "Completed")
            self._append_log(line)
            return
        if line.startswith(DONE_PREFIX):
            payload = json.loads(line[len(DONE_PREFIX):])
            if payload.get("success"):
                self.status_label.setText("Analysis complete")
                self.progress.setValue(100)
            elif payload.get("cancelled"):
                self.status_label.setText("Cancelled")
            else:
                self.status_label.setText("Analysis failed")
                self._append_log(f"ERROR: {payload.get('error', 'Unknown error')}")
            return
        self._append_log(line)

    def _append_log(self, line: str) -> None:
        self.log_view.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_status_for_image(self, image: str, status: str) -> None:
        resolved = str(Path(image).resolve())
        for row in range(self.image_table.rowCount()):
            if str(Path(self.image_table.item(row, 0).data(Qt.UserRole)).resolve()) == resolved:
                self.image_table.item(row, 2).setText(status)
                break

    def _load_previews(self, output: Path) -> None:
        stage_candidates = [output / "compartment_pipeline_stages_50pct.png", output / "pipeline_stages_50pct.png"]
        nuclei_candidates = [output / "nuclei_validation_montage.png", output / "nuclei_validation_raw_model_overlay.png", output / "nuclei_validation_overlay.png"]
        class_candidates = [output / "nuclei_class_overlay.png", output / "nuclei_class_uncertainty_overlay.png", output / "nuclei_class_legend.png"]
        region_candidates = [output / "tissue_region_overlay.png", output / "nuclei_graph_overlay.png"]
        comp_candidates = [output / "compartment_overlay_50pct.png"]
        dab_candidates = [output / "compartment_dab_overlay_50pct.png", output / "pipeline_overlay_50pct.png"]
        self.stage_preview.set_image(next((p for p in stage_candidates if p.exists()), stage_candidates[0]))
        self.nuclei_preview.set_image(next((p for p in nuclei_candidates if p.exists()), nuclei_candidates[0]))
        self.nucleus_class_preview.set_image(next((p for p in class_candidates if p.exists()), class_candidates[0]))
        self.tissue_region_preview.set_image(next((p for p in region_candidates if p.exists()), region_candidates[0]))
        self.compartment_preview.set_image(next((p for p in comp_candidates if p.exists()), comp_candidates[0]))
        self.dab_preview.set_image(next((p for p in dab_candidates if p.exists()), dab_candidates[0]))
        if any(p.exists() for p in class_candidates):
            self.result_tabs.setCurrentWidget(self.nucleus_class_preview)
        elif any(p.exists() for p in nuclei_candidates):
            self.result_tabs.setCurrentWidget(self.nuclei_preview)
        else:
            self.result_tabs.setCurrentWidget(self.stage_preview)

    def _process_finished(self, exit_code: int, _status) -> None:
        if self._stdout_buffer.strip():
            self._handle_output_line(self._stdout_buffer.strip())
            self._stdout_buffer = ""
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if exit_code != 0 and self.status_label.text() not in {"Cancelled", "Analysis failed"}:
            self.status_label.setText(f"Process exited with code {exit_code}")
            QMessageBox.warning(self, "Analysis did not finish", "Review the Log tab for details.")
        self.process = None

    def _process_error(self, error) -> None:
        self._append_log(f"Process error: {error}")
        self.status_label.setText("Process error")

    def open_results(self) -> None:
        target = self.current_output or (Path(self.output_root.text()) if self.output_root.text() else None)
        if target and target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        else:
            QMessageBox.information(self, "No results", "No result folder is available yet.")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.process and self.process.state() != QProcess.NotRunning:
            answer = QMessageBox.question(self, "Analysis running", "Cancel the analysis and close?", QMessageBox.Yes | QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.process.kill()
            self.process.waitForFinished(2000)
        self._save_settings()
        event.accept()


def launch() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("HistoAnalyzer")
    app.setOrganizationName("HistoAnalyzer")
    icon = resource_path("assets/icon.png")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    window = MainWindow()
    window.show()
    return app.exec()
