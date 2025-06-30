from transformers import AutoModelForCausalLM, AutoProcessor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QListWidget, QLabel, QTextEdit, QPushButton,
    QListWidgetItem, QDialog, QScrollArea, QFileDialog, QSplitter, QWidget,
    QVBoxLayout, QMessageBox
)
from PySide6.QtGui import QPixmap, QIcon, QAction, QFont, QGuiApplication
from PySide6.QtCore import Qt, QSize
from PIL import Image as PILImage
from PIL import ImageGrab, Image
import torch
import sys
import os
import subprocess
import platform

class ImagePromptManager(QMainWindow):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.data_dir = os.path.join(root_dir, "image_data")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.setWindowTitle("Image Prompt Manager")
        self.resize(1200, 800)

        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.zoom_factor = 1.0
        self.image_pixmap = None

        self.init_ui()
        self.init_menu()

    def init_menu(self):
        menubar = self.menuBar()
        tagger_menu = menubar.addMenu("Florence Tagger")
        run_action = QAction("Run Florence Tagger (missing tags only)", self)
        run_action.triggered.connect(self.run_florence_tagger)
        tagger_menu.addAction(run_action)

    def init_ui(self):
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: black;
                width: 1px;
                height: 1px;
            }
        """)
        self.setCentralWidget(self.splitter)

        label_font = QFont()
        label_font.setPointSize(12)

        # Column 1: Images
        col1_widget = QWidget()
        col1_layout = QVBoxLayout(col1_widget)
        col1_label = QLabel("Images")
        col1_label.setAlignment(Qt.AlignCenter)
        col1_label.setFont(label_font)
        col1_layout.addWidget(col1_label)

        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.load_image_data)
        col1_layout.addWidget(self.image_list)
        self.splitter.addWidget(col1_widget)

        # Column 2: Image Viewer
        col2_widget = QWidget()
        col2_layout = QVBoxLayout(col2_widget)
        col2_label = QLabel("Image Viewer")
        col2_label.setAlignment(Qt.AlignCenter)
        col2_label.setFont(label_font)
        col2_layout.addWidget(col2_label)

        self.image_scroll = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_scroll.setWidget(self.image_label)
        col2_layout.addWidget(self.image_scroll)

        self.copy_clipboard_button = QPushButton("Copy Reference Image to Clipboard")
        self.copy_clipboard_button.clicked.connect(self.copy_reference_to_clipboard)
        col2_layout.addWidget(self.copy_clipboard_button)

        self.splitter.addWidget(col2_widget)

        # Column 3: Tags
        col3_widget = QWidget()
        col3_layout = QVBoxLayout(col3_widget)
        col3_label = QLabel("Tags")
        col3_label.setAlignment(Qt.AlignCenter)
        col3_label.setFont(label_font)
        col3_layout.addWidget(col3_label)

        self.original_tag = QTextEdit()
        self.original_tag.setFont(label_font)
        self.original_tag.setPlaceholderText("Original tag")
        self.original_tag.textChanged.connect(self.save_original_tag)
        orig_label = QLabel("Original Tag")
        orig_label.setFont(label_font)
        col3_layout.addWidget(orig_label)
        col3_layout.addWidget(self.original_tag)

        self.edited_tag = QTextEdit()
        self.edited_tag.setFont(label_font)
        self.edited_tag.setPlaceholderText("Edited tag")
        self.edited_tag.textChanged.connect(self.save_edited_tag)
        edit_label = QLabel("Edited Tag")
        edit_label.setFont(label_font)
        col3_layout.addWidget(edit_label)
        col3_layout.addWidget(self.edited_tag)

        self.splitter.addWidget(col3_widget)

        # Column 4: Examples
        col4_widget = QWidget()
        col4_layout = QVBoxLayout(col4_widget)
        col4_label = QLabel("Examples")
        col4_label.setAlignment(Qt.AlignCenter)
        col4_label.setFont(label_font)
        col4_layout.addWidget(col4_label)

        self.example_list = QListWidget()
        self.example_list.setViewMode(QListWidget.IconMode)
        self.example_list.setIconSize(QSize(128, 128))
        self.example_list.itemClicked.connect(self.show_large_image)
        col4_layout.addWidget(self.example_list)

        button_row = QWidget()
        button_layout = QVBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.clicked.connect(self.open_data_folder)
        button_layout.addWidget(self.open_folder_button)
        
        self.refresh_examples_button = QPushButton("Refresh Examples")
        self.refresh_examples_button.clicked.connect(self.refresh_current_image_data)
        button_layout.addWidget(self.refresh_examples_button)

        self.paste_button = QPushButton("Paste image from clipboard")
        self.paste_button.clicked.connect(self.paste_clipboard_image)
        button_layout.addWidget(self.paste_button)


        col4_layout.addWidget(button_row)

        self.splitter.addWidget(col4_widget)

        self.splitter.setSizes([300, 300, 300, 300])
        self.load_image_list()

    def load_image_list(self):
        self.image_list.clear()
        for file in os.listdir(self.images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                base = os.path.splitext(file)[0]
                folder = os.path.join(self.data_dir, base)
                num_examples = len([
                    f for f in os.listdir(folder)
                    if f.lower().endswith(('.png', '.jpg'))
                ]) if os.path.exists(folder) else 0
                item = QListWidgetItem(f"{file} [{num_examples}]")
                item.setData(Qt.UserRole, file)
                self.image_list.addItem(item)
                
    def copy_reference_to_clipboard(self):
        item = self.image_list.currentItem()
        if not item:
            return

        filename = item.data(Qt.UserRole)
        image_path = os.path.join(self.images_dir, filename)

        if not os.path.exists(image_path):
            return

        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            QGuiApplication.clipboard().setPixmap(pixmap)

    def load_image_data(self, current, previous):
        if not current:
            return
        filename = current.data(Qt.UserRole)
        base = os.path.splitext(filename)[0]
        folder = os.path.join(self.data_dir, base)
        image_path = os.path.join(self.images_dir, filename)

        pixmap = QPixmap(image_path)
        self.image_pixmap = pixmap
        self.zoom_factor = 1.0
        self.update_image_display()

        orig_path = os.path.join(folder, "tags_original.txt")
        edit_path = os.path.join(folder, "tags_edited.txt")
        if os.path.exists(orig_path):
            with open(orig_path, "r", encoding="utf-8") as f:
                self.original_tag.blockSignals(True)
                self.original_tag.setPlainText(f.read())
                self.original_tag.blockSignals(False)
        else:
            self.original_tag.blockSignals(True)
            self.original_tag.setPlainText("")
            self.original_tag.blockSignals(False)
        if os.path.exists(edit_path):
            with open(edit_path, "r", encoding="utf-8") as f:
                self.edited_tag.blockSignals(True)
                self.edited_tag.setPlainText(f.read())
                self.edited_tag.blockSignals(False)
        else:
            self.edited_tag.blockSignals(True)
            self.edited_tag.setPlainText("")
            self.edited_tag.blockSignals(False)

        self.example_list.clear()
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg')):
                    icon = QIcon(os.path.join(folder, f))
                    item = QListWidgetItem(icon, f)
                    item.setData(Qt.UserRole, os.path.join(folder, f))
                    self.example_list.addItem(item)
                    
                    
    def refresh_current_image_data(self):
        current = self.image_list.currentItem()
        if current:
            self.load_image_data(current, None)

    def update_image_display(self):
        if self.image_pixmap is None:
            return
        scaled_pixmap = self.image_pixmap.scaled(
            self.image_pixmap.size() * self.zoom_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

    def wheelEvent(self, event):
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            angle = event.angleDelta().y()
            if angle > 0:
                self.zoom_factor *= 1.1
            else:
                self.zoom_factor /= 1.1
            self.update_image_display()
        else:
            super().wheelEvent(event)

    def save_original_tag(self):
        item = self.image_list.currentItem()
        if not item:
            return
        filename = item.data(Qt.UserRole)
        base = os.path.splitext(filename)[0]
        folder = os.path.join(self.data_dir, base)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "tags_original.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.original_tag.toPlainText())

    def save_edited_tag(self):
        item = self.image_list.currentItem()
        if not item:
            return
        filename = item.data(Qt.UserRole)
        base = os.path.splitext(filename)[0]
        folder = os.path.join(self.data_dir, base)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "tags_edited.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.edited_tag.toPlainText())

    # def paste_clipboard_image(self):
    #     clipboard = QApplication.clipboard()
    #     image = clipboard.image()
    #     if image.isNull():
    #         return
    #     item = self.image_list.currentItem()
    #     if not item:
    #         return
    #     filename = item.data(Qt.UserRole)
    #     base = os.path.splitext(filename)[0]
    #     folder = os.path.join(self.data_dir, base)
    #     os.makedirs(folder, exist_ok=True)
    #     existing = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg'))]
    #     example_name = f"example_{len(existing) + 1}.png"
    #     path = os.path.join(folder, example_name)
    #     image.save(path)
    #     self.load_image_data(item, None)
        

    def paste_clipboard_image(self):
        item = self.image_list.currentItem()
        if not item:
            return

        image = ImageGrab.grabclipboard()
        if image is None:
            QMessageBox.warning(self, "Paste Failed", "No image found in clipboard.")
            return

        filename = item.data(Qt.UserRole)
        base = os.path.splitext(filename)[0]
        folder = os.path.join(self.data_dir, base)
        os.makedirs(folder, exist_ok=True)

        existing = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg'))]
        example_name = f"example_{len(existing) + 1}.png"
        path = os.path.join(folder, example_name)

        image.save(path)
        self.load_image_data(item, None)

    def open_data_folder(self):
        item = self.image_list.currentItem()
        if not item:
            return
        filename = item.data(Qt.UserRole)
        base = os.path.splitext(filename)[0]
        folder = os.path.join(self.data_dir, base)
        os.makedirs(folder, exist_ok=True)

        if platform.system() == "Windows":
            os.startfile(folder)
        elif platform.system() == "Darwin":
            subprocess.run(["open", folder])
        else:
            subprocess.run(["xdg-open", folder])

    def show_large_image(self, item):
        path = item.data(Qt.UserRole)
        dlg = QDialog(self)
        dlg.setWindowTitle(item.text())
        vbox = QVBoxLayout(dlg)
        label = QLabel()
        pixmap = QPixmap(path)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(label)
        vbox.addWidget(scroll)
        dlg.resize(600, 600)
        dlg.exec()

    def run_florence_tagger(self):
        if self.model is None or self.processor is None:
            print("Loading Florence2 model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True
            ).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(
                "MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True
            )
            print("Model loaded.")

        prompt = "<MIXED_CAPTION>"

        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            filename = item.data(Qt.UserRole)
            base = os.path.splitext(filename)[0]
            folder = os.path.join(self.data_dir, base)
            orig_path = os.path.join(folder, "tags_original.txt")
            if os.path.exists(orig_path):
                continue

            image_path = os.path.join(self.images_dir, filename)
            pil_image = PILImage.open(image_path).convert("RGB")

            inputs = self.processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=3
                )

            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(pil_image.width, pil_image.height)
            )

            os.makedirs(folder, exist_ok=True)
            with open(orig_path, "w", encoding="utf-8") as f:
                f.write(parsed_answer[prompt].strip())
            print(f"Tag generated for {filename}")

        self.load_image_data(self.image_list.currentItem(), None)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    root = QFileDialog.getExistingDirectory(None, "Select Root Directory")
    if root:
        w = ImagePromptManager(root)
        w.show()
        sys.exit(app.exec())
