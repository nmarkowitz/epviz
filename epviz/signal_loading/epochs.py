import sys
import re
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QListWidget, QPushButton, QHBoxLayout, QDialog
)
from PyQt5.QtCore import Qt

class EpochAnnotChooser(QDialog):
    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.selected_annots = []
        self.setWindowTitle("Choose annotations to mark as epochs")
        self.resize(400, 500)

        # Main layout
        layout = QVBoxLayout(self)

        # Search input
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("Search (supports regex)")
        self.search_input.textChanged.connect(self.filter_items)
        layout.addWidget(self.search_input)

        # List widget
        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.list_widget)

        # Populate list
        self.items = items
        for item in items:
            self.list_widget.addItem(item)

        # Button layout
        button_layout = QHBoxLayout()

        # Apply button
        self.apply_button = QPushButton("Apply", self)
        self.apply_button.clicked.connect(self.apply_selection)
        button_layout.addWidget(self.apply_button)

        # Cancel button
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def filter_items(self):
        """Filter items based on the search input using regex."""
        search_text = self.search_input.text()
        try:
            pattern = re.compile(search_text, re.IGNORECASE)
        except re.error:
            # Invalid regex, show all items
            pattern = None

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            match = pattern.search(item.text()) if pattern else True
            item.setHidden(not match)

    def apply_selection(self):
        """Return the selected items."""
        selected_items = [
            item.text()
            for item in self.list_widget.selectedItems()
        ]
        self.selected_annots = selected_items
        print("Selected items:", selected_items)
        self.accept()
