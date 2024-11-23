import sys
import re
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLineEdit, QTableWidget,
    QTableWidgetItem, QPushButton, QHBoxLayout, QDialog, QWidget, QMenuBar, QStatusBar
)
from PyQt5.QtCore import Qt


class EpochAnnotChooser(QDialog):
    def __init__(self, annotations, times, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filterable List with Two Columns")
        self.resize(600, 500)

        if len(annotations) != len(times):
            raise ValueError("The annotations and times lists must have the same length.")

        # Main layout
        layout = QVBoxLayout(self)

        self.selected_annots = []

        # Search input
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("Search (supports regex)")
        self.search_input.textChanged.connect(self.filter_items)
        layout.addWidget(self.search_input)

        # Table widget
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Annotation", "Time (s)"])
        self.table_widget.setSelectionBehavior(self.table_widget.SelectRows)
        self.table_widget.setSelectionMode(self.table_widget.SingleSelection)
        layout.addWidget(self.table_widget)

        # Populate table
        self.annotations = annotations
        self.times = times
        self.populate_table()

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

    def populate_table(self):
        """Populate the table with annotations and times."""
        self.table_widget.setRowCount(len(self.annotations))
        for row, (annotation, time) in enumerate(zip(self.annotations, self.times)):
            self.table_widget.setItem(row, 0, QTableWidgetItem(annotation))
            # Convert time to float if it's not already, and format it
            try:
                time_value = float(time)  # Convert to float
                time_text = f"{time_value:.2f}"  # Format as a floating-point string
            except ValueError:
                time_text = str(time)  # Fallback: use the raw string if conversion fails
            self.table_widget.setItem(row, 1, QTableWidgetItem(time_text))

    def filter_items(self):
        """Filter rows based on the search input using regex."""
        search_text = self.search_input.text()
        try:
            pattern = re.compile(search_text, re.IGNORECASE)
        except re.error:
            # Invalid regex, show all rows
            pattern = None

        for row in range(self.table_widget.rowCount()):
            annotation_item = self.table_widget.item(row, 0)
            time_item = self.table_widget.item(row, 1)
            match = (
                pattern.search(annotation_item.text()) or
                pattern.search(time_item.text())
            ) if pattern else True
            self.table_widget.setRowHidden(row, not match)

    def apply_selection(self):
        """Return the selected rows."""
        selected_items = []
        selected_rows = set(index.row() for index in self.table_widget.selectedIndexes())

        for row in selected_rows:
            annotation = self.table_widget.item(row, 0).text()
            time = float(self.table_widget.item(row, 1).text())
            selected_items.append((annotation, time))

        #print("Selected items:", selected_items)
        self.selected_annots = selected_items
        self.accept()
