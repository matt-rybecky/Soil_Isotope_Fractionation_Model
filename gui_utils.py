"""
GUI Utility Functions for Soil Evaporation Model
===============================================

Reusable GUI components and utility functions for the soil evaporation application.
Provides standardized widgets, dialog functions, and common GUI operations.

Author: Enhanced Python Implementation for Soil Evaporation Research
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple


class StandardStyles:
    """Standardized styling constants for consistent GUI appearance."""
    
    # Colors
    PRIMARY_COLOR = "#2E86AB"
    SUCCESS_COLOR = "#28A745"
    WARNING_COLOR = "#FFC107"
    ERROR_COLOR = "#DC3545"
    BACKGROUND_COLOR = "#F8F9FA"
    
    # Fonts
    HEADER_FONT = ("Arial", 12, "bold")
    BODY_FONT = ("Arial", 10)
    MONO_FONT = ("Courier", 9)
    
    # Padding
    STANDARD_PADDING = 5
    SECTION_PADDING = 10


class GUIUtils:
    """Collection of reusable GUI utility functions."""
    
    @staticmethod
    def create_labeled_entry(parent: tk.Widget, label_text: str, 
                           variable: tk.Variable, width: int = 15,
                           row: int = None, column: int = 0, 
                           sticky: str = "w") -> Tuple[tk.Label, tk.Entry]:
        """
        Create a labeled entry widget with consistent styling.
        
        Args:
            parent: Parent widget
            label_text: Text for the label
            variable: tkinter Variable to bind to entry
            width: Width of entry widget
            row: Grid row (if None, no grid placement)
            column: Grid column for label
            sticky: Grid sticky option
            
        Returns:
            Tuple of (label, entry) widgets
        """
        label = tk.Label(parent, text=label_text, font=StandardStyles.BODY_FONT)
        entry = tk.Entry(parent, textvariable=variable, width=width, 
                        font=StandardStyles.BODY_FONT)
        
        if row is not None:
            label.grid(row=row, column=column, sticky=sticky, 
                      padx=StandardStyles.STANDARD_PADDING,
                      pady=StandardStyles.STANDARD_PADDING)
            entry.grid(row=row, column=column+1, sticky=sticky,
                      padx=StandardStyles.STANDARD_PADDING,
                      pady=StandardStyles.STANDARD_PADDING)
        
        return label, entry
    
    @staticmethod
    def create_labeled_combobox(parent: tk.Widget, label_text: str,
                              variable: tk.Variable, values: List[str],
                              width: int = 15, row: int = None, 
                              column: int = 0, sticky: str = "w",
                              callback: Optional[Callable] = None) -> Tuple[tk.Label, ttk.Combobox]:
        """
        Create a labeled combobox with consistent styling.
        
        Args:
            parent: Parent widget
            label_text: Text for the label
            variable: tkinter Variable to bind to combobox
            values: List of combobox values
            width: Width of combobox widget
            row: Grid row (if None, no grid placement)
            column: Grid column for label
            sticky: Grid sticky option
            callback: Optional callback function for selection change
            
        Returns:
            Tuple of (label, combobox) widgets
        """
        label = tk.Label(parent, text=label_text, font=StandardStyles.BODY_FONT)
        combobox = ttk.Combobox(parent, textvariable=variable, values=values,
                               width=width, font=StandardStyles.BODY_FONT)
        
        if callback:
            combobox.bind('<<ComboboxSelected>>', callback)
        
        if row is not None:
            label.grid(row=row, column=column, sticky=sticky,
                      padx=StandardStyles.STANDARD_PADDING,
                      pady=StandardStyles.STANDARD_PADDING)
            combobox.grid(row=row, column=column+1, sticky=sticky,
                         padx=StandardStyles.STANDARD_PADDING,
                         pady=StandardStyles.STANDARD_PADDING)
        
        return label, combobox
    
    @staticmethod
    def create_status_label(parent: tk.Widget, initial_text: str = "Ready",
                          row: int = None, column: int = 0, 
                          columnspan: int = 1) -> tk.Label:
        """
        Create a status label with standardized appearance.
        
        Args:
            parent: Parent widget
            initial_text: Initial status text
            row: Grid row (if None, no grid placement)
            column: Grid column
            columnspan: Grid columnspan
            
        Returns:
            Status label widget
        """
        status_label = tk.Label(parent, text=initial_text, 
                               font=StandardStyles.BODY_FONT,
                               foreground=StandardStyles.PRIMARY_COLOR)
        
        if row is not None:
            status_label.grid(row=row, column=column, columnspan=columnspan,
                             padx=StandardStyles.STANDARD_PADDING,
                             pady=StandardStyles.STANDARD_PADDING)
        
        return status_label
    
    @staticmethod
    def create_section_frame(parent: tk.Widget, title: str, 
                           row: int = None, column: int = 0,
                           columnspan: int = 1, sticky: str = "ew") -> tk.LabelFrame:
        """
        Create a labeled frame for organizing GUI sections.
        
        Args:
            parent: Parent widget
            title: Frame title
            row: Grid row (if None, no grid placement)
            column: Grid column
            columnspan: Grid columnspan
            sticky: Grid sticky option
            
        Returns:
            LabelFrame widget
        """
        frame = tk.LabelFrame(parent, text=title, font=StandardStyles.HEADER_FONT,
                             padx=StandardStyles.SECTION_PADDING,
                             pady=StandardStyles.SECTION_PADDING)
        
        if row is not None:
            frame.grid(row=row, column=column, columnspan=columnspan,
                      sticky=sticky, padx=StandardStyles.STANDARD_PADDING,
                      pady=StandardStyles.STANDARD_PADDING)
        
        return frame
    
    @staticmethod
    def create_file_browser_frame(parent: tk.Widget, title: str, 
                                variable: tk.Variable, browse_callback: Callable,
                                file_types: List[Tuple[str, str]] = None,
                                row: int = None) -> tk.LabelFrame:
        """
        Create a file browser section with browse button and status.
        
        Args:
            parent: Parent widget
            title: Section title
            variable: Variable to store selected file path
            browse_callback: Function to call when browse button is clicked
            file_types: List of (description, pattern) tuples for file types
            row: Grid row (if None, no grid placement)
            
        Returns:
            LabelFrame containing the file browser controls
        """
        if file_types is None:
            file_types = [("All files", "*.*")]
        
        frame = GUIUtils.create_section_frame(parent, title, row=row)
        
        # File path display
        file_label = tk.Label(frame, text="Selected file:", font=StandardStyles.BODY_FONT)
        file_label.grid(row=0, column=0, sticky="w", pady=2)
        
        file_display = tk.Label(frame, textvariable=variable, font=StandardStyles.MONO_FONT,
                               foreground=StandardStyles.PRIMARY_COLOR, wraplength=400)
        file_display.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        
        # Browse button
        browse_btn = tk.Button(frame, text="Browse...", command=browse_callback,
                              font=StandardStyles.BODY_FONT)
        browse_btn.grid(row=0, column=1, sticky="e", padx=5)
        
        return frame
    
    @staticmethod
    def show_error_dialog(title: str, message: str, details: str = None):
        """
        Show standardized error dialog.
        
        Args:
            title: Dialog title
            message: Main error message
            details: Optional detailed error information
        """
        full_message = message
        if details:
            full_message += f"\n\nDetails:\n{details}"
        messagebox.showerror(title, full_message)
    
    @staticmethod
    def show_success_dialog(title: str, message: str):
        """
        Show standardized success dialog.
        
        Args:
            title: Dialog title
            message: Success message
        """
        messagebox.showinfo(title, message)
    
    @staticmethod
    def show_warning_dialog(title: str, message: str) -> bool:
        """
        Show standardized warning dialog with Yes/No options.
        
        Args:
            title: Dialog title
            message: Warning message
            
        Returns:
            True if user clicked Yes, False otherwise
        """
        return messagebox.askyesno(title, message)
    
    @staticmethod
    def update_status_safely(root: tk.Tk, status_label: tk.Label, 
                           text: str, color: str = None):
        """
        Thread-safe status label update.
        
        Args:
            root: Root tkinter window
            status_label: Status label to update
            text: New status text
            color: Optional text color
        """
        def update():
            status_label.config(text=text)
            if color:
                status_label.config(foreground=color)
        
        root.after(0, update)




class DataPathManager:
    """
    Centralized data path management system.
    
    Handles file paths for different data types (ERA5, lab data, field data)
    and provides file browser functionality.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize path manager.
        
        Args:
            base_dir: Base directory for the project. If None, uses current script directory.
        """
        if base_dir is None:
            base_dir = Path(__file__).parent
        
        self.base_dir = Path(base_dir)
        
        # Default data directories
        self.field_data_dir = self.base_dir / "field_data" 
        self.era5_data_dir = self.base_dir / "ERA5_DATA"
        self.output_dir = self.base_dir / "output_results"
        
        # Currently selected files
        self.selected_files: Dict[str, Optional[Path]] = {
            'era5_file': None,
            'field_data_file': None
        }
        
        # Ensure directories exist
        self._ensure_directories_exist()
    
    def _ensure_directories_exist(self):
        """Create data directories if they don't exist."""
        for directory in [self.field_data_dir, 
                         self.era5_data_dir, self.output_dir]:
            directory.mkdir(exist_ok=True)
    
    def browse_era5_data(self, parent=None) -> Optional[Path]:
        """
        Open file browser for ERA5 data files.
        
        Args:
            parent: Parent widget for the dialog
            
        Returns:
            Selected file path or None if cancelled
        """
        file_path = filedialog.askopenfilename(
            parent=parent,
            title="Select ERA5 Data File",
            initialdir=str(self.era5_data_dir),
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            selected_path = Path(file_path)
            self.selected_files['era5_file'] = selected_path
            return selected_path
        return None
    
    
    def browse_field_data(self, parent=None) -> Optional[Path]:
        """
        Open file browser for field data files.
        
        Args:
            parent: Parent widget for the dialog
            
        Returns:
            Selected file path or None if cancelled
        """
        file_path = filedialog.askopenfilename(
            parent=parent,
            title="Select Field Data File",
            initialdir=str(self.field_data_dir),
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            selected_path = Path(file_path)
            self.selected_files['field_data_file'] = selected_path
            return selected_path
        return None
    
    def get_available_files(self, data_type: str) -> List[Path]:
        """
        Get list of available data files by type.
        
        Args:
            data_type: Type of data ('era5', 'lab', 'field')
            
        Returns:
            List of available file paths
        """
        if data_type == 'era5':
            directory = self.era5_data_dir
            patterns = ['*.csv']
        elif data_type == 'field':
            directory = self.field_data_dir
            patterns = ['*.csv', '*.xlsx']
        else:
            return []
        
        files = []
        for pattern in patterns:
            files.extend(directory.glob(pattern))
        
        return sorted(files)
    
    def get_selected_file(self, file_type: str) -> Optional[Path]:
        """
        Get currently selected file for a given type.
        
        Args:
            file_type: Type of file ('era5_file', 'lab_temp_rh_file', etc.)
            
        Returns:
            Selected file path or None
        """
        return self.selected_files.get(file_type)
    
    def set_selected_file(self, file_type: str, file_path: Path):
        """
        Set selected file for a given type.
        
        Args:
            file_type: Type of file ('era5_file', 'lab_temp_rh_file', etc.)
            file_path: Path to the selected file
        """
        if file_type in self.selected_files:
            self.selected_files[file_type] = Path(file_path)
    
    def get_output_path(self, filename: str) -> Path:
        """
        Get path for output file.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Full path for the output file
        """
        return self.output_dir / filename
    
    def validate_file_exists(self, file_type: str) -> bool:
        """
        Check if selected file exists.
        
        Args:
            file_type: Type of file to check
            
        Returns:
            True if file exists, False otherwise
        """
        file_path = self.selected_files.get(file_type)
        return file_path is not None and file_path.exists()
    
    def get_status_summary(self) -> Dict[str, str]:
        """
        Get status summary of all selected files.
        
        Returns:
            Dictionary with file types and their status
        """
        status = {}
        for file_type, file_path in self.selected_files.items():
            if file_path is None:
                status[file_type] = "Not selected"
            elif file_path.exists():
                status[file_type] = f"✅ {file_path.name}"
            else:
                status[file_type] = f"❌ {file_path.name} (not found)"
        
        return status