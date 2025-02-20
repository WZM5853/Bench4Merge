import csv
from os.path import exists

class CircularCSVLogger:
    def __init__(self, filename, max_rows=140):
        self.filename = filename
        self.max_rows = max_rows + 1  # Adjust for the fixed first row
        self.data = self._load_existing_data()


    # x,y,width,height,v_x,v_y,acc_x,acc_y,road_ID,heading,yaw_rate
    def _load_existing_data(self):
        """Load existing data from the CSV file if it exists, keeping the first row fixed."""
        if not exists(self.filename):
            return  [['time', 'vehicle-ID', 'x', 'y', 'width', 'height','v_x','v_y','acc_x','acc_y','road_ID','heading']]  # Initialize with a fixed first row if file does not exist
        
        with open(self.filename, 'r', newline='') as file:
            reader = csv.reader(file)
            data_list = list(reader)
            return data_list if data_list else [['time', 'vehicle-ID', 'x', 'y', 'width', 'height','v_x','v_y','acc_x','acc_y','road_ID','heading']]

    def add_row(self, row):
        """Add a new row to the data list, ensuring only the last 10 entries are dynamic."""
        if len(self.data) >= self.max_rows:  # Check if the total length reaches the max rows limit
            self.data.pop(1)  # Remove the second row (the oldest dynamic row)
        self.data.append(row)  # Append new row as the newest entry
        self._write_to_csv()

    def _write_to_csv(self):
        """Write the current list of data rows to the CSV file."""
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)

    def _write_to_csv(self):
        """Write the current list of data rows to the CSV file."""
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)

class CircularCSVLogger2:
    def __init__(self, filename, max_rows=14):
        self.filename = filename
        self.max_rows = max_rows + 1  # Adjust for the fixed first row
        self.data = self._load_existing_data()

    def _load_existing_data(self):
        """Load existing data from the CSV file if it exists, keeping the first row fixed."""
        if not exists(self.filename):
            return  [['time', 'vehicle-ID', 'x', 'y', 'width', 'height','v_x','v_y','acc_x','acc_y','road_ID','heading']]  # Initialize with a fixed first row if file does not exist
        
        with open(self.filename, 'r', newline='') as file:
            reader = csv.reader(file)
            data_list = list(reader)
            return data_list if data_list else [['time', 'vehicle-ID', 'x', 'y', 'width', 'height','v_x','v_y','acc_x','acc_y','road_ID','heading']]

    def add_row(self, row):
        """Add a new row to the data list, ensuring only the last 10 entries are dynamic."""
        if len(self.data) >= self.max_rows:  # Check if the total length reaches the max rows limit
            self.data.pop(1)  # Remove the second row (the oldest dynamic row)
        self.data.append(row)  # Append new row as the newest entry
        self._write_to_csv()

    def _write_to_csv(self):
        """Write the current list of data rows to the CSV file."""
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)

    def _write_to_csv(self):
        """Write the current list of data rows to the CSV file."""
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)

class CircularCSVLogger3:
    def __init__(self, filename, max_rows=14):
        self.filename = filename
        self.max_rows = max_rows + 1  # Adjust for the fixed first row
        self.data = self._load_existing_data()

    def _load_existing_data(self):
        """Load existing data from the CSV file if it exists, keeping the first row fixed."""
        if not exists(self.filename):
            return  [['time','ID','v_x','v_y','heading']]  # Initialize with a fixed first row if file does not exist
        
        with open(self.filename, 'r', newline='') as file:
            reader = csv.reader(file)
            data_list = list(reader)
            return data_list if data_list else [['time','ID','v_x','v_y','heading']]

    def add_row(self, row):
        """Add a new row to the data list, ensuring only the last 10 entries are dynamic."""
        if len(self.data) >= self.max_rows:  # Check if the total length reaches the max rows limit
            self.data.pop(1)  # Remove the second row (the oldest dynamic row)
        self.data.append(row)  # Append new row as the newest entry
        self._write_to_csv()

    def _write_to_csv(self):
        """Write the current list of data rows to the CSV file."""
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)

    def _write_to_csv(self):
        """Write the current list of data rows to the CSV file."""
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)
