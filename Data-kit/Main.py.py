import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class DataPrepKit:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_format = self._determine_file_format()
        self.df = self.read_data()

    def _determine_file_format(self):
        _, file_extension = os.path.splitext(self.file_path)
        return file_extension[1:]  # remove the dot

    def read_data(self):
        """
        Read data from a file and return a Pandas DataFrame.
        """
        if self.file_format == 'csv':
            return pd.read_csv(self.file_path)
        elif self.file_format in ['xls', 'xlsx']:
            return pd.read_excel(self.file_path)
        elif self.file_format == 'json':
            return pd.read_json(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

    def summarize_data(self):
        """
        Print key statistical summaries of the data and visualize basic analysis.
        """
        print("Data Summary:")
        print(self.df.describe())
        self.visualize_data()

    def handle_missing_values(self, method='mean'):
        """
        Handle missing values in the DataFrame.
        """
        if method == 'mean':
            self.df = self.df.fillna(self.df.mean())
        elif method == 'median':
            self.df = self.df.fillna(self.df.median())
        elif method == 'mode':
            self.df = self.df.fillna(self.df.mode().iloc[0])
        elif method == 'ffill':
            self.df = self.df.fillna(method='ffill')
        elif method == 'bfill':
            self.df = self.df.fillna(method='bfill')
        else:
            raise ValueError(f"Unsupported method: {method}")

    def encode_categorical_data(self):
        """
        Encode categorical data in the DataFrame.
        """
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        self.df = pd.get_dummies(self.df, columns=categorical_cols)

    def export_data(self, export_format='csv'):
        """
        Export the DataFrame to a file.
        """
        if export_format == 'csv':
            self.df.to_csv('output.csv', index=False)
        elif export_format in ['xls', 'xlsx']:
            self.df.to_excel('output.xlsx', index=False)
        elif export_format == 'json':
            self.df.to_json('output.json', orient='records')
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

    def visualize_data(self):
        """
        Visualize basic analysis of the data using histograms for numerical features
        and bar charts for categorical features.
        """
        for column in self.df.columns:
            if self.df[column].dtype in ['int64', 'float64']:
                self.df[column].hist()
                plt.title(column)
                plt.show()
            elif self.df[column].dtype == 'object':
                self.df[column].value_counts().plot(kind='bar')
                plt.title(column)
                plt.show()

# Example usage
file_path = 'data.csv'
prep_kit = DataPrepKit(file_path)
prep_kit.summarize_data()
prep_kit.handle_missing_values(method='mean')
prep_kit.encode_categorical_data()
prep_kit.export_data()

