# %% Import libraries
import os, sys
sys.path.insert(1, os.getcwd())

from src.msa.feature_extraction import features
from src.msa.visualization import plot
# from internal.meda import meda

import obspy
from obspy import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker 

# %% Execution parameters
show_plots = False
save_data = True


# %% Read files
path = "/media/gsus/76C8A92EC8A8EE15/data/mseed/"#C7.PLPI..*.D.*.*"

files = ['*.PLPI..*.D.2021.213', '*.PLPI..*.D.2021.214', '*.PLPI..*.D.2021.215', '*.PLPI..*.D.2021.216', '*.PLPI..*.D.2021.217', '*.PLPI..*.D.2021.218', '*.PLPI..*.D.2021.219', '*.PLPI..*.D.2021.220', '*.PLPI..*.D.2021.221', '*.PLPI..*.D.2021.222', '*.PLPI..*.D.2021.223', '*.PLPI..*.D.2021.224', '*.PLPI..*.D.2021.225', '*.PLPI..*.D.2021.226', '*.PLPI..*.D.2021.227', '*.PLPI..*.D.2021.228', '*.PLPI..*.D.2021.229', '*.PLPI..*.D.2021.230', '*.PLPI..*.D.2021.231', '*.PLPI..*.D.2021.232', '*.PLPI..*.D.2021.233', '*.PLPI..*.D.2021.234', '*.PLPI..*.D.2021.235', '*.PLPI..*.D.2021.236', '*.PLPI..*.D.2021.237', '*.PLPI..*.D.2021.238', '*.PLPI..*.D.2021.239', '*.PLPI..*.D.2021.240', '*.PLPI..*.D.2021.241', '*.PLPI..*.D.2021.242', '*.PLPI..*.D.2021.243', '*.PLPI..*.D.2021.244', '*.PLPI..*.D.2021.245', '*.PLPI..*.D.2021.246', '*.PLPI..*.D.2021.247', '*.PLPI..*.D.2021.248', '*.PLPI..*.D.2021.249', '*.PLPI..*.D.2021.250', '*.PLPI..*.D.2021.251', '*.PLPI..*.D.2021.252', '*.PLPI..*.D.2021.253', '*.PLPI..*.D.2021.254', '*.PLPI..*.D.2021.255', '*.PLPI..*.D.2021.256', '*.PLPI..*.D.2021.257', '*.PLPI..*.D.2021.258', '*.PLPI..*.D.2021.259', '*.PLPI..*.D.2021.260', '*.PLPI..*.D.2021.261', '*.PLPI..*.D.2021.262', '*.PLPI..*.D.2021.263', '*.PLPI..*.D.2021.264', '*.PLPI..*.D.2021.265', '*.PLPI..*.D.2021.266', '*.PLPI..*.D.2021.267', '*.PLPI..*.D.2021.268', '*.PLPI..*.D.2021.269', '*.PLPI..*.D.2021.270', '*.PLPI..*.D.2021.271', '*.PLPI..*.D.2021.272', '*.PLPI..*.D.2021.273', '*.PLPI..*.D.2021.274', '*.PLPI..*.D.2021.275', '*.PLPI..*.D.2021.276', '*.PLPI..*.D.2021.277', '*.PLPI..*.D.2021.278', '*.PLPI..*.D.2021.279', '*.PLPI..*.D.2021.280', '*.PLPI..*.D.2021.281', '*.PLPI..*.D.2021.282', '*.PLPI..*.D.2021.283', '*.PLPI..*.D.2021.284', '*.PLPI..*.D.2021.285', '*.PLPI..*.D.2021.286', '*.PLPI..*.D.2021.287', '*.PLPI..*.D.2021.288', '*.PLPI..*.D.2021.289', '*.PLPI..*.D.2021.290', '*.PLPI..*.D.2021.291', '*.PLPI..*.D.2021.292', '*.PLPI..*.D.2021.293', '*.PLPI..*.D.2021.294', '*.PLPI..*.D.2021.295', '*.PLPI..*.D.2021.296', '*.PLPI..*.D.2021.297', '*.PLPI..*.D.2021.298', '*.PLPI..*.D.2021.299', '*.PLPI..*.D.2021.300', '*.PLPI..*.D.2021.301', '*.PLPI..*.D.2021.302', '*.PLPI..*.D.2021.303', '*.PLPI..*.D.2021.304', '*.PLPI..*.D.2021.305', '*.PLPI..*.D.2021.306', '*.PLPI..*.D.2021.307', '*.PLPI..*.D.2021.308', '*.PLPI..*.D.2021.309', '*.PLPI..*.D.2021.310', '*.PLPI..*.D.2021.311', '*.PLPI..*.D.2021.312', '*.PLPI..*.D.2021.313', '*.PLPI..*.D.2021.314', '*.PLPI..*.D.2021.315', '*.PLPI..*.D.2021.316', '*.PLPI..*.D.2021.317', '*.PLPI..*.D.2021.318', '*.PLPI..*.D.2021.319', '*.PLPI..*.D.2021.320', '*.PLPI..*.D.2021.321', '*.PLPI..*.D.2021.322', '*.PLPI..*.D.2021.323', '*.PLPI..*.D.2021.324', '*.PLPI..*.D.2021.325', '*.PLPI..*.D.2021.326', '*.PLPI..*.D.2021.327', '*.PLPI..*.D.2021.328', '*.PLPI..*.D.2021.329', '*.PLPI..*.D.2021.330', '*.PLPI..*.D.2021.331', '*.PLPI..*.D.2021.332', '*.PLPI..*.D.2021.333', '*.PLPI..*.D.2021.334', '*.PLPI..*.D.2021.335', '*.PLPI..*.D.2021.336', '*.PLPI..*.D.2021.337', '*.PLPI..*.D.2021.338', '*.PLPI..*.D.2021.339', '*.PLPI..*.D.2021.340', '*.PLPI..*.D.2021.341', '*.PLPI..*.D.2021.342', '*.PLPI..*.D.2021.343', '*.PLPI..*.D.2021.344', '*.PLPI..*.D.2021.345', '*.PLPI..*.D.2021.346', '*.PLPI..*.D.2021.347', '*.PLPI..*.D.2021.348', '*.PLPI..*.D.2021.349', '*.PLPI..*.D.2021.350', '*.PLPI..*.D.2021.351', '*.PLPI..*.D.2021.352', '*.PLPI..*.D.2021.353', '*.PLPI..*.D.2021.354', '*.PLPI..*.D.2021.355', '*.PLPI..*.D.2021.356', '*.PLPI..*.D.2021.357', '*.PLPI..*.D.2021.358', '*.PLPI..*.D.2021.359', '*.PLPI..*.D.2021.360', '*.PLPI..*.D.2021.361', '*.PLPI..*.D.2021.362', '*.PLPI..*.D.2021.363', '*.PLPI..*.D.2021.364', '*.PLPI..*.D.2021.365', '*.PLPI..*.D.2022.001', '*.PLPI..*.D.2022.002', '*.PLPI..*.D.2022.003', '*.PLPI..*.D.2022.004', '*.PLPI..*.D.2022.005', '*.PLPI..*.D.2022.006', '*.PLPI..*.D.2022.007', '*.PLPI..*.D.2022.008', '*.PLPI..*.D.2022.009', '*.PLPI..*.D.2022.010', '*.PLPI..*.D.2022.011', '*.PLPI..*.D.2022.012', '*.PLPI..*.D.2022.013', '*.PLPI..*.D.2022.014', '*.PLPI..*.D.2022.015', '*.PLPI..*.D.2022.016', '*.PLPI..*.D.2022.017', '*.PLPI..*.D.2022.018', '*.PLPI..*.D.2022.019', '*.PLPI..*.D.2022.020', '*.PLPI..*.D.2022.021', '*.PLPI..*.D.2022.022', '*.PLPI..*.D.2022.023', '*.PLPI..*.D.2022.024', '*.PLPI..*.D.2022.025', '*.PLPI..*.D.2022.026', '*.PLPI..*.D.2022.027', '*.PLPI..*.D.2022.028', '*.PLPI..*.D.2022.029', '*.PLPI..*.D.2022.030', '*.PLPI..*.D.2022.031', '*.PLPI..*.D.2022.032']
files = ['*.PPMA..*.D.2021.213', '*.PPMA..*.D.2021.214', '*.PPMA..*.D.2021.215', '*.PPMA..*.D.2021.216', '*.PPMA..*.D.2021.217', '*.PPMA..*.D.2021.218', '*.PPMA..*.D.2021.219', '*.PPMA..*.D.2021.220', '*.PPMA..*.D.2021.221', '*.PPMA..*.D.2021.222', '*.PPMA..*.D.2021.223', '*.PPMA..*.D.2021.224', '*.PPMA..*.D.2021.225', '*.PPMA..*.D.2021.226', '*.PPMA..*.D.2021.227', '*.PPMA..*.D.2021.228', '*.PPMA..*.D.2021.229', '*.PPMA..*.D.2021.230', '*.PPMA..*.D.2021.231', '*.PPMA..*.D.2021.232', '*.PPMA..*.D.2021.233', '*.PPMA..*.D.2021.234', '*.PPMA..*.D.2021.235', '*.PPMA..*.D.2021.236', '*.PPMA..*.D.2021.237', '*.PPMA..*.D.2021.238', '*.PPMA..*.D.2021.239', '*.PPMA..*.D.2021.240', '*.PPMA..*.D.2021.241', '*.PPMA..*.D.2021.242', '*.PPMA..*.D.2021.243', '*.PPMA..*.D.2021.244', '*.PPMA..*.D.2021.245', '*.PPMA..*.D.2021.246', '*.PPMA..*.D.2021.247', '*.PPMA..*.D.2021.248', '*.PPMA..*.D.2021.249', '*.PPMA..*.D.2021.250', '*.PPMA..*.D.2021.251', '*.PPMA..*.D.2021.252', '*.PPMA..*.D.2021.253', '*.PPMA..*.D.2021.254', '*.PPMA..*.D.2021.255', '*.PPMA..*.D.2021.256', '*.PPMA..*.D.2021.257', '*.PPMA..*.D.2021.258', '*.PPMA..*.D.2021.259', '*.PPMA..*.D.2021.260', '*.PPMA..*.D.2021.261', '*.PPMA..*.D.2021.262', '*.PPMA..*.D.2021.263', '*.PPMA..*.D.2021.264', '*.PPMA..*.D.2021.265', '*.PPMA..*.D.2021.266', '*.PPMA..*.D.2021.267', '*.PPMA..*.D.2021.268', '*.PPMA..*.D.2021.269', '*.PPMA..*.D.2021.270', '*.PPMA..*.D.2021.271', '*.PPMA..*.D.2021.272', '*.PPMA..*.D.2021.273', '*.PPMA..*.D.2021.274', '*.PPMA..*.D.2021.275', '*.PPMA..*.D.2021.276', '*.PPMA..*.D.2021.277', '*.PPMA..*.D.2021.278', '*.PPMA..*.D.2021.279', '*.PPMA..*.D.2021.280', '*.PPMA..*.D.2021.281', '*.PPMA..*.D.2021.282', '*.PPMA..*.D.2021.283', '*.PPMA..*.D.2021.284', '*.PPMA..*.D.2021.285', '*.PPMA..*.D.2021.286', '*.PPMA..*.D.2021.287', '*.PPMA..*.D.2021.288', '*.PPMA..*.D.2021.289', '*.PPMA..*.D.2021.290', '*.PPMA..*.D.2021.291', '*.PPMA..*.D.2021.292', '*.PPMA..*.D.2021.293', '*.PPMA..*.D.2021.294', '*.PPMA..*.D.2021.295', '*.PPMA..*.D.2021.296', '*.PPMA..*.D.2021.297', '*.PPMA..*.D.2021.298', '*.PPMA..*.D.2021.299', '*.PPMA..*.D.2021.300', '*.PPMA..*.D.2021.301', '*.PPMA..*.D.2021.302', '*.PPMA..*.D.2021.303', '*.PPMA..*.D.2021.304', '*.PPMA..*.D.2021.305', '*.PPMA..*.D.2021.306', '*.PPMA..*.D.2021.307', '*.PPMA..*.D.2021.308', '*.PPMA..*.D.2021.309', '*.PPMA..*.D.2021.310', '*.PPMA..*.D.2021.311', '*.PPMA..*.D.2021.312', '*.PPMA..*.D.2021.313', '*.PPMA..*.D.2021.314', '*.PPMA..*.D.2021.315', '*.PPMA..*.D.2021.316', '*.PPMA..*.D.2021.317', '*.PPMA..*.D.2021.318', '*.PPMA..*.D.2021.319', '*.PPMA..*.D.2021.320', '*.PPMA..*.D.2021.321', '*.PPMA..*.D.2021.322', '*.PPMA..*.D.2021.323', '*.PPMA..*.D.2021.324', '*.PPMA..*.D.2021.325', '*.PPMA..*.D.2021.326', '*.PPMA..*.D.2021.327', '*.PPMA..*.D.2021.328', '*.PPMA..*.D.2021.329', '*.PPMA..*.D.2021.330', '*.PPMA..*.D.2021.331', '*.PPMA..*.D.2021.332', '*.PPMA..*.D.2021.333', '*.PPMA..*.D.2021.334', '*.PPMA..*.D.2021.335', '*.PPMA..*.D.2021.336', '*.PPMA..*.D.2021.337', '*.PPMA..*.D.2021.338', '*.PPMA..*.D.2021.339', '*.PPMA..*.D.2021.340', '*.PPMA..*.D.2021.341', '*.PPMA..*.D.2021.342', '*.PPMA..*.D.2021.343', '*.PPMA..*.D.2021.344', '*.PPMA..*.D.2021.345', '*.PPMA..*.D.2021.346', '*.PPMA..*.D.2021.347', '*.PPMA..*.D.2021.348', '*.PPMA..*.D.2021.349', '*.PPMA..*.D.2021.350', '*.PPMA..*.D.2021.351', '*.PPMA..*.D.2021.352', '*.PPMA..*.D.2021.353', '*.PPMA..*.D.2021.354', '*.PPMA..*.D.2021.355', '*.PPMA..*.D.2021.356', '*.PPMA..*.D.2021.357', '*.PPMA..*.D.2021.358', '*.PPMA..*.D.2021.359', '*.PPMA..*.D.2021.360', '*.PPMA..*.D.2021.361', '*.PPMA..*.D.2021.362', '*.PPMA..*.D.2021.363', '*.PPMA..*.D.2021.364', '*.PPMA..*.D.2021.365', '*.PPMA..*.D.2022.001', '*.PPMA..*.D.2022.002', '*.PPMA..*.D.2022.003', '*.PPMA..*.D.2022.004', '*.PPMA..*.D.2022.005', '*.PPMA..*.D.2022.006', '*.PPMA..*.D.2022.007', '*.PPMA..*.D.2022.008', '*.PPMA..*.D.2022.009', '*.PPMA..*.D.2022.010', '*.PPMA..*.D.2022.011', '*.PPMA..*.D.2022.012', '*.PPMA..*.D.2022.013', '*.PPMA..*.D.2022.014', '*.PPMA..*.D.2022.015', '*.PPMA..*.D.2022.016', '*.PPMA..*.D.2022.017', '*.PPMA..*.D.2022.018', '*.PPMA..*.D.2022.019', '*.PPMA..*.D.2022.020', '*.PPMA..*.D.2022.021', '*.PPMA..*.D.2022.022', '*.PPMA..*.D.2022.023', '*.PPMA..*.D.2022.024', '*.PPMA..*.D.2022.025', '*.PPMA..*.D.2022.026', '*.PPMA..*.D.2022.027', '*.PPMA..*.D.2022.028', '*.PPMA..*.D.2022.029', '*.PPMA..*.D.2022.030', '*.PPMA..*.D.2022.031', '*.PPMA..*.D.2022.032']
files = ['*.PPMA..HHN.D.2021.213', '*.PPMA..HHN.D.2021.214', '*.PPMA..HHN.D.2021.215', '*.PPMA..HHN.D.2021.216', '*.PPMA..HHN.D.2021.217', '*.PPMA..HHN.D.2021.218', '*.PPMA..HHN.D.2021.219', '*.PPMA..HHN.D.2021.220', '*.PPMA..HHN.D.2021.221', '*.PPMA..HHN.D.2021.222', '*.PPMA..HHN.D.2021.223', '*.PPMA..HHN.D.2021.224', '*.PPMA..HHN.D.2021.225', '*.PPMA..HHN.D.2021.226', '*.PPMA..HHN.D.2021.227', '*.PPMA..HHN.D.2021.228', '*.PPMA..HHN.D.2021.229', '*.PPMA..HHN.D.2021.230', '*.PPMA..HHN.D.2021.231', '*.PPMA..HHN.D.2021.232', '*.PPMA..HHN.D.2021.233', '*.PPMA..HHN.D.2021.234', '*.PPMA..HHN.D.2021.235', '*.PPMA..HHN.D.2021.236', '*.PPMA..HHN.D.2021.237', '*.PPMA..HHN.D.2021.238', '*.PPMA..HHN.D.2021.239', '*.PPMA..HHN.D.2021.240', '*.PPMA..HHN.D.2021.241', '*.PPMA..HHN.D.2021.242', '*.PPMA..HHN.D.2021.243', '*.PPMA..HHN.D.2021.244', '*.PPMA..HHN.D.2021.245', '*.PPMA..HHN.D.2021.246', '*.PPMA..HHN.D.2021.247', '*.PPMA..HHN.D.2021.248', '*.PPMA..HHN.D.2021.249', '*.PPMA..HHN.D.2021.250', '*.PPMA..HHN.D.2021.251', '*.PPMA..HHN.D.2021.252', '*.PPMA..HHN.D.2021.253', '*.PPMA..HHN.D.2021.254', '*.PPMA..HHN.D.2021.255', '*.PPMA..HHN.D.2021.256', '*.PPMA..HHN.D.2021.257', '*.PPMA..HHN.D.2021.258', '*.PPMA..HHN.D.2021.259', '*.PPMA..HHN.D.2021.260', '*.PPMA..HHN.D.2021.261', '*.PPMA..HHN.D.2021.262', '*.PPMA..HHN.D.2021.263', '*.PPMA..HHN.D.2021.264', '*.PPMA..HHN.D.2021.265', '*.PPMA..HHN.D.2021.266', '*.PPMA..HHN.D.2021.267', '*.PPMA..HHN.D.2021.268', '*.PPMA..HHN.D.2021.269', '*.PPMA..HHN.D.2021.270', '*.PPMA..HHN.D.2021.271', '*.PPMA..HHN.D.2021.272', '*.PPMA..HHN.D.2021.273', '*.PPMA..HHN.D.2021.274', '*.PPMA..HHN.D.2021.275', '*.PPMA..HHN.D.2021.276', '*.PPMA..HHN.D.2021.277', '*.PPMA..HHN.D.2021.278', '*.PPMA..HHN.D.2021.279', '*.PPMA..HHN.D.2021.280', '*.PPMA..HHN.D.2021.281', '*.PPMA..HHN.D.2021.282', '*.PPMA..HHN.D.2021.283', '*.PPMA..HHN.D.2021.284', '*.PPMA..HHN.D.2021.285', '*.PPMA..HHN.D.2021.286', '*.PPMA..HHN.D.2021.287', '*.PPMA..HHN.D.2021.288', '*.PPMA..HHN.D.2021.289', '*.PPMA..HHN.D.2021.290', '*.PPMA..HHN.D.2021.291', '*.PPMA..HHN.D.2021.292', '*.PPMA..HHN.D.2021.293', '*.PPMA..HHN.D.2021.294', '*.PPMA..HHN.D.2021.295', '*.PPMA..HHN.D.2021.296', '*.PPMA..HHN.D.2021.297', '*.PPMA..HHN.D.2021.298', '*.PPMA..HHN.D.2021.299', '*.PPMA..HHN.D.2021.300', '*.PPMA..HHN.D.2021.301', '*.PPMA..HHN.D.2021.302', '*.PPMA..HHN.D.2021.303', '*.PPMA..HHN.D.2021.304', '*.PPMA..HHN.D.2021.305', '*.PPMA..HHN.D.2021.306', '*.PPMA..HHN.D.2021.307', '*.PPMA..HHN.D.2021.308', '*.PPMA..HHN.D.2021.309', '*.PPMA..HHN.D.2021.310', '*.PPMA..HHN.D.2021.311', '*.PPMA..HHN.D.2021.312', '*.PPMA..HHN.D.2021.313', '*.PPMA..HHN.D.2021.314', '*.PPMA..HHN.D.2021.315', '*.PPMA..HHN.D.2021.316', '*.PPMA..HHN.D.2021.317', '*.PPMA..HHN.D.2021.318', '*.PPMA..HHN.D.2021.319', '*.PPMA..HHN.D.2021.320', '*.PPMA..HHN.D.2021.321', '*.PPMA..HHN.D.2021.322', '*.PPMA..HHN.D.2021.323', '*.PPMA..HHN.D.2021.324', '*.PPMA..HHN.D.2021.325', '*.PPMA..HHN.D.2021.326', '*.PPMA..HHN.D.2021.327', '*.PPMA..HHN.D.2021.328', '*.PPMA..HHN.D.2021.329', '*.PPMA..HHN.D.2021.330', '*.PPMA..HHN.D.2021.331', '*.PPMA..HHN.D.2021.332', '*.PPMA..HHN.D.2021.333', '*.PPMA..HHN.D.2021.334', '*.PPMA..HHN.D.2021.335', '*.PPMA..HHN.D.2021.336', '*.PPMA..HHN.D.2021.337', '*.PPMA..HHN.D.2021.338', '*.PPMA..HHN.D.2021.339', '*.PPMA..HHN.D.2021.340', '*.PPMA..HHN.D.2021.341', '*.PPMA..HHN.D.2021.342', '*.PPMA..HHN.D.2021.343', '*.PPMA..HHN.D.2021.344', '*.PPMA..HHN.D.2021.345', '*.PPMA..HHN.D.2021.346', '*.PPMA..HHN.D.2021.347', '*.PPMA..HHN.D.2021.348', '*.PPMA..HHN.D.2021.349', '*.PPMA..HHN.D.2021.350', '*.PPMA..HHN.D.2021.351', '*.PPMA..HHN.D.2021.352', '*.PPMA..HHN.D.2021.353', '*.PPMA..HHN.D.2021.354', '*.PPMA..HHN.D.2021.355', '*.PPMA..HHN.D.2021.356', '*.PPMA..HHN.D.2021.357', '*.PPMA..HHN.D.2021.358', '*.PPMA..HHN.D.2021.359', '*.PPMA..HHN.D.2021.360', '*.PPMA..HHN.D.2021.361', '*.PPMA..HHN.D.2021.362', '*.PPMA..HHN.D.2021.363', '*.PPMA..HHN.D.2021.364', '*.PPMA..HHN.D.2021.365', '*.PPMA..HHN.D.2022.001', '*.PPMA..HHN.D.2022.002', '*.PPMA..HHN.D.2022.003', '*.PPMA..HHN.D.2022.004', '*.PPMA..HHN.D.2022.005', '*.PPMA..HHN.D.2022.006', '*.PPMA..HHN.D.2022.007', '*.PPMA..HHN.D.2022.008', '*.PPMA..HHN.D.2022.009', '*.PPMA..HHN.D.2022.010', '*.PPMA..HHN.D.2022.011', '*.PPMA..HHN.D.2022.012', '*.PPMA..HHN.D.2022.013', '*.PPMA..HHN.D.2022.014', '*.PPMA..HHN.D.2022.015', '*.PPMA..HHN.D.2022.016', '*.PPMA..HHN.D.2022.017', '*.PPMA..HHN.D.2022.018', '*.PPMA..HHN.D.2022.019', '*.PPMA..HHN.D.2022.020', '*.PPMA..HHN.D.2022.021', '*.PPMA..HHN.D.2022.022', '*.PPMA..HHN.D.2022.023', '*.PPMA..HHN.D.2022.024', '*.PPMA..HHN.D.2022.025', '*.PPMA..HHN.D.2022.026', '*.PPMA..HHN.D.2022.027', '*.PPMA..HHN.D.2022.028', '*.PPMA..HHN.D.2022.029', '*.PPMA..HHN.D.2022.030', '*.PPMA..HHN.D.2022.031', '*.PPMA..HHN.D.2022.032']
# files = ['C7.PPMA..HHE.D.2021.262'] # PLPI, PPMA, PGAR
files = files[30:35]
title = ""
filepath = []

for file in files:
    filepath.append(path+file)
    
print(f"Reading files ...")

if type(filepath)==str:
    st = obspy.read(filepath)
else: 
    st = obspy.read(filepath[0])
    for i in range(1,len(filepath)):
        try:
            st+= obspy.read(filepath[i])
            print(f"Reading {filepath[i]} data...")
        except:print(f"{filepath[i]} not read.")

# st.trim( UTCDateTime("2021-09-19T00:00:00"),
        #  UTCDateTime("2021-09-19T00:10:00"))

print(st.__str__(extended=True))

print(f"Merging {len(st)} traces ...")
st.merge()
tr = st[0]

detrend = "constant" # Apply demean on each window

# %% STFT parameters
n_channels = len(st)
sr = tr.stats.sampling_rate 
window_length = 1800.0     # s. Length of the windows in seconds
window_samples = int(window_length*sr)
shift  = window_length/2  # s. Length of the window shift in seconds
n_bins = int(window_samples);  # Number of bins to use for the STFT calculation


npts = max(tr.stats.npts for tr in st)
n_windows = int(round(npts/window_samples))
n_freqs = int((n_bins//2)) 

# %% Calculate the STFT (Spectrogram)
print(f"Calculating STFT ...")
times_stft = np.zeros((n_channels, n_windows))
freqs_stft = np.zeros((n_channels, n_freqs))
Zre = np.zeros((n_channels,n_freqs, n_windows))
Zim = np.zeros((n_channels,n_freqs, n_windows))
Sxx = np.zeros((n_channels,n_freqs, n_windows))

signals = []
for i in range(len(st)):
    signals.append(st[i].data)

startime = st[-1].stats.starttime.datetime
endtime = st[-1].stats.endtime.datetime
# del st

for i in range(n_channels):
    print(f"Computing for trace {i}...")
    time, freq, Sx = features.stft(st[i].data, sr, window_samples, "hann", "odd",
                                   detrend,n_bins, t_phase=window_length/2)
    times_stft[i,:] = time[:-1]
    freqs_stft[i,:] = freq[:-1]
    Zre[i,:,:] = Sx[:-1, :-1].real
    Zim[i,:,:] = Sx[:-1, :-1].imag
Sxx = Zre**2 + Zim**2

# %% Save Spectrogram data
if save_data:
    for i in range(n_channels):
        timeUTC = np.linspace(startime, endtime, num= len(times_stft[i]), dtype='datetime64[h]').astype("datetime64[m]")
        print(timeUTC)
        features.save(f"data/involcan_{st[i].stats.station}_channel_{st[i].stats.channel}.mat",np.squeeze(Sxx[i]).T,row_names=timeUTC,column_names=freqs_stft[i])
        print(f"Data saved at: data/involcan_{st[i].stats.station}_channel_{st[i].stats.channel}.mat")




# %% Plot signal

if show_plots:
    print("Plotting signal and spectrogram")
    fig, axes = plt.subplots(2,1, figsize = (14,3), sharex=True)
    _,_,mesh = plot.spectrogram(times_stft[i], freqs_stft[i], Sxx[i], ax=axes[1],
                                                # vmin=0, vmax=np.max(Sxx),
                                                logscale=True)
    timeUTC = np.linspace(startime, endtime, num= len(times_stft[0]), dtype='datetime64[h]').astype("datetime64[m]")
    aux = mdates.date2num(timeUTC)
    skip:int = n_windows//5
    locs = times_stft[0, 0::skip]
    timeUTC = timeUTC[0::skip]
    date_formatter = mdates.DateFormatter('%Y/%m/%d %H:%M')
    formatted_labels = [date_formatter(mdates.date2num(t)) for t in timeUTC]
    axes[1].set_xticks(locs[:], labels=formatted_labels[:],fontsize=14)

    signal = st[0].data

    axes[0].plot(np.linspace(0, np.max(times_stft), len(signal)),signal, c='black')
    axes[0].set_xlim(0, np.max(times_stft))

    labelsize = 14
    axes[0].set_ylabel("Amplitude", fontsize = labelsize)
    axes[1].set_ylabel("Frequency \n(Hz)", fontsize = labelsize)

    ticksize = 12
    axes[0].tick_params(axis='y', labelsize=ticksize)
    axes[1].tick_params(axis='y', labelsize=ticksize)


    tbox = axes[0].get_position()
    bbox = axes[1].get_position()

    fig.subplots_adjust(right=0.86)
    cbar_ax = fig.add_axes([0.87, bbox.y0, 0.01, bbox.height])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=ticksize) 
    cbar.ax.yaxis.get_offset_text().set_fontsize(ticksize)
    offset_text = cbar.ax.yaxis.get_offset_text()
    offset_text.set_position((2.5, offset_text.get_position()[1]))


    fig.savefig("external/tutorials/double_plot.png", bbox_inches='tight')
    print(f"Plot saved at: external/tutorials/double_plot.png")

input("End?")