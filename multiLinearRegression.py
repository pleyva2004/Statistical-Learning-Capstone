import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


from loadData import load_data

df = load_data()
print(df.head())
