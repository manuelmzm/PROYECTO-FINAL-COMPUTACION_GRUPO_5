# Importar librerias
import pandas
import numpy as np
from model import Sequential_model
from sklearn.model_selection import train_test_split
from azureml.core import Workspace, Dataset

# Obtener el dataset de iris del Blob Datastore
subscription_id = '<subscription_id>'
resource_group = 'myMLresources'
workspace_name = 'myMLworkspace'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='dataset_procesed')
dataset.to_pandas_dataframe()

data = dataset.values

x_train, x_test, y_train, y_test = train_test_split(data[:,0:4], data[:,4], test_size=0.33, random_state=42)

model = Sequential_model.get_model()

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
    
model.fit(x_train, y_train, epochs=1500, verbose=0)