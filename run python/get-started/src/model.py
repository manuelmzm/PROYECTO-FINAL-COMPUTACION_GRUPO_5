import tensorflow as tf
import itertools                       

class Sequential_model(tf.keras.Model):
  def __init__(self, name=None):
    super(Sequential_model, self).__init__(name=name)
    self.dense_1 = tf.keras.layers.Flatten(input_shape=(4,))
    self.dense_2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
    self.dense_3 = tf.keras.layers.Dropout(0.2)
    self.pred_layer = tf.keras.layers.Dense(3, activation=tf.nn.softmax)
  
  def call(self, inputs):
    x = self.dense_1(inputs)
    x = self.dense_2(x)
    x = self.dense_3(x)
    return self.pred_layer(x)
    
  def get_model():
    return Sequential_model(name="Sequential")