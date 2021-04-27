import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import  pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, DenseNet121, ResNet50, DenseNet201
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from dataclasses import dataclass, asdict
import yaml
from typing import Optional, Union, List
from dacite import from_dict
import numpy as np
import math
import json
import itertools
import ast
import cv2
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class LossParams:
  func: str
  params: Optional[str]

@dataclass
class OptimizerParams:
  func: str
  params: Optional[str]

@dataclass
class InputDataParams:
  input_size: str

@dataclass
class ModelParams:
  batch_size: int
  arch: str
  freeze_pretrained: bool
  steps_per_epoch: int
  metrics: List[str]
  pretrained_weight: Optional[str]
  loss: Optional[LossParams]
  optimizer: Optional[OptimizerParams]
  class_weight_mu: float

@dataclass
class ModelsConfig:
  model_name: str
  input_params: InputDataParams
  model_params: ModelParams


class ModelTrainer():

  def __init__(self, train_dir, test_dir, model_name, model_dir, batch_size = 16, 
               target_size = (224,224), model_params = None, class_weight_mu = 0.4, retrain = False):
    self.train_dir = train_dir
    self.test_dir = test_dir
    self.batch_size = batch_size
    self.target_size = target_size
    self.model_name = model_name
    self.model_dir = model_dir
    self.checkpoint_path = f"{self.model_dir}/{self.model_name}"
    self.train_generator, self.validation_generator = self.get_generators()
    self.set_class_weight(class_weight_mu)
    self.model_params = model_params
    if retrain == True or not(self.load_trained_model()):
      if model_params:
        self.register_model(self.model_architecture(model_params))

  @staticmethod
  def load_from_config(config, base_dir = '/content/gdrive/MyDrive/Capstone', retrain = False):
    train_dir = f"{base_dir}/dataset/HAM10000/HAM10000_train_by_class/"
    test_dir = f"{base_dir}/dataset/HAM10000/HAM10000_test_by_class/"
    model_dir = f"{base_dir}/dataset/models"
    cfg = from_dict(data_class=ModelsConfig, data=config)
    model_trainer = ModelTrainer(train_dir, test_dir, cfg.model_name, model_dir, batch_size = cfg.model_params.batch_size or 16,
                                 target_size = eval(cfg.input_params.input_size),class_weight_mu = cfg.model_params.class_weight_mu,
                                 model_params = cfg.model_params, retrain = retrain)    
    return model_trainer

  @classmethod
  def img_normalize(cls,img):
    #img = cv2.imread(img)
    mean = [0.5456423, 0.5700427, 0.7630366]
    sd = [0.15261365, 0.16997027, 0.14092803]
    img = cv2.resize(img, (224,224))
    img = img.astype('float32')/255.
    img = (img - mean) / sd
    #img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    return img       
  
  def register_model(self, model):
    self.model = model
    self.model.compile(loss=self.model_params.loss.func, optimizer=self.optimizer(), metrics=self.model_params.metrics)
    if not os.path.exists(self.checkpoint_path):
      os.makedirs(self.checkpoint_path)
    self.model.save(f"{self.checkpoint_path}/model.h5")

  def optimizer(self):
    optimizer_func = eval(self.model_params.optimizer.func)
    optimizer_params = ast.literal_eval(self.model_params.optimizer.params)
    optimizer = optimizer_func(**optimizer_params)
    return optimizer

  def summary(self):
    self.model.summary()

  def show_samples(self, rows = 4, columns = 4):
    x, y = next(self.train_generator)
    fig = plt.figure(figsize=(8, 8))
    for i in range(0, columns*rows):
      img = x[i].astype(int)
      fig.add_subplot(rows, columns, i+1)
      plt.imshow(img)
    plt.show()
  
  def get_generators(self):
    #drive.mount('/content/gdrive',force_remount=True)
    train_datagen = ImageDataGenerator(
                                  #featurewise_center=True, 
                                  #featurewise_std_normalization=True,
                                  #rotation_range = 20,
                                  #width_shift_range = 0.2,
                                  #height_shift_range = 0.2,
                                  #shear_range = 0.2,
                                  horizontal_flip = True,
                                  vertical_flip = True,
                                  preprocessing_function = ModelTrainer.img_normalize)
                                  #fill_mode = 'nearest')
    test_datagen = ImageDataGenerator(preprocessing_function = ModelTrainer.img_normalize
                                   #featurewise_center=True, 
                                   #featurewise_std_normalization=True
                                     )
    train_generator = train_datagen.flow_from_directory(directory=self.train_dir, class_mode='sparse',shuffle=True,
                                                  batch_size=self.batch_size,target_size=self.target_size)
    validation_generator = test_datagen.flow_from_directory(directory=self.test_dir, class_mode='sparse',shuffle=False,
                                                       batch_size=self.batch_size,target_size=self.target_size)
    return train_generator, validation_generator
  
  def create_class_weight(self, labels_dict, mu):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
      score = math.log(mu*total/float(labels_dict[key]))
      score = mu*total/float(labels_dict[key])
      class_weight[key] = score if score > 1.0 else 1.0
    return class_weight

  def set_class_weight(self, mu):  
    class_dict = dict()
    for dir in os.listdir(self.train_dir):
      class_dict[dir] = len(os.listdir(f"{self.train_dir}/{dir}"))

    weights = self.create_class_weight(class_dict, mu)
    self.class_weight = {}
    class_indices = self.train_generator.class_indices
    for cls in weights:
      self.class_weight[class_indices[cls]] = weights[cls]
  
  def _callback(self):
    filepath = self.checkpoint_path + '/weights.h5'
    checkpoint_dir = os.path.dirname(self.checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                    save_weights_only=True,
                                                    verbose=2,
                                                    save_best_only=True)
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
    return [cp_callback,learning_rate_reduction]
  
  def model_architecture(self, model_params, CLASS_N = 7):
    arch = eval(model_params.arch)
    pretrained = arch(input_shape = (224,224,3), include_top=False, weights=model_params.pretrained_weight or None)

    for layer in pretrained.layers:
      layer.trainable = not(model_params.freeze_pretrained)
    
    x = Flatten()(pretrained.layers[-1].output)
    #x = Dense(5000, kernel_regularizer=regularizers.l1_l2(0.00001), activity_regularizer=regularizers.l2(0.00001), activation='relu',kernel_initializer=tf.keras.initializers.he_normal())(x) 
    #x = Dropout(0.5)(x)
    #x = Dense(2000, kernel_regularizer=regularizers.l1_l2(0.00001), activity_regularizer=regularizers.l2(0.00001), activation='relu',kernel_initializer=tf.keras.initializers.he_normal())(x) 
    #x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(7, activation = 'softmax')(x)
    
    model = Model(inputs = pretrained.input, outputs = x)
    print("New model created")
    return model
  
  def load_trained_model(self):
    if os.path.exists(self.checkpoint_path):
      print("Trained model exists and it will be loaded")
      self.model = load_model(f'{self.checkpoint_path}/model.h5')
      self.model.load_weights(f'{self.checkpoint_path}/weights.h5')
      return True
    return False
  
  def train(self, epochs=10, verbose=2):
    with tf.device('/device:GPU:0'):
      model_info = self.model.fit(
                      x=self.train_generator, 
                      steps_per_epoch=self.train_generator.samples // self.batch_size+1,  
                      epochs=epochs, 
                      validation_steps=self.validation_generator.samples // self.batch_size+1,
                      validation_data=self.validation_generator, 
                      verbose=verbose,
                      callbacks=self._callback(),
                      class_weight=self.class_weight
                  )
      self.model_info = model_info
      with open(f'{self.model_dir}/history.json','w') as fp:
        json.dump(str(self.model_info.history), fp)
    
  def confusion_matrix(self):    
    Y_pred = self.model.predict(self.validation_generator, self.validation_generator.samples // self.batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(self.validation_generator.classes, y_pred)
    target_names = list(model_trainer.validation_generator.class_indices.keys())
    cls_rpt = classification_report(self.validation_generator.classes, y_pred, target_names=target_names)
    #self.plot_confusion_matrix(cm, target_names)

  def plot_confusion_matrix(self):
    
    normalize=False
    title='Confusion matrix'
    cmap=plt.cm.Blues
    
    Y_pred = self.model.predict(self.validation_generator, self.validation_generator.samples // self.batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(self.validation_generator.classes, y_pred)
    classes = list(model_trainer.validation_generator.class_indices.keys())
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')  
    
    cls_rpt = classification_report(self.validation_generator.classes, y_pred, target_names=classes)
    print(cls_rpt)
