#Python version
from platform import python_version

#Google mount and reading data
import os
from google.colab import drive

#TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50 #ResNet50
from tensorflow.keras.applications import VGG16 #VGG16
from tensorflow.keras import layers, models, optimizers, applications, regularizers
from tensorflow.keras.models import load_model

#Matplotlib
from matplotlib import pyplot as plt

#Dataclasses
from dataclasses import dataclass, asdict

#Yaml
import yaml

#Typing
from typing import Optional, Union, List

#Datacite
from dacite import from_dict

# To look better/efficiency
import warnings #removes warnings
import timeit #checks execution time of code. Example of how to use it here: https://pythonhow.com/measure-execution-time-python-code/


print("python version used in colab:",python_version())
seed(2021)
## uncomment below code when finalising code
  #warnings.filterwarnings('ignore')


@dataclass
class ModelsConfig:
    model_name: str
    model_dir: str
    train_dir: str
    batch_size: Optional[int]
    optimizer: str
    loss: str
    metrics: List[str]


class ModelTrainer():

    def __init__(self, train_dir, model_name, model_dir, batch_size=16, target_size=(32, 32), \
                 loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.model_name = model_name
        self.model_dir = model_dir
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.checkpoint_path = f"{self.model_dir}/{self.model_name}"
        self.train_generator, self.validation_generator = self.get_generators()

    @staticmethod
    def load_from_config(config):
        cfg = from_dict(data_class=ModelsConfig, data=config)
        model_trainer = ModelTrainer(cfg.train_dir, cfg.model_name, cfg.model_dir, cfg.batch_size or 16)
        return model_trainer

    def register_model(self, model):
        self.model = model
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.model.save(f"{self.checkpoint_path}/model.h5")

    def show_samples(self, rows=4, columns=4):
        x, y = next(self.train_generator)
        fig = plt.figure(figsize=(8, 8))
        for i in range(0, columns * rows):
            img = x[i].astype(int)
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(img)
        plt.show()

    def get_generators(self):
        drive.mount('/content/gdrive', force_remount=True)
        datagen = ImageDataGenerator(validation_split=0.2)
        train_generator = datagen.flow_from_directory(directory=self.train_dir, class_mode='categorical',
                                                      batch_size=self.batch_size, target_size=self.target_size,
                                                      subset='training')
        validation_generator = datagen.flow_from_directory(directory=self.train_dir, class_mode='categorical',
                                                           batch_size=self.batch_size, target_size=self.target_size,
                                                           subset='validation')
        return train_generator, validation_generator

    def _callback(self):
        filepath = self.checkpoint_path + '/{epoch:02d}-loss{val_loss:.2f}.hdf5'
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                         save_weights_only=True,
                                                         verbose=2,
                                                         save_best_only=True)
        return cp_callback

    def train(self, steps_per_epoch=10, epochs=10, validation_steps=10, verbose=2):
        with tf.device('/device:GPU:0'):
            model_info = self.model.fit(
                x=self.train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_steps=validation_steps,
                validation_data=self.validation_generator,
                verbose=verbose,
                callbacks=[self._callback()]
            )
