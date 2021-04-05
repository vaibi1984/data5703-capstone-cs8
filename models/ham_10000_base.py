# code modified from: https://www.tensorflow.org/tutorials/images/cnn
#base function
def ham_10000_base():
  adam = optimizers.Adam

  model = models.Sequential() # Sequential model type allows to build CNN model layer by layer
  model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3))) #convolution layer seen as 2D matrices, #activation function is rectified linear activation directly outputs input if its pos value
  model.add(layers.MaxPooling2D((2, 2))) #dimensionality reduction computing maximum value in each window of 2x2
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten()) #connects Conv2D and dense layers
  model.add(layers.Dense(64, activation='softmax')) #output layer, changed to softmax activation function so output can be interpreted as a probability
  model.add(layers.Dense(7)) #output layer

  return model


if __name__ == "__main__":
    name = 'ham_10000_base'
    model_config2 = yaml.safe_load(yaml_config2)
    model_trainer2 = ModelTrainer.load_from_config(model_config2[name])
    model2 = ham_10000_base()
    model_trainer2.register_model(model2)
    model_trainer2.train()
