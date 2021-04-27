def ham_10000_resnet50():
  adam = optimizers.Adam
  resnet = ResNet50(input_shape = (32,32,3), weights = 'imagenet', include_top = False, pooling = 'max') #model pre-trained on imagenet dataset

  for layer in resnet.layers:
      layer.trainable = True

  x = Flatten()(resnet.layers[-1].output) #extra flatten layer with dense layer to be interpreted as probability
  x = Dense(7, activation='softmax', name='softmax')(x) #extra flatten layer with dense layer to be interpreted as probability
  model = Model(inputs = resnet.input, outputs = x) #defining inputs and outputs in model

  return model


if __name__ == "__main__":
    name = 'ham_10000_resnet50'
    model_config3 = yaml.safe_load(yaml_config3)
    model_trainer3 = ModelTrainer.load_from_config(model_config3[name])
    model3 = ham_10000_base()
    model_trainer3.register_model(model3)
    model_trainer.train()
