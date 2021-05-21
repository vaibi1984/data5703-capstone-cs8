def ham_10000_resnet50(CLASS_N=3):

    #model pre-trained on imagenet dataset
    resnet_base = ResNet50(input_shape = (600,450,3), weights = 'imagenet', include_top = False, pooling = 'avg') 

    for layer in resnet_base.layers:
        layer.trainable = False

    x = Flatten()(resnet_base.layers[-1].output)
    x = Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)#regulariser reduces overfitting
    x = Dropout(0.5)(x) #50% change in the output of neuron made 0 # also reduces overfitting
    x = Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)#regulariser reduces overfitting
    x = Dropout(0.5)(x) #50% change in the output of neuron made 0 # also reduces overfitting
    x = Dense(CLASS_N, activation = 'softmax', kernel_regularizer = regularizers.l2(0.001))(x)

    model = Model(inputs = resnet_base.input, outputs = x)

    return model


if __name__ == "__main__":
    name = 'ham_10000_resnet50'
    model_config3 = yaml.safe_load(yaml_config3)
    model_trainer = ModelTrainer.load_from_config(model_config3[name],'/home/alol_elba/download/anaconda3/Capstone', 
                                               retrain = True)
    model_trainer.class_weight = {0:2.0,1:2.0,2:1.0}
    model_trainer.train(epochs=33, verbose = 1)
    model_trainer.plot_confusion_matrix()
