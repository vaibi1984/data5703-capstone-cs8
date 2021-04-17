def ham_10000_xception():
    adam = optimizers.Adam(lr=0.05)
    xcept = Xception(input_shape=(224,224,3), weights='imagenet', include_top=False,pooling = 'max')

    for layer in xcept.layers:
        layer.trainable = False

    x = Flatten()(xcept.layers[-1].output)
    x = Dense(7, activation='softmax')(x)
    model = Model(inputs=xcept.input, outputs=x)
    return model


if __name__ == "__main__":
    name = 'ham_10000_xception'
    model_config = yaml.safe_load(yaml_config)
    model_trainer = ModelTrainer.load_from_config(model_config[name])

    model = ham_10000_xception()
    model_trainer.register_model(model)

    model_trainer.train()