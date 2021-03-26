def ham_10000_vgg16():
    adam = optimizers.Adam(lr=0.05)
    vgg = VGG16(input_shape=(32, 32, 3), weights='imagenet', include_top=False)

    for layer in vgg.layers:
        layer.trainable = False

    x = Flatten()(vgg.layers[-1].output)
    # x = Dense(2000, kernel_regularizer=regularizers.l1_l2(0.00001), activity_regularizer=regularizers.l2(0.00001), activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(7, activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=x)
    return model


if __name__ == "__main__":
    name = 'ham_10000_vgg16'
    model_config = yaml.safe_load(yaml_config)
    model_trainer = ModelTrainer.load_from_config(model_config[name])

    model = ham_10000_vgg16()
    model_trainer.register_model(model)

    model_trainer.train()