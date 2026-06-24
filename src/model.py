import tensorflow as tf


def build_mlp(input_shape, num_classes=1, hidden_units=(64, 32)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    for u in hidden_units:
        x = tf.keras.layers.Dense(u, activation="relu")(x)
    if num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        loss = "mse"
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_cnn2d(input_shape, num_classes=1, filters=(16, 32)):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for f in filters:
        x = tf.keras.layers.Conv2D(f, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    if num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        loss = "mse"
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_ds_cnn(input_shape, num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    if num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        loss = "mse"
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_conv1d(input_shape, num_classes=1, filters=(16, 32)):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for f in filters:
        x = tf.keras.layers.Conv1D(f, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    if num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        loss = "mse"
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_lstm(input_shape, num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(32)(inputs)
    if num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        loss = "mse"
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_model(model_type, input_shape, num_classes=1, **kwargs):
    model_type = model_type.lower()
    if model_type == "mlp":
        return build_mlp(input_shape, num_classes=num_classes, **kwargs)
    if model_type == "cnn2d":
        return build_cnn2d(input_shape, num_classes=num_classes, **kwargs)
    if model_type == "ds_cnn":
        return build_ds_cnn(input_shape, num_classes=num_classes)
    if model_type == "conv1d":
        return build_conv1d(input_shape, num_classes=num_classes, **kwargs)
    if model_type == "lstm":
        return build_lstm(input_shape, num_classes=num_classes)
    raise ValueError(f"Unknown model_type '{model_type}'. Choose from mlp, cnn2d, ds_cnn, conv1d, lstm.")


def build_cnn2d(input_shape, num_classes=1, filters=(16, 32)):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for f in filters:
        x = tf.keras.layers.Conv2D(f, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    if num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        loss = "mse"
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_ds_cnn(input_shape, num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    if num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        loss = "mse"
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_conv1d(input_shape, num_classes=1, filters=(16, 32)):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for f in filters:
        x = tf.keras.layers.Conv1D(f, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    if num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        loss = "mse"
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_lstm(input_shape, num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(32)(inputs)
    if num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        loss = "mse"
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_model(model_type, input_shape, num_classes=1, **kwargs):
    model_type = model_type.lower()
    if model_type == "mlp":
        return build_mlp(input_shape, num_classes=num_classes, **kwargs)
    if model_type == "cnn2d":
        return build_cnn2d(input_shape, num_classes=num_classes, **kwargs)
    if model_type == "ds_cnn":
        return build_ds_cnn(input_shape, num_classes=num_classes)
    if model_type == "conv1d":
        return build_conv1d(input_shape, num_classes=num_classes, **kwargs)
    if model_type == "lstm":
        return build_lstm(input_shape, num_classes=num_classes)
    raise ValueError(f"Unknown model_type '{model_type}'. Choose from mlp, cnn2d, ds_cnn, conv1d, lstm.")
