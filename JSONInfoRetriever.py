from tensorflow.keras import layers


def get_name(model):
    name = model["Name"]
    print(name)
    return name


def get_loss_func(model):
    lossfunc_dict = {
        'Rectified Linear Unit': 'relu',
        'Linear': 'linear',
        'Exponential Linear Unit': 'elu',
        'Exponential': 'exponential',
        'Scaled Exponential Linear Unit': 'selu',
        'Hyperbolic Tangent': 'tanh',
        'Sigmoid': 'sigmoid',
        'Hard Sigmoid': 'hard_sigmoid',
        'Softmax': 'softmax',
        'Softplus': 'softplus',
        'Softsign': 'softsign',
        'Categorical Crossentropy': 'categorical_crossentropy',
        'Binary Crossentropy': 'binary_crossentropy',
        'Sparse Categorical Crossentropy': 'sparse_categorical_crossentropy',
        'Mean Squared Error': 'mean_squared_error'
    }
    lossfunc = lossfunc_dict.get(model["LossFunction"])
    print(lossfunc)
    return lossfunc


def get_optimizer(model):
    opt_dict = {
        'Adam Optimizer': 'adam',
        'Stochastic Gradient Descent Optimizer': 'sgd',
        'Nadam Optimizer': 'nada',
        'Adamax Optimizer': 'adamax',
        'Adagrad Optimizer': 'adag',
        'Adadelta Optimizer': 'adam',
        'RMSProp Optimizer': 'rms',
        'Ftrl Optimizer': 'ftrl'
    }
    opt = opt_dict.get(model["Optimizer"])
    print(opt)
    return opt


def get_layers(model):
    res_layers = model["Layers"]
    return res_layers


def get_layer_type(layer):
    layer_dict = {
        "Convolutional 1D Layer": layers.Conv1D(32, (3)),
        "Convolutional 2D Layer": layers.Conv2D(32, (3)),
        "Convolutional 3D Layer": layers.Conv3D(32, (3)),
        "Convolutional 1D Transpose Layer": layers.Conv1DTranspose(32, (3)),
        "Convolutional 2D Transpose Layer": layers.Conv2DTranspose(32, (3)),
        "Convolutional 3D Transpose Layer": layers.Conv3DTranspose(32, (3)),
        "Cropping 1D Layer": layers.Cropping1D(),
        "Cropping 2D Layer": layers.Cropping2D(),
        "Cropping 3D Layer": layers.Cropping3D(),
        "Max Pooling 1D Layer": layers.MaxPool1D((2)),
        "Max Pooling 2D Layer": layers.MaxPool2D((2)),
        "Max Pooling 3D Layer": layers.MaxPool2D((2)),
        "Separable Convolutional 1D Layer": layers.SeparableConv1D(32, (3)),
        "Separable Convolutional 2D Layer": layers.SeparableConv2D(32, (3)),
        "UpSampling 1D Layer": layers.UpSampling1D(),
        "UpSampling 2D Layer": layers.UpSampling2D(),
        "UpSampling 3D Layer": layers.UpSampling3D(),
        "Zero Padding 1D Layer": layers.ZeroPadding1D(),
        "Zero Padding 2D Layer": layers.ZeroPadding2D(),
        "Zero Padding 3D Layer": layers.ZeroPadding3D(),
        "Dense Layer": layers.Dense(10),
        "Locally Connected 1D Layer": layers.LocallyConnected1D(32, (3)),
        "Locally Connected 2D Layer": layers.LocallyConnected2D(32, (3)),
        "Average Pooling 1D Layer": layers.AveragePooling1D((2)),
        "Average Pooling 2D Layer": layers.AveragePooling2D((2)),
        "Average Pooling 3D Layer": layers.AveragePooling3D((2)),
        "Global Max Pooling 1D Layer": layers.GlobalMaxPooling1D(),
        "Global Max Pooling 2D Layer": layers.GlobalMaxPooling2D(),
        "Global Max Pooling 3D Layer": layers.GlobalMaxPooling3D(),
        "Global Average Pooling 1D Layer": layers.GlobalAveragePooling1D(),
        "Global Average Pooling 2D Layer": layers.GlobalAveragePooling2D(),
        "Global Average Pooling 3D Layer": layers.GlobalAveragePooling3D(),
        "Input Layer": layers.Input((32, 32)),
        "Dropout Layer": layers.Dropout(0.5),
        "Batch Normalization Layer": layers.BatchNormalization(),
        "Flatten Layer": layers.Flatten(),
        "Activation Layer": layers.Activation("relu"),
        "Custom Layer": "Placeholder"
    }
    type = layer_dict.get(layer["LayerType"])
    return type