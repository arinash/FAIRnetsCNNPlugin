from tensorflow.keras import layers
from tensorflow.keras import optimizers

model = None

def get_name():
    name = model["Name"]
    return name

def get_loss_func():
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
    return lossfunc


def get_optimizer(learning_rate):
    opt_dict = {
        'Adam Optimizer': optimizers.Adam(learning_rate),
        'Stochastic Gradient Descent Optimizer': optimizers.SGD(learning_rate),
        'Nadam Optimizer': optimizers.Nadam(learning_rate),
        'Adamax Optimizer': optimizers.Adamax(learning_rate),
        'Adagrad Optimizer': optimizers.Adagrad(learning_rate),
        'Adadelta Optimizer': optimizers.Adadelta(learning_rate),
        'RMSProp Optimizer': optimizers.RMSprop(learning_rate),
        'Ftrl Optimizer': optimizers.Ftrl(learning_rate)
    }
    opt = opt_dict.get(model["Optimizer"])
    return opt


def get_layers():
    res_layers = model["Layers"]
    return res_layers

def get_layers_numer():
    return len(model["Layers"])

def get_layer_type(layer, dropout, dense_activ):
    layer_dict = {
        "Convolutional 1D Layer": layers.Conv1D(64, (2)),
        "Convolutional 2D Layer": layers.Conv2D(64, (2)),
        "Convolutional 3D Layer": layers.Conv3D(64, (2)),
        "Convolutional 1D Transpose Layer": layers.Conv1DTranspose(64, (2)),
        "Convolutional 2D Transpose Layer": layers.Conv2DTranspose(64, (2)),
        "Convolutional 3D Transpose Layer": layers.Conv3DTranspose(64, (2)),
        "Cropping 1D Layer": layers.Cropping1D(),
        "Cropping 2D Layer": layers.Cropping2D(),
        "Cropping 3D Layer": layers.Cropping3D(),
        "Max Pooling 1D Layer": layers.MaxPool1D((2)),
        "Max Pooling 2D Layer": layers.MaxPool2D((2)),
        "Max Pooling 3D Layer": layers.MaxPool2D((2)),
        "Separable Convolutional 1D Layer": layers.SeparableConv1D(64, (2)),
        "Separable Convolutional 2D Layer": layers.SeparableConv2D(64, (2)),
        "UpSampling 1D Layer": layers.UpSampling1D(),
        "UpSampling 2D Layer": layers.UpSampling2D(),
        "UpSampling 3D Layer": layers.UpSampling3D(),
        "Zero Padding 1D Layer": layers.ZeroPadding1D(),
        "Zero Padding 2D Layer": layers.ZeroPadding2D(),
        "Zero Padding 3D Layer": layers.ZeroPadding3D(),
        "Dense Layer": layers.Dense(10, activation=dense_activ),
        "Locally Connected 1D Layer": layers.LocallyConnected1D(64, (2)),
        "Locally Connected 2D Layer": layers.LocallyConnected2D(64, (2)),
        "Average Pooling 1D Layer": layers.AveragePooling1D((2)),
        "Average Pooling 2D Layer": layers.AveragePooling2D((2)),
        "Average Pooling 3D Layer": layers.AveragePooling3D((2)),
        "Global Max Pooling 1D Layer": layers.GlobalMaxPooling1D(),
        "Global Max Pooling 2D Layer": layers.GlobalMaxPooling2D(),
        "Global Max Pooling 3D Layer": layers.GlobalMaxPooling3D(),
        "Global Average Pooling 1D Layer": layers.GlobalAveragePooling1D(),
        "Global Average Pooling 2D Layer": layers.GlobalAveragePooling2D(),
        "Global Average Pooling 3D Layer": layers.GlobalAveragePooling3D(),
        "Dropout Layer": layers.Dropout(dropout),
        "Batch Normalization Layer": layers.BatchNormalization(),
        "Flatten Layer": layers.Flatten(),
        "Activation Layer": layers.Activation("relu"),
        "Custom Layer": "Placeholder"
    }
    type = layer_dict.get(layer["LayerType"])
    return type