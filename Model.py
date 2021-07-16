import json
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from sklearn.model_selection import GridSearchCV
import JSONInfoRetriever as inf


def load_data(dataset_name):
    res_dict = {}
    data_dict = {
        'mnist': keras.datasets.mnist,
        'fashion mnist': keras.datasets.fashion_mnist,
        'cifar10': keras.datasets.cifar10
    }
    data = data_dict[dataset_name]
    (x_train, y_train), (x_test, y_test) = data.load_data()
    sample_shape = x_train[0].shape
    input_shape = (sample_shape[0], sample_shape[1], 1)
    x_train = x_train.reshape(len(x_train), input_shape[0], input_shape[1], input_shape[2])
    x_test = x_test.reshape(len(x_test), input_shape[0], input_shape[1], input_shape[2])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalizing the data
    x_train /= 255
    x_test /= 255

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    res_dict["input_shape"] = input_shape
    res_dict["x_train"] = x_train
    res_dict["x_test"] = x_test
    res_dict["y_train"] = y_train
    res_dict["y_test"] = y_test
    return res_dict

#Create model with Keras Sequential API
def create_model(given_model, input_shape):
    lossfunc = inf.get_loss_func(given_model)
    optimizer = inf.get_optimizer(given_model)
    layers = inf.get_layers(given_model)
    metrics = ['accuracy']

    model = keras.Sequential()
    model.add(keras.Input(input_shape))
    size = len(given_model["Layers"])
    for i in range(size):
        model.add(inf.get_layer_type(layers[i]))
    model.compile(loss=lossfunc, metrics=metrics, optimizer=optimizer)
    model.summary()
    return model

if __name__ == '__main__':
    # params = {
    #     'batches': [2, 5, 10, 20,30],
    #     'epochs': [50, 100, 150],
    #     'dropout': [0, 0.5, 5],
    #     'validation_split': [0.1, 0.2],
    #     'activation': ["relu", "elu"]
    # }
    data_path = r"./documents/result.json"
    with open(data_path, 'r', encoding="utf8") as f:
        nets_dict = json.load(f)

    dict = load_data("mnist")
    temp_model = nets_dict[0]
    model = create_model(temp_model, dict["input_shape"])

    history = model.fit(dict["x_train"], dict["y_train"], batch_size=64, epochs=2, validation_split=0.2)
    test_scores = model.evaluate(dict["x_test"], dict["y_test"], verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    #param_grid = dict(epochs=params["epochs"], batch_size=params["batches"])
    #grid = GridSearchCV(estimator=model, param_grid=param_grid)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=1, scoring=scorers, refit="precision_score")
    # grid_result = grid.fit(d["x_train"], d["y_train"])
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # parameters = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, parameters):
    #     print("%f (%f) with: %r" % (mean, stdev, parameters))