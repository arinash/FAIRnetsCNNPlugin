import json
import numpy as np
from tensorflow import keras
from keras.utils import np_utils
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
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
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

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
    metrics = ["accuracy"]

    model = keras.Sequential()
    model.add(keras.Input(input_shape))
    size = len(given_model["Layers"])
    for i in range(size):
        model.add(inf.get_layer_type(layers[i]))
    model.compile(loss=lossfunc, metrics=metrics, optimizer=optimizer)
    model.summary()
    return model

if __name__ == '__main__':

    #Hyperparameters' dictionary
    params = {
        "learning_rate": [0.01, 0.1, 0.5, 1],
        "batches": [10, 30, 60, 120],
        "epochs": [50, 100, 150],
        "dropout": [0, 0.2, 0.5],
        "validation_split": [0.1, 0.2, 0.3],
        "activation": ["relu", "elu"]
    }
    data_path = r"./documents/result.json"
    with open(data_path, "r", encoding="utf8") as f:
        nets_dict = json.load(f)

    loaded_dict = load_data("mnist")
    temp_model = nets_dict[4]
    #created_model = create_model(temp_model, loaded_dict["input_shape"])

    # print(loaded_dict["x_train"], loaded_dict["y_train"])
    # history = model.fit(loaded_dict["x_train"], loaded_dict["y_train"], batch_size=64, epochs=2, validation_split=0.2)
    # test_scores = model.evaluate(loaded_dict["x_test"], loaded_dict["y_test"], verbose=2)
    # print("Test loss:", test_scores[0])
    # print("Test accuracy:", test_scores[1])

    #Randomized hyperparameters' search
    res_model = KerasClassifier(build_fn=lambda: create_model(given_model=temp_model, input_shape=loaded_dict["input_shape"]), verbose=1)
    param_grid = dict(epochs=params["epochs"], batch_size=params["batches"], validation_split=params["validation_split"])
    grid = RandomizedSearchCV(estimator=res_model, param_distributions=param_grid, scoring="accuracy", verbose=1, n_iter=2)
    print("Tuning is done")
    grid_result = grid.fit(loaded_dict["x_train"], loaded_dict["y_train"])
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    parameters = grid_result.cv_results_["params"]
    for mean, stdev, param in zip(means, stds, parameters):
        print("%f (%f) with: %r" % (mean, stdev, parameters))