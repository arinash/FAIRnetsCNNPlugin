import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import np_utils
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import JSONInfoRetriever as inf
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

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
    input_shape = (sample_shape[0], sample_shape[1], 3) if dataset_name == "cifar10" else (sample_shape[0], sample_shape[1], 1)
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
def create_model(learning_rate, dropout, dense_activ, conv_filter, conv_kernel):
    lossfunc = inf.get_loss_func()
    optimizer = inf.get_optimizer(learning_rate)
    layers = inf.get_layers()
    metrics = ["accuracy"]

    model = keras.Sequential()
    model.add(keras.Input(inf.get_input_shape()))
    size = inf.get_layers_numer()
    for i in range(size):
        model.add(inf.get_layer_type(layers[i], dropout, dense_activ, conv_filter, conv_kernel))
    model.compile(loss=lossfunc, metrics=metrics, optimizer=optimizer)
    model.summary()
    return model

#Plot dependencies
def plot_accuracy(grid_result, fst_param, snd_param, title):
    history = grid_result.best_estimator_.model.history
    plt.plot(history.history[fst_param])
    plt.title(title)
    plt.ylabel(fst_param)
    plt.xlabel(snd_param)
    plt.legend(['train'], loc='upper left')
    plt.show()

if __name__ == '__main__':

    #Hyperparameters' dictionary
    params = {
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "batches": [10, 30, 60, 120],
        "epochs": [50, 100, 150],
        "dropout": [0, 0.2, 0.5],
        "validation_split": [3, 5],
        "activation": ["relu", "elu"],
        "conv_filter": [64, 128],
        "conv_kernel": [(2), (4)]
    }
    data_path = r"./documents/result.json"
    with open(data_path, "r", encoding="utf8") as f:
        nets_dict = json.load(f)

    loaded_dict = load_data("mnist")
    inf.model = nets_dict[5]
    input_shape = loaded_dict["input_shape"]
    inf.model_input_shape = input_shape

    #Randomized hyperparameters' search
    res_model = KerasClassifier(build_fn=create_model, verbose=1)
    param_grid = dict(epochs=params["epochs"], batch_size=params["batches"], learning_rate=params["learning_rate"],
                      dropout=params["dropout"], dense_activ=params["activation"], conv_filter=params["conv_filter"],
                      conv_kernel=params["conv_kernel"])
    grid = RandomizedSearchCV(estimator=res_model, param_distributions=param_grid, scoring=["accuracy", "f1_macro"],
                              verbose=1, n_iter=3, refit="accuracy", cv=StratifiedKFold(n_splits=3))
    grid_result = grid.fit(loaded_dict["x_train"], np.argmax(loaded_dict["y_train"], axis=1))

    #Print best results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    for i in grid_result.cv_results_:
        print(i, grid_result.cv_results_[i])
    means_acc = grid_result.cv_results_["mean_test_accuracy"][grid_result.best_index_]
    stds_acc = grid_result.cv_results_["std_test_accuracy"][grid_result.best_index_]
    means_f1 = grid_result.cv_results_["mean_test_f1_macro"][grid_result.best_index_]
    stds_f1 = grid_result.cv_results_["std_test_f1_macro"][grid_result.best_index_]
    try:
        means_loss = np.average(grid_result.best_estimator_.model.history.history['loss'])
        print("mean loss: %f" % (means_loss))
    except Exception as exc:
        print(exc)
    print("mean accuracy: %f" % (means_acc))
    print("std accuracy: %f" % (stds_acc))
    print("mean f1: %f" % (means_f1))
    print("std f1: %f" % (stds_f1))

    results = pd.DataFrame(grid_result.cv_results_)
    results.sort_values(by='rank_test_accuracy', inplace=True)
    params_1st_best = results.loc[0, 'params']
    clf_1st_best = grid.best_estimator_.set_params(**params_1st_best)
    params_2nd_best = results.loc[1, 'params']
    clf_2nd_best = grid.best_estimator_.set_params(**params_2nd_best)
    params_3rd_best = results.loc[2, 'params']
    clf_3rd_best = grid.best_estimator_.set_params(**params_3rd_best)
    with open(r"./documents/params.json", "w") as outfile:
        outfile.write(results.to_json(orient="index"))

    #Plot accuracy's changing with epoochs
    plot_accuracy(grid_result, "accuracy", "epochs", "model accuracy")