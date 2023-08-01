
# imports
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.initializers as initializers
from tensorflow.keras.regularizers import L1L2
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle5
from phynteny_utils import format_data
import numpy as np
import random
import pkg_resources
import absl.logging
import tensorflow as tf

absl.logging.set_verbosity(absl.logging.ERROR)


def get_dict(dict_path):
    """
    Helper function to import dictionaries
    """

    with open(dict_path, "rb") as handle:
        dictionary = pickle5.load(handle)
    handle.close()

    return dictionary


def get_optimizer(optimizer_function, learning_rate):
    """
    Get the optimization function to train the LSTM based on the provided inputs

    :param optimizer_function: optimization function. One of ['adam', 'rmsprop', 'adagrad', 'sgd']
    :param learning_rate: initial learning rate for the opimization function
    :return: opimization function
    """

    # optimization function
    if optimizer_function == "adam":
        optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1)
    elif optimizer_function == "rmsprop":
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_function == "adagrad":
        optimizer = optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_function == "sgd":
        optimizer_function = optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(
            "Invalid optimizer function. Must be One of ['adam', 'rmsprop', 'adagrad', 'sgd']"
        )

    return optimizer_function


def get_initializer(initializer_function):
    """
    Get the kernel initializer to train the LSTM

    :param initializer_function: string describing which intializer to use
    :return: kernel_initializer
    """

    if initializer_function == "zeros":
        kernel_initializer = initializers.Zeros()
    elif initializer_function == "random_normal":
        kernel_initializer = initializers.RandomNormal(stddev=0.01, seed=42)
    elif initializer_function == "random_uniform":
        kernel_initializer = initializers.RandomUniform(
            minval=-0.05, maxval=0.05, seed=42
        )
    elif initializer_function == "truncated_normal":
        kernel_initializer = initializers.TruncatedNormal(
            mean=0.0, stddev=0.05, seed=42
        )

    else:
        raise ValueError(
            "Invalid optimizer function. Must be One of ['zeros', 'random_normal', 'random_uniform', 'truncated_normal']"
        )

    return kernel_initializer


class Model:
    def __init__(
        self,
        phrog_path=pkg_resources.resource_filename(
            "phynteny_utils", "phrog_annotation_info/phrog_integer.pkl"
        ),
        max_length=120,
        layers=1,
        neurons=100,
        batch_size=32,
        dropout=0.1,
        activation="tanh",
        optimizer_function="adam",
        learning_rate=0.0001,
        patience=5,
        min_delta=0.0001,
        l1_regularizer=0,
        l2_regularizer=0,
        kernel_initializer="zeros",
    ):
        """
        :param phrog_categories_path: location of the dictionary describing the phrog_categories :param
        :param max_length: maximum length of prophage to consider
        :param layers: number of hidden layers to use in the model
        :param neurons: number of memory cells in hidden layers
        :param batch_size: batch size for training
        :param kernel_regularizer: kernel regulzarizer
        :param dropout: dropout rate to implement
        :param optimizer_function: which optimization function to use
        :param activation: which activation function to use for the hidden layers
        :param learning_rate: learning rate for training
        :param patience: number of epochs with no improvement after which training will be stopped
        :param min_delta: minimum change in validation loss considered an improvement
        """

        # set general information for the model
        self.phrog_categories = get_dict(phrog_path)
        self.num_functions = len(
            list(set(self.phrog_categories.values()))
        )  # dimension describing the number of functions
        self.max_length = max_length

        # set the hyperparameters for the model
        self.layers = layers
        self.neurons = neurons
        self.batch_size = batch_size
        self.kernel_regularizer = L1L2(l1_regularizer, l2_regularizer)
        self.kernel_intializer = kernel_initializer
        self.dropout = dropout
        self.activation = activation
        self.optimizer_function = optimizer_function
        self.learning_rate = learning_rate

        # set early stopping conditions
        self.patience = patience
        self.min_delta = min_delta

        # placeholder variables
        self.X = []
        self.y = []

    def fit_data(self, data_path):
        """
        Fit data to model object
        """

        # get data from specified path
        data = get_dict(data_path)

        # process the data
        self.X, self.y = format_data.generate_dataset(
            data, self.num_functions, self.max_length
        )

    def parse_masked_data(self, X_path, y_path):
        """
        Parse a pre-masked dataset
        """
        # get the X data
        X = get_dict(X_path)
        self.X = np.array(list(X.values()))

        # get the y data
        y = get_dict(y_path)
        self.y = np.array(list(y.values()))

    def get_callbacks(self, model_out):
        """
        Callbacks to use for training the LSTM

        :param model_out: the prefix of the model output
        """

        es = EarlyStopping(
            monitor="loss",
            mode="min",
            verbose=1,
            patience=self.patience,
            min_delta=self.min_delta,
        )

        mc = ModelCheckpoint(
            model_out + "best_val_accuracy.h5",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
            save_freq="epoch",
        )  # model with the best validation loss therefore minimise
        mc2 = ModelCheckpoint(
            model_out + "best_val_loss.h5",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
            save_freq="epoch",
        )

        callbacks = [es, mc, mc2]

        return callbacks

    def generate_LSTM(self):
        """
        Function for generating a LSTM model

        :return: model ready to be trained
        """

        print("building model")
        # define the model
        model = Sequential()

        # get the kernel initializer
        kernel_initializer = get_initializer(self.kernel_intializer)

        print("Number of layers: " + str(self.layers))

        # loop to add layers to the model
        for layer in range(self.layers):
            # add the input layer
            if layer < self.layers - 1:
                print("adding a layer")
                model.add(
                    Bidirectional(
                        LSTM(
                            self.neurons,
                            return_sequences=True,
                            dropout=self.dropout,
                            kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=kernel_initializer,
                            activation=self.activation,
                        ),
                        input_shape=(self.max_length, self.num_functions),
                    )
                )

            # add the final hidden layer
            else:
                print("adding the last layer")

                model.add(
                    Bidirectional(
                        LSTM(
                            self.neurons,
                            dropout=self.dropout,
                            kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=kernel_initializer,
                            activation=self.activation,
                        ),
                        input_shape=(self.max_length, self.num_functions),
                    )
                )

        # output layer
        model.add(Dense(self.num_functions, activation="softmax"))

        # get the optimization function
        optimizer = get_optimizer(self.optimizer_function, self.learning_rate)

        model.compile(
            loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer
        )
        print(model.summary(), flush=True)

        return model

    def train_model(
        self,
        X_1,
        y_1,
        X_val,
        y_val,
        model_out="model",
        history_out="history",
        epochs=140,
        save=True,
    ):
        """
        Function to train the LSTM model and save the trained model

        :param X_1: X training data for the model
        :param y_1: y training data for the model
        :param X_val: X validation data for the model
        :param y_val: y validation data for the model
        :param model_out: string - prefix of model output
        :param history_out: string - prefix of history dictionary output
        :param epochs: number of epochs to train the model for
        :param save: whether to save the model - default = True
        """

        # model with the best validation set accuracy therefore maximise
        model = self.generate_LSTM()

        history = model.fit(
            X_1,
            y_1,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=self.get_callbacks(model_out),
            validation_data=(X_val, y_val),
            verbose=1,
        )

        # save the model
        if save:
            model.save(model_out + "final_model")

        # save the history dictionary as a pickle
        with open(history_out + ".pkl", "wb") as handle:
            pickle5.dump(history.history, handle, protocol=pickle5.HIGHEST_PROTOCOL)

    def train_crossValidation(
        self,
        model_out="model",
        history_out="history",
        n_splits=10,
        epochs=140,
        save=True,
        include=0,
    ):
        """
        Perform stratified cross-validation
        Random states are used such that the splits can be reproduced between replicates

        :param model_out: prefix of output models
        :param history_out: prefix of history file
        :param n_splits: number of k-folds to include. 1 is added to this to also generate a test set of equal size
        :param epochs: number of epochs to train for
        :param include: which kfolds to use for training. Default is to loop through all of the kfolds.
        """

        # get the masked category in each instance
        print('categories', flush = True)
        masked_cat = [np.where(self.y[i] == 1)[0][0] for i in range(len(self.y))]

        # split into kfolds which are stratified
        print('skf')
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # count the number of folds
        counter = 0

        # create a list of list which stores the required indicies
        train_index_kfold = []
        val_index_kfold = []
        for train_index, val_index in skf.split(np.zeros(len(masked_cat)), masked_cat):
         
            train_index_kfold.append(train_index)
            val_index_kfold.append(val_index)


        print(val_index_kfold)
        # create list of kfolds to consider
        kfolds = []
        if include == 0:
            kfolds = list(range(n_splits))
        else:
            kfolds = [include-1]
            counter = include-1

        for k in kfolds:
            
            # generate stratified test and train sets
            X_train = self.X[train_index_kfold[k], :, :]
            y_train = self.y[train_index_kfold[k], :]

            # generate validation data for the training
            X_val = self.X[val_index_kfold[k], :, :]
            y_val = self.y[val_index_kfold[k], :]

            # reshape
            #print('reshaping') 
            #print(y_train.shape)
            #y_train = y_train.reshape((len(y_train), self.num_functions))
            #y_val = y_val.reshape((len(y_val), self.num_functions))

            # use the compile function here
            self.train_model(
                X_train,
                y_train,
                X_val,
                y_val,
                model_out=model_out + ".rep_" + str(counter) + ".",
                history_out=history_out + ".rep_" + str(counter) + ".",
                epochs=epochs,
                save=save,
            )

            # update counter
            counter += 1

def mean_metric(history_out, n_splits):
    """
    Return the mean score for a model based on output history files

    :param history_out: prefix used for the history output
    :param n_splits: number of splits used for the cross validation
    """

    # intialise lists to store data
    best_loss = []
    best_val_loss = []
    best_acc = []
    best_val_acc = []

    # loop through each of the k-fold splits
    for i in range(n_splits):
        # read in the history from the split
        hist = get_dict(history_out + "_" + str(i) + ".pkl")

        # compare the history
        best_loss.append(np.min(hist.get("loss")))
        best_val_loss.append(np.min(hist.get("val_loss")))
        best_acc.append(np.min(hist.get("accuracy")))
        best_val_acc.append(np.min(hist.get("val_accuracy")))

    return {
        "loss": np.mean(best_loss),
        "val_loss": np.mean(best_val_loss),
        "accuracy": np.mean(best_acc),
        "val_accuracy": np.mean(best_val_acc),
    }


def check_parameters(hyperparameters, num_trials):
    """
    Check the validity of the hyperparameter dictionary

    :param hyperparameters: dictionary of the hyperparameters to test
    :param num_trials: number of trials to use in the random search
    """

    poss_parameters = [
        "max_length",
        "layers",
        "neurons",
        "batch_size",
        "dropout",
        "activation",
        "optimizer_function",
        "learning_rate",
        "patience",
        "min_delta",
        "l1_regularizer",
        "l2_regularizer",
    ]

    # extract the hyperparameters used here
    keys = list(hyperparameters.keys())

    # get the total number of parameters in the dictionary
    parameter_count = 1
    for k in keys:
        # Check that the parameter is allowed
        if k not in poss_parameters:
            raise ValueError(
                " illegal parameter value. Parameters must be one of 'max_length', "
                "'layers', 'neurons', 'batch_size', 'dropout', 'activation', 'optimizer_function', "
                "'learning_rate', 'patience', 'min_delta', 'l1_regularizer', 'l2_regularizer'"
            )

        # update the number of hyperparameters
        parameter_count *= len(hyperparameters.get(k))

    # calculate the number of allowed trials
    if num_trials > parameter_count:
        raise ValueError(
            "num_trials must be equal or less than the possible number of hyperparamters! Try using a lower number of "
            "trials"
        )


def random_search(
    data_path,
    hyperparameters,
    num_trials,
    model_out="model",
    history_out="history",
    k_folds=10,
    epochs=140,
    save=False,
):
    """
    Method for random parameter search

    :param data_path: path to the pickled training data
    :param hyperparameters: dictionary of hyperparameters to test
    :param model_out: prefix of the model files
    :param history_out: prefix of history file
    :param k_folds: number of folds to use for cross validation
    :param num_trials: number of random parameter combinations to try
    :param save: boolean whether or not to save the model
    """

    check_parameters(hyperparameters, num_trials)

    # initialise helper variables
    tried_params = []
    best_params = None
    best_loss = float("inf")

    for i in range(num_trials):
        # Randomly select some hyperparameters to use for this trial, ensuring that each combination is unique
        params = None
        while params is None or params in tried_params:
            params = {
                key: random.choice(values) for key, values in hyperparameters.items()
            }
        tried_params.append(params)

        # create a model object using the hyperparameters
        model = Model(**params)
        model.fit_data(data_path)  # data is the path to the training data

        # perform stratified cross validation
        model.train_crossValidation(
            model_out=model_out,
            history_out=history_out + "_" + str(i),
            n_splits=k_folds,
            epochs=epochs,
            save=save,
        )
        mean_metrics = mean_metric(history_out + "_" + str(i), k_folds)

        # get the average metrics across the k-folds
        val_loss = mean_metrics.get("val_loss")
        val_acc = mean_metrics.get("val_accuracy")
        # print(f'Validation loss: {val_loss:.4f}')
        # print(f'Validation accuracy: {val_acc:.4f}')

        # need a way to return the loss
        if val_loss < best_loss:
            best_params = params
            best_loss = val_loss

    print(f"Best params: {best_params}")
    print(f"Best validation loss: {best_loss:.4f}")
