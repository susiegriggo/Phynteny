"""
Train LSTM with gene order.

This script trains a single instance of an LSTM using 10-fold stratified crossvalidation
"""

# imports
from phynteny_utils import train_model
import click
import pkg_resources


@click.command()
# @click.option("--data", "-d", help="File path to training data")
@click.option("--x_path", "-x", help="File path to X training data")
@click.option("--y_path", "-y", help="File path to y training data")
@click.option(
    "--max_length",
    "-ml",
    default=120,
    type=int,
    help="Maximum length of a phage genome",
)
@click.option(
    "--layers",
    "-l",
    default=1,
    type=int,
    help="Number of hidden layers to use in model. Default = 1",
)
@click.option(
    "--memory_cells",
    "-m",
    default=100,
    type=int,
    help="Number of memory cells to train the model. Default = 100",
)
@click.option(
    "--batch_size",
    "-b",
    default=32,
    type=int,
    help="Batch size to use for training the model",
)
@click.option(
    "--dropout",
    "-dr",
    default=0.1,
    type=float,
    help="Dropout applied to prevent overfitting",
)
@click.option(
    "--activation",
    "-a",
    default="tanh",
    type=click.Choice(["tanh", "sigmoid", "relu"]),
    help="activtion function applied to the input and hidden layers. Must be one of ['tanh', 'sigmoid', 'relu']",
)
@click.option(
    "--optimizer",
    "-opt",
    default="adam",
    type=click.Choice(["adam", "rmsprop", "adagrad", "sgd"]),
    help="Optimization function. Must be one of ['adam', 'rmsprop', 'adagrad', 'sgd']",
)
@click.option(
    "--learning_rate",
    "-lr",
    default=0.0001,
    type=float,
    help="Learning rate applied to opimization function",
)
@click.option(
    "--patience",
    "-p",
    default=5,
    type=int,
    help="number of epochs with no improvement after which training will be stopped",
)
@click.option(
    "--min_delta",
    "-md",
    default=0.0001,
    type=float,
    help="minimum change in validation loss considered an improvement",
)
@click.option(
    "--model_out", "-o", default="model", type=str, help="prefix of the model output"
)
@click.option(
    "--history_out",
    "-ho",
    default="history",
    type=str,
    help="prefix of history dictionary output",
)
@click.option(
    "--k_folds",
    "-k",
    default=10,
    type=int,
    help="Number of folds to use for k-fold cross-validation",
)
@click.option(
    "--epochs",
    "-e",
    default=140,
    type=int,
    help="Maximum number of epochs to train for",
)
@click.option(
    "--l1_regularize", "-l1", default=0, type=float, help="L1 regularization. Default=0"
)
@click.option(
    "--l2_regularize", "-l1", default=0, type=float, help="L2 regularization. Default=0"
)
@click.option(
    "--kernel_initializer",
    "-ki",
    default="random_normal",
    type=str,
#    type=click.Choice(["zeros", "random_normal", "random_uniform", "truncated_normal"]),
    help="kernel initializer",
)

def main(
    x_path,
    y_path,
    max_length,
    layers,
    memory_cells,
    batch_size,
    dropout,
    activation,
    optimizer,
    learning_rate,
    patience,
    min_delta,
    model_out,
    history_out,
    k_folds,
    epochs,
    l1_regularize,
    l2_regularize,
    kernel_initializer,
):
    print("STARTING")
    # create a model object
    model = train_model.Model(
        phrog_path=pkg_resources.resource_filename(
            "phynteny_utils", "phrog_annotation_info/phrog_integer.pkl"
        ),
        max_length=max_length,
        layers=layers,
        neurons=memory_cells,
        batch_size=batch_size,
        dropout=dropout,
        activation=activation,
        optimizer_function=optimizer,
        learning_rate=learning_rate,
        patience=patience,
        min_delta=min_delta,
        l1_regularizer=l1_regularize,
        l2_regularizer=l2_regularize,
        kernel_initializer=kernel_initializer,
    )

    # fit data to the model
    # model.fit_data(data)

    print("PARSING DATA")
    # parse the pre-masked data
    model.parse_masked_data(x_path, y_path)

    # perform stratified k-fold validation
    print("Performing cross validation... ")
    model.train_crossValidation(
        model_out=model_out, history_out=history_out, n_splits=k_folds, epochs=epochs
    )


if __name__ == "__main__":
    main()
