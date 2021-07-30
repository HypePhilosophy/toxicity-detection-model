# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Dropout, \
    GlobalMaxPooling1D, LSTM, Bidirectional
from tensorflow.keras.models import Model

from data_processing import DataPreprocess
from source.configuration import *


# -------------------------------------------------------------------------
#                   Build and Train the RNN Model Architecture
# -------------------------------------------------------------------------


def build_rnn_model(data, target_classes, embedding_layer):
    """
    Build and Train the RNN architecture (Bidirectional LSTM)
    :param data: the preprocessed padded data
    :param target_classes: Assigned target labels for the comments
    :param embedding_layer: Embedding layer comprising preprocessed comments
    :return: the trained model
    """
    # Create an LSTM Network with a single LSTM

    # Input includes the sequence length of the embedding layer. This is based on the word embeddings.
    # Current vector count is set at 100
    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))

    # Create the layer with Keras tensor
    x = embedding_layer(input_)

    # Create Bidirectional LSTM

    """
    :param units: a GRU unit which can be configured to any number. Trial and error based
    :param return_sequences: boolean on whether or not to return last output value
    :param recurrent_dropout: 0 - 1 float for fraction of units to drop during linear transformation. 
        Read more: https://arxiv.org/pdf/1512.05287.pdf
    """
    x = Bidirectional(LSTM(units=64,
                           return_sequences=True,
                           recurrent_dropout=0.2))(x)

    # Define Pooling Operation for down-sampling an input representation (in this case, its a hidden layer)
    # Notes: Down-sampling is the process of reducing the sampling rate by an integer factor.

    # GlobalMaxPooling1D is a pooling method for temporal data
        # Ex: If input is 0,1,2,2,5,2, output will be 5. It returns the max vector of the input over the steps dimension
    x = GlobalMaxPooling1D()(x)

    # Create Dense Layer
        # A Dense Layer is one of the two choices of layers in RNN models. The other one being a recurrent layer
        # "Fully-connected Layer" - aka Dense Layer
        # It implements the operation output = X * W + b, where x is the input, W and b are weights and bias of the layer

    # Activation Function(s) read more:
        # https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/

    # ReLU vs Sigmoid:
        # https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks

    """
    Advantages:
        Sigmoid: not blowing up activation
        Relu : not vanishing gradient
        Relu : More computationally efficient to compute than Sigmoid like functions since Relu just needs to pick max(0,x) 
        and not perform expensive exponential operations as in Sigmoids
        Relu : In practice, networks with Relu tend to show better convergence performance than sigmoid. (Krizhevsky et al.)
    
    Disadvantages: 
    Sigmoid: tend to vanish gradient (cause there is a mechanism to reduce the gradient as "a" increase, 
        where "a" is the input of a sigmoid function. 
        Gradient of Sigmoid: 
            S′(a)=S(a)(1−S(a)). 
            When "a" grows to infinite large , 
            S′(a)=S(a)(1−S(a))=1×(1−1)=0).

    Relu : tend to blow up activation (there is no mechanism to constrain the output of the neuron, as "a" itself is the output)

    Relu : Dying Relu problem - if too many activations get below zero then most of the units(neurons)
           in network with Relu will simply output zero, in other words, 
           die and thereby prohibiting learning.(This can be handled, to some extent, by using Leaky-Relu instead.)
    """

    """
    :param units: Positive integer, dimensionality of the output space.
        How many output nodes of dense layer should be returned
        
    :param activation: An activation function in a neural network defines how the weighted sum of the input is 
        transformed into an output from a node or nodes in a layer of the network. 
        The modern default activation function for hidden layers is the ReLU function
        ReLU: https://towardsdatascience.com/under-the-hood-of-neural-networks-part-1-fully-connected-5223b7f78528
        
        Calculation: max(0.0, x)
        This means that if the input value (x) is negative, then a value 0.0 is returned, otherwise, the value is returned.
    """
    x = Dense(units=64, activation='relu')(x)

    # Set Dropout Layer
        # Randomly sets inputs to 0
    """
    :param rate: Float between 0 and 1. Fraction of the input units to drop.
        Randomly sets inputs to 0 with a frequency designated by the rate parameter.
        Helps prevent overfitting
    """
    x = Dropout(rate=0.2)(x)

    #  Sigmoid Classifier

    """
    :param units: Positive integer, dimensionality of the output space.
        How many output nodes of dense layer should be returned
        In this case, it returns the DETECTION_CLASSES Length, which is 7

    :param activation: An activation function in a neural network defines how the weighted sum of the input is 
        Sigmoid:
            The function takes any real value as input and outputs values in the range 0 to 1. 
            The larger the input (more positive), the closer the output value will be to 1.0, 
            whereas the smaller the input (more negative), the closer the output will be to 0.0.
            Calculation: 1.0 / (1.0 + e^-x)
            Where e is a mathematical constant, which is the base of the natural logarithm.
    """

    output = Dense(len(DETECTION_CLASSES), activation="sigmoid")(x)

    # Define Model with input and output

    model = Model(input_, output)

    # Display Model
    model.summary()

    # Compile Model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Define Callbacks
    # TODO Check whether to use the restore_best_weights
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=5,
                               mode='min',
                               restore_best_weights=True)

    checkpoint = ModelCheckpoint(filepath=MODEL_LOC,  # saves the 'best' model
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min')

    # Fit Model
    history = model.fit(data,
                        target_classes,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        callbacks=[early_stop, checkpoint],
                        verbose=1)

    # Return Model Training History
    return model, history


# -------------------------------------------------------------------------
#                   Plotting the training history
# -------------------------------------------------------------------------
def plot_training_history(rnn_model, history, data, target_classes):
    """
    Generates plots for accuracy and loss
    :param rnn_model: the trained model
    :param history: the model history
    :param data: preprocessed data
    :param target_classes: target classes for every comment
    :return: None
    """
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("../plots/accuracy.jpeg")
    plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("../plots/loss.jpeg")
    plt.show()

    # Print Average ROC_AUC_Score
    p = rnn_model.predict(data)
    aucs = []
    for j in range(len(DETECTION_CLASSES)):
        auc = roc_auc_score(target_classes[:, j], p[:, j])
        aucs.append(auc)
    print(f'Average ROC_AUC Score: {np.mean(aucs)}')


@click.command()
@click.option('--data', default=TRAINING_DATA_LOC, help="Training Data (CSV) Location")
def execute(data):
    """
    Import the training data csv file and save it into a dataframe
    :param data: the training data (CSV) location
    """
    training_data = pd.read_csv(data)

    preprocessing = DataPreprocess(training_data)
    rnn_model, history = build_rnn_model(preprocessing.padded_data,
                                         preprocessing.target_classes,
                                         preprocessing.embedding_layer)
    plot_training_history(rnn_model,
                          history,
                          preprocessing.padded_data,
                          preprocessing.target_classes)


# -------------------------------------------------------------------------
#                               Main Execution
# -------------------------------------------------------------------------
if __name__ == '__main__':
    execute()