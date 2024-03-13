# Deep Neural Network with NumPy

This project is an implementation of a deep neural network using only NumPy. Designed to understand the fundamentals of neural networks, this tool can be utilized for both classification and regression problems with support for minibatch training and customizable network depth and activation functions.

## Features

- **Single Hidden Layer Neural Network**: A basic neural network with one hidden layer for straightforward tasks.
- **Arbitrarily Deep Neural Network**: Supports creating neural networks with any number of hidden layers to tackle more complex problems.
- **Classification and Regression Modes**: Can be configured to solve either classification or regression problems.
- **Minibatch Training**: Improves training efficiency by using minibatches, with customizable batch sizes.
- **Command-Line Interface**: Easy-to-use command-line interface for training and evaluating the neural network.
- **Customizable Parameters**: Allows setting various hyperparameters like learning rate, number of epochs, hidden units, activation functions, and initialization ranges.

## Setup

1. **Clone the repository**:
    ```
    git clone <repository-url>
    ```
2. **Create a Python Virtual Environment** (Optional but recommended):
    ```
    python3 -m venv ~/.venv_dnn
    source ~/.venv_dnn/bin/activate
    ```
3. **Install Dependencies**:
    ```
    pip3 install numpy
    ```

## Usage

Navigate to the project directory and run `prog1.py` with the necessary command-line arguments:

```
python prog1.py [-v] -train_feat TRAIN_FEAT_FN -train_target TRAIN_TARGET_FN -dev_feat DEV_FEAT_FN -dev_target DEV_TARGET_FN -epochs EPOCHS -learnrate LEARNRATE -nunits NUM_HIDDEN_UNITS -type PROBLEM_MODE -hidden_act HIDDEN_UNIT_ACTIVATION -init_range INIT_RANGE [-num_classes C] [-mb MINIBATCH_SIZE] [-nlayers NUM_HIDDEN_LAYERS]
```
