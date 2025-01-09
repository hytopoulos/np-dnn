# Deep Neural Network with NumPy

This project is an implementation of a deep neural network using only NumPy. It supports both classification and regression problems, minibatch training, and customizable network depth and activation functions.

## Setup

1. **Clone the repository**:
    ```
    git clone <repository-url>
    ```
2. **Create Virtual Environment**:
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
