from typing import List
import numpy as np
from layer import Layer

class Network:
    layers: List[Layer] # hidden layers and output layer
    D: int # dimension of features
    C: int # dimension of targets
    is_regression: bool
    epochs: int
    learnrate: float
    mb: int
    verbose: bool


    def __init__(self, D, C, problem_mode, activation, learnrate, init_range, num_hidden_units, num_hidden_layers, epochs, mb_size, verbose):
        self.verbose = verbose
        self.epochs = epochs
        self.learnrate = learnrate
        self.layers = []
        self.D = D
        self.C = C
        self.mb = mb_size
        match problem_mode:
            case 'C':
                self.is_regression = False
            case 'R':
                self.is_regression = True
            case _:
                raise RuntimeError("Unknown problem mode")
        last_dim = D
        for i in range(num_hidden_layers):
            self.layers.append(Layer(last_dim, num_hidden_units, activation, init_range))
            last_dim = num_hidden_units
        # create hidden-to-output layer
        if not self.is_regression and C != 2:
            final_act = "softmax"
        elif not self.is_regression:
            final_act = "sig"
        else:
            final_act = "ident"
        self.layers.append(Layer(last_dim, C, final_act, init_range))


    def fit(self, x_train, y_train, x_dev, y_dev):
        ''' Fit model to training data and report error
        '''
        x_train = x_train.reshape(-1, self.D)
        y_train = self.onehotencode(y_train, self.C)
        x_dev = x_dev.reshape(-1, self.D)
        y_dev = self.onehotencode(y_dev, self.C)
        for e in range(1, self.epochs + 1):
            if self.mb == 0:
                self.epoch(x_train, y_train, x_dev, y_dev, e)
            else:
                x_train, y_train = self.shuffle(x_train, y_train)
                self.epoch_mb(x_train, y_train, x_dev, y_dev, e)


    def onehotencode(self, y, num_classes):
        if self.is_regression:
            return y.reshape(-1, self.C)

        l = np.arange(y.shape[0])
        onehot = np.zeros((y.shape[0], num_classes))
        onehot[l, y.astype(int)] = 1
        return onehot.reshape(-1, self.C)


    def shuffle(self, x_train, y_train):
        ''' Randomly shuffle feature/target set
        '''
        # TODO: smarter way to do this?
        temp = np.hstack((x_train, y_train.reshape(-1, self.C)))
        np.random.shuffle(temp) 
        return temp.T[:self.D].T, temp.T[-self.C:].T


    def epoch(self, x_train, y_train, x_dev, y_dev, e):
        ''' Pass data through the model.
            
            x: model input
            y: model output
            x_dev: dev data input
            y_dev: dev data output
            returns: loss (MSE if regression, accuracy if classification)
        '''
        perf_train = self.update(x_train, y_train)
        perf_dev = self.score(x_dev, y_dev)
        self.print_epoch(e, perf_train, perf_dev)


    def epoch_mb(self, x_train, y_train, x_dev, y_dev, e):
        num_batches = x_train.shape[0] // self.mb
        for i in range(1, num_batches + 1):
            mb_x = x_train[(i-1)*self.mb:i*self.mb]
            mb_y = y_train[(i-1)*self.mb:i*self.mb]
            perf_train = self.update(mb_x, mb_y)
            if self.verbose:
                perf_dev = self.score(x_dev, y_dev)
                self.print_update(i, perf_train, perf_dev)
        # calc performence of this epoch for dev
        perf_train = self.score(x_train, y_train)
        perf_dev = self.score(x_dev, y_dev)
        self.print_epoch(e, perf_train, perf_dev)


    def update(self, x, y):
        ''' Run data through the model and update weights according to loss.
            Should only be used with training data (not dev data).

            x: input
            y: expected model output
            returns: calculated MSE/accuracy
        '''
        # run forward through all hidden layers (ignored if linear model)
        y_pred = self.forward(x)
        loss = self.compute_loss(y_pred, y)
        grads = self.backprop(x, loss)
        self.update_weights(grads)
        return self.compute_error(y_pred, y)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
        return x


    def transform(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.transform(x)
        return x


    def score(self, x, y):
        y_pred = self.transform(x)
        return self.compute_error(y_pred, y)


    def backprop(self, x, loss):
        ''' Calculate loss gradients for each layer. Uses cached layer outputs
            from most recent `update()` to compute at each layer.

            x: model input
            loss: output loss
            returns: list of gradients at layers 0 ... k
        '''
        scale = 1 / self.mb if self.mb > 0 else 1 / x.shape[0]
        grad = []
        for i, layer in enumerate(reversed(self.layers[1:])):
            prev = self.layers[-i-2]
            dL_dw = scale * prev.a.T @ loss
            dL_db = scale * (np.ones((prev.a.shape[0])).T @ loss).T
            grad.append((dL_dw, dL_db))
            # calc loss for prev layer -- (out,mb) * (units,mb)
            loss = (prev.activation.derivative(prev.z).T * (layer.w @ loss.T)).T
        # input-to-hidden
        dL_dw = scale * x.T @ loss
        dL_db = scale * (np.ones((x.shape[0])).T @ loss).T
        grad.append((dL_dw, dL_db))
        return list(reversed(grad))


    def update_weights(self, grads):
        for i, layer in enumerate(self.layers):
            w, b = grads[i]
            layer.w += w * self.learnrate
            layer.b += b * self.learnrate


    def compute_loss(self, y_pred, y):
        if self.is_regression:
            loss = y - y_pred
        else:
            epsilon = np.finfo(float).eps # avoid log(0)
            loss = -y * np.log(epsilon + np.abs(y_pred))
        return loss


    def compute_error(self, y_pred, y):
        ''' Compute error (MSE if regression, accuracy if classification)
        '''
        N = len(y_pred)
        if self.is_regression:
            return np.mean(np.abs(y_pred - y))
        else:
            return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))


    def print_epoch(self, e, perf_train, perf_dev):
        print(f"Epoch {str(e).rjust(3, '0')}: train={'%.3f' % perf_train} dev={'%.3f' % perf_dev}")


    def print_update(self, i, perf_train, perf_dev):
        print(f"Update {str(i).rjust(6, '0')}: train={'%.3f' % perf_train} dev={'%.3f' % perf_dev}")


    def __str__(self):
        return f"""
        Network: R^{self.D} -> R^{self.C}
        epochs={self.epochs}
        layers:\n{chr(0xa).join([str(l) for l in self.layers])}"""

