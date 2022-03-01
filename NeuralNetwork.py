import numpy as np
import math


def sigmoid(z):
    z = 1 / (1 + np.exp(-z))

    return z


def relu(z):
    return np.maximum(0, z)


def sigmoid_derivative(a):
    res = sigmoid(a) * (1 - sigmoid(a))

    return res


def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1

    return x


def eta(x):
    e = 0.0000000001

    return np.maximum(x, e)


def predict(x, y, paras, cache, L):
    cache['A0'] = x
    m = y.shape[1]

    for l in range(1, L):
        cache['Z' + str(l)] = np.dot(paras['W' + str(l)],
                                     cache['A' + str(l - 1)]) + paras['b' + str(l)]
        cache['A' + str(l)] = np.maximum(0, cache['Z' + str(l)])  # relu

    cache['Z' + str(L)] = np.dot(paras['W' + str(L)],
                                 cache['A' + str(L - 1)]) + paras['b' + str(L)]
    cache['A' + str(L)] = sigmoid(cache['Z' + str(L)])

    y_pred = cache['A' + str(L)]
    y_pred = y_pred > 0.5
    y_pred = y_pred + 0

    return accuracy(y, y_pred, m)


def accuracy(y, y_pred, m):  # a = original_y, b = predicted
    y = np.squeeze(y)
    y_pred = np.squeeze(y_pred)

    count = 0

    # print("A:", a)
    # print("B:", b)

    for i in range(m - 1):
        if y[i] == y_pred[i]:
            count = count + 1

    # print("Total: ", m)
    # print("Correct: ", count)

    return count / m * 100


def random_mini_batches(X, Y,
                        mini_batch_size=64):  # Step 0 (Not necessary but can speed up the minimizing cost: convert whole x, y into mini x and y of size 64(size)
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_mini_batches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_mini_batches):
        mini_batch_X = shuffled_X[:, k *
                                  mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k *
                                  mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[:,
                                  num_complete_mini_batches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,
                                  num_complete_mini_batches * mini_batch_size:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# Step 1: Initializing the parameters W, b (and v, s required for Adam optimization)
def initialize_parameters(layer):
    paras = {}
    v = {}
    s = {}

    L = len(layer) - 1

    for l in range(1, L + 1):
        paras["W" + str(l)] = np.random.randn(layer[l], layer[l - 1])
        paras["W" + str(l)] = paras["W" + str(l)] * np.sqrt(
            2 / layer[l - 1])  # He initialization (Not necessary but can speed up the minimizing cost

        paras["b" + str(l)] = np.zeros((layer[l], 1))

        v["dW" + str(l)] = np.zeros_like(paras["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(paras["b" + str(l)])

        s["dW" + str(l)] = np.zeros_like(paras["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(paras["b" + str(l)])

    return paras, v, s


# Step 2: Calculating predicted values from parameters
def forward_propagation(x, paras, L):
    cache = {'A' + str(0): x}

    for l in range(1, L):  # Getting values from 1 - (L-1) layers with activation function: relu
        cache['Z' + str(l)] = np.dot(paras['W' + str(l)],
                                     cache['A' + str(l - 1)]) + paras['b' + str(l)]
        cache['A' + str(l)] = relu(cache['Z' + str(l)])

    cache['Z' + str(L)] = np.dot(paras['W' + str(L)],
                                 cache['A' + str(L - 1)]) + paras['b' + str(L)]
    cache['A' + str(L)] = sigmoid(cache['Z' + str(L)])

    # Getting final predicted value with activation function: sigmoid

    return cache


def calculate_cost(y, paras, cache, L,
                   lambd=0):  # Step 3: Calculating how much does our predicted values differ from the actual
    m = y.shape[1]

    log_probs = (np.dot(y, np.log(cache['A' + str(L)]).T) +
                 np.dot((1 - y), np.log(1 - cache['A' + str(L)]).T)) / -m

    # Calculating how much should we increase the effect of given parameter 'W' on cost function
    sum = np.square(paras['W' + str(1)])
    for l in range(2, L):
        sum = sum + np.sum(np.dot(paras['W' + str(l)].T, paras['W' + str(l)]))

    # Calculating how much do we want increase the effect of given parameter 'W' on cost function i.e., L2 regularization
    reg = lambd * np.sum(sum) / (2 * m)
    cost = log_probs + reg

    cost = np.squeeze(cost)

    return cost


def backward_propagation(y, paras, cache, lambd,
                         L):  # Step 4: Calculating how much each parameter (W, b) effect on the cost
    grads = {}
    m = y.shape[1]

    grads['dA' + str(L)] = np.divide(1 - y, eta(1 - cache['A' + str(L)])
                                     ) - np.divide(y, eta(cache['A' + str(L)]))
    grads['dZ' + str(L)] = grads['dA' + str(L)] * \
        (cache['A' + str(L)] * (1 - cache['A' + str(L)]))

    grads['dW' + str(L)] = np.dot(grads['dZ' + str(L)], cache['A' + str(L - 1)].T) / m + (
        lambd * paras['W' + str(L)]) / m  # For the regularization
    grads['db' + str(L)] = np.sum(grads['dZ' + str(L)],
                                  axis=1, keepdims=True) / m

    for l in range(L - 1, 0, -1):
        grads['dA' + str(l)] = np.dot(paras["W" + str(l + 1)].T,
                                      grads['dZ' + str(l + 1)])
        grads['dZ' + str(l)] = grads['dA' + str(l)] * \
            relu_derivative(cache['Z' + str(l)])

        grads['dW' + str(l)] = np.dot(grads['dZ' + str(l)], cache['A' + str(l - 1)].T) / m + (
            lambd * paras['W' + str(l)]) / m
        grads['db' + str(l)] = np.sum(grads['dZ' + str(l)],
                                      axis=1, keepdims=True) / m

    return grads


def update_parameters_with_adam(paras, grads, v, s, t, lambd, m, L, learning_rate=0.01,
                                beta1=0.9, beta2=0.999,
                                epsilon=1e-8):  # Step 5A (Kind of necessary as it can speed up the minimizing cost
    # process): Updating each parameter with the value got from the backward propagation (dW, db) as it was the value
    # which tells how much this specific parameter was increasing the cost

    v_corrected = {}  # Initializing first moment estimate
    s_corrected = {}  # Initializing second moment estimate

    # Perform Adam update on all parameters
    for l in range(0, L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + \
            (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + \
            (1 - beta1) * grads['db' + str(l + 1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)
                                           ] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)
                                           ] / (1 - np.power(beta1, t))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + \
            (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + \
            (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)
                                           ] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)
                                           ] / (1 - np.power(beta2, t))

        # Update parameters. Inputs: "parameters, learning, v_corrected, s_corrected, epsilon". Output: "parameters".
        paras["W" + str(l + 1)] = (1 - (lambd * learning_rate) / m) * paras["W" + str(l + 1)] - learning_rate * \
            v_corrected["dW" + str(l + 1)] / \
            np.sqrt(s["dW" + str(l + 1)] + epsilon)
        paras["b" + str(l + 1)] = paras["b" + str(l + 1)] - learning_rate * v_corrected[
            "db" + str(l + 1)] / np.sqrt(s["db" + str(l + 1)] + epsilon)

    return paras, v, s


# Step 5B (Should use upper one though): Updating each
def update_parameters(paras, grads, learning_rate, L):
    # parameter with the value got from the backward propagation (dW, db) as it was the value
    # which tells how much this specific parameter was increasing the cost

    for l in range(1, L + 1):
        paras["W" + str(l)] = paras["W" + str(l)] - \
            learning_rate * grads["dW" + str(l)]
        paras["b" + str(l)] = paras["b" + str(l)] - \
            learning_rate * grads["db" + str(l)]

    return paras


# L-Layer Neural Network (MAIN)
def model(x, y, dims, learning_rate=0.01, iterations=100, lambd=0, mini_batch_size=64):

    # x shape - (features, examples)
    # y shape - (features, examples)

    # dims = [features of (x, y), L1, L2,..., LN] (Sample: [19, 10, 1])

    # np.random.seed(1)

    m = x.shape[1]
    L = len(dims) - 1

    paras, v, s = initialize_parameters(dims)  # Step 1

    t = 0
    cache = {}
    costs = []
    cost = 0.0

    for i in range(iterations):

        mini_batches = random_mini_batches(x, y,
                                           mini_batch_size)  # Step 1.2 (Optional but i have used it as it's faster)

        for mini_batch in mini_batches:
            x_mini, y_mini = mini_batch

            # Step 2: Forward propagation to get predicted values
            cache = forward_propagation(x_mini, paras, L)

            cost = calculate_cost(y_mini, paras, cache, L,
                                  lambd)  # Step 3: Calculating cost i.e., difference b/w actual and predicted values

            grads = backward_propagation(y_mini, paras, cache, lambd,
                                         L)  # Step 4: Backward propagation to get how much each parameter has effect on cost and how much and which direction should we decrease them

            t = t + 1  # Used because of step 1.2 and step 5A
            paras, v, c = update_parameters_with_adam(paras, grads, v, s, t, lambd, m, L,
                                                      learning_rate)  # Step 5A: Update parameters with Adam method to get the better parameters than the previous iteration (Loop)

            # paras = update_parameters(paras, grads, learning_rate, L)  # Step 5B: Simple update (Shouldn't be used)

        costs.append(cost)

        if i % 1000 == 0:
            print("Cost after {} iteration: {}".format(i, costs[i]))

    return paras, cache, costs
