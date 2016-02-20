import numpy as np
import matplotlib.pyplot as plt

forward = np.zeros((60, 60))

def main():
    experiment(trials = 100, light_on = [10, 20], alpha = 0.20, gamma = 0.99)

def experiment(light_on = [10],
               trials = 100,
               time_steps = 60,
               alpha = 0.1,
               gamma = 0.5):

    global forward
    for i in range(60):
        if i < 59:
            forward[i][i + 1] = 1
    
    weights = np.array([0. for _ in range(time_steps)])
    X = np.array([[0. for i in range(time_steps)] for j in range(time_steps)])

    reward = np.array([0. for _ in range(time_steps)])
    reward[time_steps - 1] = 1

    for t in light_on:
        turn_on(X, t)

    for i in range(trials):
        V_hat, delta = trial(X, weights, reward, alpha, gamma)
        if i == i:
            plot_data(V_hat, delta)

    turn_on(X, 20, off=True)

    V_hat, delta = trial(X, weights, reward, alpha, gamma)
    plot_data(V_hat, delta)

def trial(X, weights, reward, alpha, gamma, log = False):
    global forward
    time_steps = len(X)
    
    V_hat = predict_values(weights, X)
    delta = reward + gamma * np.dot(forward, V_hat) - V_hat
    weights += alpha * np.dot(X.T, delta)

    return V_hat, delta

def turn_on(X, t, off = False):
    time_steps = len(X)
    for i in range(t, time_steps):
            X[i][i - t] = 1 if not off else 0.
    
def predict_values(weights, X):
    return np.dot(X, weights)

def plot_data(V_hat, delta):
    fig = plt.figure()
    
    ax = fig.add_subplot(211)
    plt.ylabel('V_hat')
    ax.plot(np.array([i for i in range(0, len(V_hat))]), V_hat, c='g')

    ax = fig.add_subplot(212)
    plt.ylabel('Dopamine')
    ax.plot(np.array([i for i in range(0, len(delta))]), delta, c='r')
    
    timer = fig.canvas.new_timer(interval = 10000)
    timer.add_callback(lambda : plt.close())
    timer.start()
    plt.show()
    
if __name__ == '__main__':
    main()
