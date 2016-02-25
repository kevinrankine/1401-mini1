import numpy as np
import matplotlib.pyplot as plt
import pylab
from copy import deepcopy

forward = None


def main():
    experiment(trials = 1000, light_on = [10,20], alpha = 0.20, gamma = 0.99, gaba_add = 1)
'''

- light_on : list of integers corresponding to the time steps at which stimulus
             is presented
- trials : number of trials
- time_steps : number of discrete steps per trial
- alpha : learning rate
- gamma : discount factor for future reward

'''
def experiment(light_on = [10],
               trials = 100,
               time_steps = 60,
               alpha = 0.1,
               gamma = 0.5,
               gaba_add =.1):

    '''
    I'll explain what forward does later, it's just an efficiency trick
    '''
    global forward
    forward = np.zeros((time_steps, time_steps))
    for i in range(time_steps):
        if i < time_steps - 1:
            forward[i][i + 1] = 1

    # weights corresponds to w in the paper
    weights = np.array([0. for _ in range(time_steps)])
    gweights = np.array([0. for _ in range(time_steps)])

    # X corresponds to x(t) from the paper
    # X[t][i] = 1 if on timestep t a stimulus was seen i steps ago
    # the for loop with turn_on(X, t) computes the values of x(t) given when
    # the stimulus is presented

    X = np.array([[0. for i in range(time_steps)] for j in range(time_steps)])
    #for t in light_on:
        #turn_on(X, t)
    turn_on(X, 10)

    # reward[t] corresponds to r(t) - the reward presented at time_step t
    # In this model the reward is presented at the last step, although we can change this
    reward = np.array([0. for _ in range(time_steps)])
    reward[time_steps - 1] = 1

    # this loop calls trial() #trials times
    # displays the plot cuberoot(#trials) times
    # stupid idea, please change
    for i in range(trials):
        V_hat, delta, gV_hat, gdelta = trial(X, weights,gweights, reward, alpha, gamma, gaba_add)
        if i % (int(pow(trials, 2./3)) - 1) == 0:
            plot_data(V_hat, delta,gV_hat, gdelta, "Trial #{0}\n Light on at t=10, Reward at t=60".format(i))

    # this step modifies X so that only the stimulus at t=10 is presented
    #turn_on(X, 20, off=True)
    #turn_on(X, 20)
    old_weights = deepcopy(weights)
    gold_weights = deepcopy(gweights)

#     # then we observe what happens on another trial
#     V_hat, delta = trial(X, weights, reward, alpha, gamma)
#     plot_data(V_hat, delta, "Light on at t=10,20, Reward at t=60")

#     # now we observe what happens if we don't present a reward instead
#     turn_on(X, 20, off=True)
#     #turn_on(X, 20)
#     reward[time_steps - 1] = 0
#     V_hat, delta = trial(X, old_weights, reward, alpha, gamma)
#     plot_data(V_hat, delta, "Light on at t=10, No Reward Presented at t=60")

    for i in [0.5, 1, 2]:
        reward[time_steps - 1] = i
        V_hat, delta, gV_hat, gdelta = trial(X, old_weights,gold_weights, reward, alpha, gamma,gaba_add)
        plot_data(V_hat, delta,gV_hat, gdelta, "Light on at t=10, Reward Presented at t=60, Reward=" + str(i))

    reward[time_steps - 1] = 0

    #reward presented early
    reward[time_steps - 31] = 1
    V_hat, delta, gV_hat, gdelta = trial(X, old_weights,gold_weights, reward, alpha, gamma,gaba_add)
    plot_data(V_hat, delta,gV_hat, gdelta, "Light on at t=10, Reward Presented at t=30")



'''
- trial() executes a single trial given the current parameters of the model

'''
def trial(X, weights, gweights, reward, alpha, gamma,gaba_add, log = False):
    global forward
    time_steps = len(X)

    # V_hat corresponds to our estimation of the value function V(t)
    # V_hat[t] = V(t)
    # calculated by taking V(t) = sum{w_i x[t][i]} - equation (4) in the paper
    V_hat = predict_values(weights, X)
    gV_hat = predict_values(gweights, X)
    gV_hat = gV_hat*gaba_add +gV_hat

    # delta[t] = delta(t) from the paper
    # delta corresponds to equation (3)
    # np.dot(forward, V_hat) makes a vector where the jth entry is V_hat(j + 1)

    delta = reward + gamma * np.dot(forward, V_hat) - V_hat
    gdelta = reward + gamma * np.dot(forward, gV_hat) - gV_hat

    # rule for updating the weights from equation (5) in the paper
    # just a compact "vectorized" form of the update rule
    weights += alpha * np.dot(X.T, delta)
    gweights += alpha * np.dot(X.T, gdelta)

    # we return our value function estimation and delta over the course of the trial
    return V_hat, delta,gV_hat, gdelta


'''
- This function does the same thing as trial except it updates weights after each time
  step rather than at the end like trial() does
'''

def online_trial(X, weights, reward, alpha, gamma, log = False):
    time_steps = len(X)
    deltas = []
    V_hats = []
    for t in range(time_steps):
        V_hat = predict_values(weights, X)
        V_hats.append(V_hat[t])
        delta = reward[t] + gamma * (0 if t == time_steps - 1 else \
                                     V_hat[t + 1]) - V_hat[t]
        weights += alpha * X[t] * delta
        deltas.append(delta)
    return np.array(V_hats), np.array(deltas)

# - helper function for modifying X to reflect that the light was turned on
#   at time t
# - can alternatively turn it off to reverse the previous call
def turn_on(X, t, off = False):
    time_steps = len(X)
    for i in range(t, time_steps):
            X[i][i - t] = 1 if not off else 0.

# helper function to compute equation (4) from the paper
def predict_values(weights, X):
    return np.dot(X, weights)

# plots the value function, delta for a trial with a title
# pretty please help make this better
def plot_data(V_hat, delta,gV_hat,gdelta, title):
    delta = (delta - np.mean(delta)) / np.sqrt(np.var(delta))
    gdelta = (gdelta - np.mean(gdelta)) / np.sqrt(np.var(gdelta))
    fig = plt.figure()

    ax = fig.add_subplot(211)
    plt.title(title)
    plt.ylabel('Predicted Reward')

    ax.plot(np.array([i for i in range(0, len(V_hat))]), V_hat, c='g')
    ax.plot(np.array([i for i in range(0, len(gV_hat))]), gV_hat, c='b')
    pylab.ylim([-1, 1])

    ax = fig.add_subplot(212)
    plt.ylabel('Delta (DA Firing Proxy)')
    plt.xlabel('Time Step')
    ax.plot(np.array([i for i in range(0, len(delta))]), delta, c='r')
    ax.plot(np.array([i for i in range(0, len(gdelta))]), gdelta, c='b')
    pylab.ylim([-10, 10])

    plt.show()

if __name__ == '__main__':
    main()
