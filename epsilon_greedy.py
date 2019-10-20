'''
Author @SKKSaikia
CS234 - Epsilon Greedy
Project Athena
Date - 20/10/2019
'''

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self,m):
        self.m = m #true mean
        self.mean = 0 #initial mean
        self.N = 0 #number of runs

    def pull(self):
        return np.random.randn() + self.m

    def update(self,x): #x is the latest sample received from the Bandit
        self.N += 1
        self.mean = (1-1.0/self.N)*self.mean + (1.0/self.N)*x

def run_experiment(m1,m2,m3,eps,N): #m1,m2,m3 are 3 different means
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    data = np.empty(N)

    for i in range(N):
        p = np.random.random()
        if p < eps:
            j = np.random.choice(3) #pull a random arm
        else:
            j = np.argmax([b.mean for b in bandits]) #pull the arm with max val

        x = bandits[j].pull()
        bandits[j].update(x)

        data[i] = x
    cumulative_avg = np.cumsum(data) / (np.arange(N)+1)

    plt.plot(cumulative_avg)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print(b.mean)
    return cumulative_avg

if __name__ == '__main__':
    c_10 = run_experiment(1.0,2.0,3.0,0.1,100000)
    c_05 = run_experiment(1.0,2.0,3.0,0.05,100000)
    c_01 = run_experiment(1.0,2.0,3.0,0.01,100000)

    #log scale pyplot
    plt.plot(c_10, label = 'eps = 0.1')
    plt.plot(c_05, label = 'eps = 0.05')
    plt.plot(c_01, label = 'eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()

    #linear pyplot
    plt.plot(c_10, label = 'eps = 0.1')
    plt.plot(c_05, label = 'eps = 0.05')
    plt.plot(c_01, label = 'eps = 0.01')
    plt.legend()
    plt.show()
