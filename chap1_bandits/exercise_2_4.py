#!/usr/bin/env python
# -*- coding: UTF8 -*-

import numpy
from numpy.random import random, randn, normal, randint

class Bandit(object):
    """
    Class that generates an n-armed bandit problem.

    Example:
    >>> bandit = Bandit(n=20, qa=(0,2), noise=(0,1))
    >>> print bandit(0) # get reward for action 0
    -0.11000063101723367
    >>> print bandit(15) # get reward for action 15
    2.0949280547560729
    """

    def __init__(self, n=10, qa=(0,1), noise=(0,1)):
        self.n = n
        # transform (mean, variance) tuples into samples of size n
        if isinstance(qa, tuple):
            qa = self.tup2samples(qa)
        elif isinstance(qa, (int, float)):
            qa = qa * numpy.ones(self.n)
        self.qa = qa
        self.noise = noise
        self.best_actions = []
        self.best_rewards = []

    def tup2samples(self, tup):
        mu, sigma2 = tup
        samples = normal(loc=mu, scale=numpy.sqrt(sigma2), size=self.n)
        return samples

    def __call__(self, action):
        # generate noise
        noise = self.tup2samples(self.noise)
        # compute the reward for each action
        reward = self.qa + noise
        # retrieve the best reward and the corresponding action and store it
        best_action = reward.argmax()
        best_reward = reward[best_action]
        self.best_actions.append(best_action)
        self.best_rewards.append(best_reward)
        # return the reward
        return reward[action]


class DriftingBandit(Bandit):
    """
    Variation of the n-armed bandit problem with drifting action values.
    Works exactly the same as a Bandit object, but with an additional parameter at creation (drift_sigma).
    """

    def __init__(self, n=10, qa=0., noise=(0,1), drift_sigma=0.05):
        super(DriftingBandit, self).__init__(n=n, qa=qa, noise=noise)
        self.drift_sigma = drift_sigma

    def __call__(self, action):
        reward = super(DriftingBandit, self).__call__(action)
        # random walk
        self.qa += normal(loc=0., scale=self.drift_sigma, size=self.n)
        return reward


class EpsGreedyLearner(object):

    def __init__(self, bandit, initval=0., eps=0., stepsize='average'):
        self.eps = eps
        self.bandit = bandit
        self.stepsize = stepsize
        self.Qa = initval * numpy.ones(self.bandit.n)
        self.ka = numpy.ones(self.bandit.n, dtype='int')
        self.rewards = []
        self.actions = []

    def step(self):
        if random() > self.eps:
            action = self.Qa.argmax()
        else:
            action = randint(self.bandit.n)
        self.ka[action] += 1
        reward = self.bandit(action)
        self.rewards.append(reward)
        self.actions.append(action)
        if self.stepsize in ['average', 'mean', 'k']:
            self.Qa[action] += (reward - self.Qa[action]) / self.ka[action]
        else:
            self.Qa[action] += (reward - self.Qa[action]) * self.stepsize
        return action, reward

    def run(self, nsteps):
        for t in xrange(nsteps):
            self.step()
        return self.actions, self.rewards


def run_drift(T=2000, eps=0.1, initval=0., stepsize='average'):
    driftbandit = DriftingBandit()
    egreedy = EpsGreedyLearner(driftbandit, eps=eps, initval=initval, stepsize=stepsize)
    actions, rewards = egreedy.run(T)
    isbest = numpy.asarray(driftbandit.best_actions) == numpy.asarray(actions)
    return numpy.asarray(actions), numpy.asarray(rewards), isbest.astype('float')

def experiments(n, T=2000, eps=0.1, initval=0., stepsize='average'):
    rewds, isb = [], []
    for i in xrange(n):
        actions, rewards, isbest = experiment(T=T, eps=eps, initval=initval, stepsize=stepsize)
        rewds.append(rewards)
        isb.append(isbest)
    return numpy.asarray(rewds).mean(axis=0), numpy.asarray(isb).mean(axis=0)



