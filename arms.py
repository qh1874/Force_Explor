""" Packages import """
import numpy as np
from scipy.stats import truncnorm as trunc_norm
from utils import convert_tg_mean


class AbstractArm(object):
    def __init__(self, mean, variance, random_state):
        """
        :param mean: float, expectation of the arm
        :param variance: float, variance of the arm
        :param random_state: int, seed to make experiments reproducible
        """
        self.mean = mean
        self.variance = variance
        self.local_random = np.random.RandomState(random_state)
      
    def sample(self):
        pass


class ArmBernoulli(AbstractArm):
    def __init__(self, p, random_state=0):
        """
        :param p: float, mean parameter
        :param random_state: int, seed to make experiments reproducible
        """
        self.p = p
        super(ArmBernoulli, self).__init__(mean=p,
                                           variance=p * (1. - p),
                                           random_state=random_state)
        

    def sample(self):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return (self.local_random.rand(1) < self.p)*1.


class ArmBeta(AbstractArm):
    def __init__(self, a, b, random_state=0):
        """
        :param a: int, alpha coefficient in beta distribution
        :param b: int, beta coefficient in beta distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.a = a
        self.b = b
        super(ArmBeta, self).__init__(mean=a/(a + b),
                                      variance=(a * b)/((a + b) ** 2 * (a + b + 1)),
                                      random_state=random_state)

    def sample(self):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return self.local_random.beta(self.a, self.b, 1)


class ArmGaussian(AbstractArm):
    def __init__(self, mu, eta, random_state=0):
        """
        :param mu: float, mean parameter in gaussian distribution
        :param eta: float, std parameter in gaussian distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.mu = mu
        self.eta = eta
        super(ArmGaussian, self).__init__(mean=mu,
                                          variance=eta**2,
                                          random_state=random_state)

    def sample(self):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return self.local_random.normal(self.mu, self.eta, 1)
        # return np.random.normal(self.mu, self.eta, 1)
        
        

class ArmFinite(AbstractArm):
    def __init__(self, X, P, random_state=0):
        """
        :param X: np.array, support of the distribution
        :param P: np.array, associated probabilities
        :param random_state: int, seed to make experiments reproducible
        """
        self.X = X
        self.P = P
        mean = np.sum(X * P)
        super(ArmFinite, self).__init__(mean=mean,
                                        variance=np.sum(X ** 2 * P) - mean ** 2,
                                        random_state=random_state)

    def sample(self):
        """
        Sampling strategy for an arm with a finite support and the associated probability distribution
        :return: float, a sample from the arm
        """
        i = self.local_random.choice(len(self.P), size=1, p=self.P)
        reward = self.X[i]
        return reward


class ArmExponential(AbstractArm):
    def __init__(self, p, random_state=0):
        """
        :param mu: float, mean parameter in exponential distribution
        :param eta: float, std parameter in exponential distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.p = p
        super(ArmExponential, self).__init__(mean=p,
                                          variance=p**2,
                                          random_state=random_state)

    def sample(self):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return self.local_random.exponential(self.p, 1)

class ArmPareto(AbstractArm):
    def __init__(self, a,x_m, random_state=0):
        """
        :param a: float, a>2, shape parameter(tail index) in pareto distribution
        :param x_m: float, minimum  value in pareto distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.a=a
        self.x_m=x_m
        super(ArmPareto, self).__init__(mean=a*x_m/(a-1),
                                          variance=(x_m/(a-1))**2*a/(a-2),
                                          random_state=random_state)

    def sample(self):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        return (self.local_random.pareto(self.a,1)+1)*self.x_m

class dirac():
    def __init__(self, c, random_state):
        """
        :param c: mean
        :param random_state: int, seed to make experiments reproducible
        """
        self.mean = c
        self.variance = 0
        self.local_random = np.random.RandomState(random_state)

    def sample(self):
        return [self.mean]


class ArmTG(AbstractArm):
    def __init__(self, mu, scale, random_state=0):
        """
        Truncated Gaussian distribution
        :param mu: mean
        :param random_state: int, seed to make experiments reproducible
        """
        self.mu = mu
        self.scale = scale
        self.dist = trunc_norm(-mu/scale, b=(1-mu)/scale, loc=mu, scale=scale)
        self.dist.random_state = random_state
        super(ArmTG, self).__init__(mean=convert_tg_mean(mu, scale), variance=scale**2,
                                      random_state=random_state)

    def sample(self):
        """
        Sampling strategy
        :return: float, a sample from the arm
        """
        x = self.local_random.normal(self.mu, self.scale, 1)
        return x * (x > 0) * (x < 1) + (x > 1)
