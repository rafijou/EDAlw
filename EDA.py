from operator import attrgetter
from sklearn.covariance import LedoitWolf
import numpy.random as np

lw = LedoitWolf()


class EDA(object):
    def __init__(self, init_population, mu, lambda_):

        sorted_pop = sorted(init_population, key=attrgetter("fitness"), reverse=True)
        self.mu = mu
        self.best = sorted_pop[:self.mu]

        lw.fit(self.best)

        self.location_ = lw.location_
        self.dim = len(lw.location_)
        self.cov_ = lw.covariance_
        self.lambda_ = lambda_

    def generate(self, ind_init):

        nrg = np.default_rng()
        arz = [nrg.multivariate_normal(self.location_, self.cov_) for i in range(self.lambda_-self.mu)]

        return list(map(ind_init, arz))

    def update(self, population):

        sorted_pop = sorted(self.best + population, key=attrgetter("fitness"), reverse=True)

        self.best = sorted_pop[:self.mu]

        lw.fit(self.best)
        self.cov_ = lw.covariance_
        self.location_ = lw.location_
