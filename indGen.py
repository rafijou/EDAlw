import random

# This class is for generate individual <tuple fashion> of size
# ind_size in a range given for range_min <list> and range_max <list>
random.seed(1234)
ind_dim = 100
lower_bounds = tuple(-100 for i in range(ind_dim))
upper_bounds = tuple(100 for j in range(ind_dim))


class IndGen:

    def __init__(self, ind_size=ind_dim, range_min=lower_bounds, range_max=upper_bounds):
        self.ind_size = ind_size
        self.range_min = range_min
        self.range_max = range_max

    def ind_Gen(self):
        return tuple(random.uniform(self.range_min[i], self.range_max[i]) for i in range(self.ind_size))
