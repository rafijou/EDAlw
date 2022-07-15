# Estimation of Distribution Algorithm
import numpy
from deap import base, creator, tools, benchmarks, algorithms
from deap.tools import HallOfFame

from indGen import IndGen
from EDA import EDA
from myAlgorithms import eaGenerateUpdate, eaGenerateUpdateW

creator.create("MyFitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.MyFitness)

toolbox = base.Toolbox()

objFun = benchmarks.sphere
toolbox.register("evaluate", objFun)

i_gen = IndGen()

toolbox.register("individual", tools.initIterate, creator.Individual, i_gen.ind_Gen)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
init_pop = toolbox.population(n=100)


def main():
    LAMBDA = 300
    MU = int(LAMBDA / 10)
    strategy = EDA(init_pop, mu=MU, lambda_=LAMBDA)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof: HallOfFame = tools.HallOfFame(1, similar=numpy.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    eaGenerateUpdateW(toolbox, n_max_gen=150, stats=stats, halloffame=hof)

    print(hof[0], " ", hof[0].fitness.values[0])

    return hof[0].fitness.values[0]


if __name__ == "__main__":
    main()
