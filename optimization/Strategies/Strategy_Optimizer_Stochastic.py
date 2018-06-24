from flask import Flask,redirect, url_for, request
from random import *
import time
from .. import Application
from .. import Profit_Calculator
from .. import Risk_Calculator

from operator import add
from functools import reduce
import csv

def initialize():
    global minK_periods
    global maxK_periods
    global stepK_periods

    global minD_periods
    global maxD_periods
    global stepD_periods

    global min_higher_lines
    global max_higher_lines
    global step_higher_lines

    global min_lower_lines
    global max_lower_lines
    global step_lower_lines

    global minStopLosss
    global maxStopLosss
    global stepStopLosss

    global minTakeProfits
    global maxTakeProfits
    global stepTakeProfits

    minK_periods = int(request.form["minK_periods"])
    maxK_periods = int(request.form["maxK_periods"])
    stepK_periods = int(request.form["stepK_periods"])

    minD_periods = int(request.form["minD_periods"])
    maxD_periods = int(request.form["maxD_periods"])
    stepD_periods = int(request.form["stepD_periods"])

    min_higher_lines = int(request.form["min_higher_lines"])
    max_higher_lines = int(request.form["max_higher_lines"])
    step_higher_lines = int(request.form["step_higher_lines"])

    min_lower_lines = int(request.form["min_lower_lines"])
    max_lower_lines = int(request.form["max_lower_lines"])
    step_lower_lines = int(request.form["step_lower_lines"])

    minStopLosss = int(request.form["minStopLosss"])
    maxStopLosss = int(request.form["maxStopLosss"])
    stepStopLosss = int(request.form["stepStopLosss"])

    minTakeProfits = int(request.form["minTakeProfits"])
    maxTakeProfits = int(request.form["maxTakeProfits"])
    stepTakeProfits = int(request.form["stepTakeProfits"])

    start_time = time.time()

    pop = population(count, minK_periods, maxK_periods, stepK_periods,
                     minD_periods, maxD_periods, stepD_periods,
                     min_higher_lines, max_higher_lines, step_higher_lines,
                     min_lower_lines, max_lower_lines, step_lower_lines,
                     minStopLosss ,maxStopLosss, stepStopLosss,
                     minTakeProfits, maxTakeProfits, stepTakeProfits)

    pool_graded = propagate(pop)
    pool_graded_sorted = sorted(pool_graded, reverse=True)

    print("Sorted graded: ", pool_graded_sorted)
    children = []
    for n in range(4):
        children = roulette_wheel_pop(pool_graded_sorted)
        pool_graded_sorted.extend(propagate(children))
        print("pool_graded_sorted",pool_graded_sorted)
        pool_graded_sorted = mutate(pool_graded_sorted)
        print("pool_graded_sorted", pool_graded_sorted)
        pool_graded_sorted = sorted(pool_graded_sorted, reverse=True)

    print("final",pool_graded_sorted)
    print("--- %s seconds ---" % (time.time() - start_time))
    return pool_graded_sorted


def individual(minK_periods, maxK_periods, stepK_periods,
                     minD_periods, maxD_periods, stepD_periods,
                     min_higher_lines, max_higher_lines, step_higher_lines,
                     min_lower_lines, max_lower_lines, step_lower_lines,
                     minStopLosss ,maxStopLosss, stepStopLosss,
                     minTakeProfits, maxTakeProfits, stepTakeProfits):
    'Create a member of the population.'
    # print([randrange(minShortMA, maxShortMA, stepShortMA)-randrange(minLongMA, maxLongMA, stepLongMA), randrange(minStopLoss, maxStopLoss, stepStopLoss), randrange(minTakeProfit, maxTakeProfit, stepTakeProfit)])

    tmp = [randrange(minK_periods, maxK_periods, stepK_periods),
               randrange(minD_periods, maxD_periods, stepD_periods),
               randrange(min_higher_lines, max_higher_lines, step_higher_lines),
               randrange(min_lower_lines, max_lower_lines, step_lower_lines),
               randrange(minStopLosss ,maxStopLosss, stepStopLosss),
               randrange(minTakeProfits, maxTakeProfits, stepTakeProfits)]

    return tmp


def population(count, minK_periods, maxK_periods, stepK_periods,
                     minD_periods, maxD_periods, stepD_periods,
                     min_higher_lines, max_higher_lines, step_higher_lines,
                     min_lower_lines, max_lower_lines, step_lower_lines,
                     minStopLosss ,maxStopLosss, stepStopLosss,
                     minTakeProfits, maxTakeProfits, stepTakeProfits):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values

    """
    return [individual(minK_periods, maxK_periods, stepK_periods,
                     minD_periods, maxD_periods, stepD_periods,
                     min_higher_lines, max_higher_lines, step_higher_lines,
                     min_lower_lines, max_lower_lines, step_lower_lines,
                     minStopLosss ,maxStopLosss, stepStopLosss,
                     minTakeProfits, maxTakeProfits, stepTakeProfits) for x in range(count)]


def fitness(individual):
    fitness = Profit_Calculator.fitness(individual,"Stochastic")
    #fitness = individual[0] + individual[1] + individual[2] + individual[3]

    #print("fitness of", individual, " ", fitness)
    return fitness


def propagate(pop):
    print("poplength", len(pop))
    retain = 0.2
    random_select = 0.05

    graded = [(fitness(x), x) for x in pop]
    print("gradedwith fitness: ", graded)


    #modGraded = [x[1] for x in sorted(graded, reverse=True)]
    #print("Modified Graded: ", modGraded)

    print("graded",graded)
    return(graded)




def roulette_wheel_pop(sortedGraded):
    retainPercentage = 0.15
    retain = int(retainPercentage * len(sortedGraded))
    print("\n ###Roulette Wheel###")
    #print("sortedGraded", sortedGraded)
    fitnessList = [x[0] for x in sortedGraded]
    sortedGradedPopulation = [x[1] for x in sortedGraded]

    #print("fitnessList" , fitnessList)

    total_fit = float(sum(fitnessList))
    #print("total fitness",total_fit)
    relative_fitness = [f / total_fit for f in fitnessList]
    #print("relative fitness" , relative_fitness)
    probabilities = [sum(relative_fitness[:i + 1])
                     for i in range(len(relative_fitness))]
    #print("probabilities",probabilities)

    parents = []
    children = []
    for m in range(retain):
        for n in range(2):
            r = random()
            #print("r", r)
            for (i, individual) in enumerate(sortedGradedPopulation):
                #print("i", i)
                #print("ind", individual)
                #print("prob",probabilities[i])
                if r <= probabilities[i]:
                    #print("yes")
                    parents.append(list(individual))
                    break

        print("parents", m , parents)
        children.append(crossover(parents))
        print("children",children)
        parents = []
    return children




def crossover(parents):
    male = parents[0]
    female = parents[1]
    half = len(male) / 2
    child = male[:int(half)] + female[int(half):]
    return(child)

def mutate(pool):
    mutation_percentage = 0.1
    for i,chromo in enumerate(pool):
        if mutation_percentage > random():
            pos_to_mutate = randint(0, len(chromo[1]) - 1)
            tmp = individual(minK_periods, maxK_periods, stepK_periods,
                     minD_periods, maxD_periods, stepD_periods,
                     min_higher_lines, max_higher_lines, step_higher_lines,
                     min_lower_lines, max_lower_lines, step_lower_lines,
                     minStopLosss ,maxStopLosss, stepStopLosss,
                     minTakeProfits, maxTakeProfits, stepTakeProfits)
            chromo[1][pos_to_mutate] = tmp[pos_to_mutate]
            #print("chromo",chromo[0],chromo)
            chromo = (fitness(chromo[1]),chromo[1])
            print("chromo", chromo[0],chromo)
            pool[i] = chromo

    return pool



def finalResult(parents):
    print("ccc")
    """evo = []
    for i in range(1):
        evo=evolve(parents)
        #print("gen",i," ",evo)
        parent_profit = [(fitness(x), x) for x in evo]
        #print("gen",i," ",parent_profit)
    print("final",evo)"""
    print("final result: ", parents)


# minShortMA, maxShortMA, stepShortMA = 0, 0, 0
# minLongMA, maxLongMA, stepLongMA = 0, 0, 0
# minStopLoss, stepStopLoss, maxStopLoss = 0, 0, 0
# minTakeProfit, stepTakeProfit, maxTakeProfit = 0, 0, 0

# pop = population(count)
# parents=evolve(pop)
# print(len(parents))
increment = 0
count = 100

#initialize(10, 50, 2, 5, 200, 5, 0, 400, 100, 0, 1000, 100)