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
    global minShortMA
    global maxShortMA
    global stepShortMA
    global minLongMA
    global maxLongMA
    global stepLongMA
    global minStopLoss
    global maxStopLoss
    global stepStopLoss
    global minTakeProfit
    global maxTakeProfit
    global stepTakeProfit
    minShortMA = int(request.form["minShortMA"])
    maxShortMA = int(request.form["maxShortMA"])
    stepShortMA = int(request.form["stepShortMA"])

    minLongMA = int(request.form["minLongMA"])
    maxLongMA = int(request.form["maxLongMA"])
    stepLongMA = int(request.form["stepLongMA"])

    minStopLoss = int(request.form["minStopLoss"])
    maxStopLoss = int(request.form["maxStopLoss"])
    stepStopLoss = int(request.form["stepStopLoss"])

    minTakeProfit = int(request.form["minTakeProfit"])
    maxTakeProfit = int(request.form["maxTakeProfit"])
    stepTakeProfit = int(request.form["stepTakeProfit"])

    start_time = time.time()

    pop = population(count, minShortMA, maxShortMA, stepShortMA, minLongMA, maxLongMA, stepLongMA, minStopLoss,
                     maxStopLoss, stepStopLoss, minTakeProfit, maxTakeProfit, stepTakeProfit)

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


def individual(minShortMA, maxShortMA, stepShortMA, minLongMA, maxLongMA, stepLongMA, minStopLoss, maxStopLoss,
               stepStopLoss, minTakeProfit, maxTakeProfit, stepTakeProfit):
    'Create a member of the population.'
    # print([randrange(minShortMA, maxShortMA, stepShortMA)-randrange(minLongMA, maxLongMA, stepLongMA), randrange(minStopLoss, maxStopLoss, stepStopLoss), randrange(minTakeProfit, maxTakeProfit, stepTakeProfit)])
    while True:
        tmp = [randrange(minShortMA, maxShortMA, stepShortMA), randrange(minLongMA, maxLongMA, stepLongMA),
               randrange(minStopLoss, maxStopLoss, stepStopLoss),
               randrange(minTakeProfit, maxTakeProfit, stepTakeProfit)]
        if (tmp[0] < tmp[1]):
            break

    return tmp


def population(count, minShortMA, maxShortMA, stepShortMA, minLongMA, maxLongMA, stepLongMA, minStopLoss, maxStopLoss,
               stepStopLoss, minTakeProfit, maxTakeProfit, stepTakeProfit):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values

    """
    return [individual(minShortMA, maxShortMA, stepShortMA, minLongMA, maxLongMA, stepLongMA, minStopLoss, maxStopLoss,
                       stepStopLoss, minTakeProfit, maxTakeProfit, stepTakeProfit) for x in range(count)]


def fitness(individual):
    fitness = Profit_Calculator.fitness(individual,"Moving Average")
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
            tmp = individual(minShortMA, maxShortMA, stepShortMA, minLongMA, maxLongMA, stepLongMA, minStopLoss,
                     maxStopLoss, stepStopLoss, minTakeProfit, maxTakeProfit, stepTakeProfit)
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