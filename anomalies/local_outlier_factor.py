from flask import Flask,redirect, url_for, request
from collections import defaultdict
import numpy as np
import pandas as pd
import heapq
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import os, gc
import anomalies.config as config

#Calculate the k-distance neighborhood:
def all_indices(value, inlist):
    out_indices = []
    idx = -1
    while True:
        try:
            idx = inlist.index(value, idx+1)
            out_indices.append(idx)
        except ValueError:
            break
    return out_indices


def detect_lof_mapper():


    # Calculate K distance
    k_distance = defaultdict(tuple)
    k_distance_neig = defaultdict(list)

    #price
    data = pd.read_csv('static/anomalies/features/'+request.form["currency"]+'_'+request.form["year"]+'.csv')
    #data = data.head(200)


    data_without_dates = data.drop(['Index'], axis = 1)
    values = data_without_dates.values #returns a numpy array

    #normalizing data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled_values = min_max_scaler.fit_transform(values)
    df = pd.DataFrame(x_scaled_values)


    instances = pd.DataFrame()
    instances_list = []
    #all_anormalies_1.csv
    #instances['Price_Gradient'] = df[1]
    #instances['CH_Gradient'] = df[2] #Index,Price_Gradient,Price,CH_Gradient,CH

    instances['Price'] = df[1]
    instances['CH_Gradient'] = df[2]
    n_partitions = 6

    #Let's get the pairwise distance between the points:
    #k = 2
    k = config.NEAREST_NEIGHBOURS
    distance = 'manhattan'

    print('here1')

    for i in range(n_partitions):
        partition = data.loc[instances.index % n_partitions == i]
        instances_list.append(instances.loc[instances.index%n_partitions==i])
        instances_list[i] = instances_list[i].as_matrix()
        x = np.squeeze(np.asarray(instances_list[i][:,0]))
        y = np.squeeze(np.asarray(instances_list[i][:,1]))
        #plt.cla()
        #plt.figure(1)
        #plt.scatter(x,y)
        dist = pairwise_distances(instances_list[i],metric=distance)

        #Let's calculate the k-distance. We will use heapq and get the k-nearest neighbors:
        print('here2')
        # For each data point
        for j in range(instances_list[i].shape[0]):
            print('here3')
            # Get its distance to all the other points.
            # Convert array into list for convienience
            distances = dist[j].tolist()
            # Get the K nearest neighbours
            ksmallest = heapq.nsmallest(k+1,distances)[1:][k-1]
            # Get their indices
            ksmallest_idx = distances.index(ksmallest)
            # For each data point store the K th nearest neighbour and its distance
            k_distance[j]=(ksmallest,ksmallest_idx)

        # For each data point
        for j in range(instances_list[i].shape[0]):
            # Get the points distances to its neighbours
            distances = dist[j].tolist()
            # Get the 1 to K nearest neighbours
            ksmallest = heapq.nsmallest(k+1,distances)[1:]
            ksmallest_set = set(ksmallest)
            ksmallest_idx = []
            # Get the indices of the K smallest elements
            for x in ksmallest_set:
                    ksmallest_idx.append(all_indices(x,distances))
            # Change a list of list to list
            ksmallest_idx = [item for sublist in ksmallest_idx for item in sublist]
            # For each data pont store the K distance neighbourhood
            k_distance_neig[j].extend(zip(ksmallest,ksmallest_idx))
        print('here4')


        #Then, calculate the reachability distance and LRD:
        #Local reachable density
        local_reach_density = defaultdict(float)
        for j in range(instances_list[i].shape[0]):
            # LRDs numerator, number of K distance neighbourhood
            no_neighbours = len(k_distance_neig[j])
            denom_sum = 0
            # Reachability distance sum
            for neigh in k_distance_neig[j]:
                # maximum(K-Distance(P), Distance(P,Q))
                denom_sum+=max(k_distance[neigh[1]][0],neigh[0])
            local_reach_density[j] = no_neighbours/(1.0*denom_sum)

        lof_list =[]
        #Local Outlier Factor
        for j in range(instances_list[i].shape[0]):
            lrd_sum = 0
            rdist_sum = 0
            for neigh in k_distance_neig[j]:
                lrd_sum+=local_reach_density[neigh[1]]
                rdist_sum+=max(k_distance[neigh[1]][0],neigh[0])
            lof_list.append((j,lrd_sum*rdist_sum))

        list_of_lists = [list(elem) for elem in lof_list]
        list_of_lists = [list(elem) for elem in zip(*list_of_lists)]
        lof_values = list_of_lists[1]
        print('here5')
        partition['lof'] = lof_values
        print(partition)
        if os.path.exists('static/anomalies/local_outlier_factors/' +
                request.form["currency"] + '_' + request.form["year"]+str(i)+'.csv'):
            os.remove('static/anomalies/local_outlier_factors/' +
                request.form["currency"] + '_' + request.form["year"]+str(i)+'.csv')

        gc.collect()
        partition.to_csv('static/anomalies/local_outlier_factors/' +
            request.form["currency"] + '_' + request.form["year"]+str(i)+'.csv')


        print('here6')

    return request.form["year"], request.form["from_month"], request.form["to_month"], request.form["currency"], "done"






