# File: kmeans.py
# --------------------
# Runs k-mean clustering on the feature vectors of all the
# faces in the database, then finds the average rating of each
# cluster and other metrics.

import math
from algorithm import *
from util import *
import random

def calculate_cost(point, k):
    cost = sum((x - k[i])**2 if i in k else x**2 for i,x in point.items())  
    cost += sum(k[i]**2 if i not in point else 0 for i,x in k.items())
    return cost

def reassign_centroid(mu, k, points):
    sums = {}
    num_points = len(points)

    if num_points > 0:
        mu[k] = {}

        for point in points:
            increment(mu[k], 1 / float(num_points), point)

def kmeans(examples, K, max_iters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    k: number of desired clusters. Assume that 0 < K <= |examples|.
    max_iters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length k list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # random.seed()
    n = len(examples)
    
    # centers
    random_keys = random.sample(examples, K)
    mu = [examples[key] for key in random_keys]

    # assignments
    assignments = {}
    prev_cost = 0

    for iters in range(max_iters):
        # step 1: estimate assignments
        total_cost = 0
        for face_id, features in examples.items():
            cost, assignments[face_id] = min((calculate_cost(features, mu[k]), k) for k in range(K))
            total_cost += cost

        # print "mu = %s, total cost = %s" % (mu, total_cost)

        # step 2: estimate centers
        for k in range(K):
            points = [features for face_id, features in examples.items() if assignments[face_id] == k]
            reassign_centroid(mu, k, points)

        if total_cost == prev_cost: return (mu, assignments, total_cost)
        prev_cost = total_cost

        print "iter = %s, total cost = %s" % (iters, total_cost)

    return (mu, assignments, prev_cost)

def average_face(K, mu, assignments, ratings):
    averages_file = open('averages.txt', 'wb')
    cluster_files = []
    for i in range(K): cluster_files.append(open('cluster' + str(i) + '.txt', 'wb'))
    average_ratings = [0 for k in range(K)]
    counts = [0 for k in range(K)]
    variance = [0 for k in range(K)]

    for face_id in assignments:
        cluster = assignments[face_id]
        average_ratings[cluster] += float(ratings[face_id])
        counts[cluster] += 1

    for k in range(K):
        average_ratings[k] /= float(counts[k])
        print average_ratings[k]
        averages_file.write('cluster: %s, %s \n' % (k, mu[k]))
        averages_file.write('average rating: %s \n' % average_ratings[k])
        cluster_files[k].write('cluster: %s, %s \n' % (k, mu[k]))
        cluster_files[k].write('average rating: %s \n \n' % average_ratings[k])

        for face_id in assignments:
            if assignments[face_id] == k:
                cluster_files[k].write('%s \n' % (face_id))
                variance[k] += (average_ratings[k] - float(ratings[face_id]))**2

        averages_file.write('variance: %s \n\n' % variance[k])

    for i in range(K): cluster_files[i].close()
    averages_file.close()

    print 'total variance: %s' % sum(variance)

def main(argv):
    data = read_file(argv[0])
    feature_extracted_data = {}
    ratings = {}
    for face_id, attr in data.items():
        feature_extracted_data[face_id] = extract_features(attr['attributes'])
        ratings[face_id] = attr['rating']

    # arguments: cluster.txt, num_clusters, iterations, feature_extractor
    mu, assignments, prev_cost = kmeans(feature_extracted_data, 10, 100)
    average_face(10, mu, assignments, ratings)

if __name__ == "__main__":
    main(sys.argv[1:])
