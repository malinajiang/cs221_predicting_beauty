import math
from algorithm import *
from util import *
import random

def calculateCost(point, k):
    
    cost = sum((x - k[i])**2 if i in k else x**2 for i,x in point.items())  
    cost += sum(k[i]**2 if i not in point else 0 for i,x in k.items())
    return cost

def reassignCentroid(mu, k, myPoints):
    sums = {}
    numPoints = len(myPoints)
    if len(myPoints) > 0:
        mu[k] = {}
        for point in myPoints:
            increment(mu[k], 1/float(len(myPoints)), point)

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # random.seed()
    n = len(examples)
    # Centers
    # mu = [copy.deepcopy(examples[random.randint(0, n - 1)]) for k in range(K)]
    random_keys = random.sample(examples, K)
    mu = [examples[key] for key in random_keys]
    # print mu

    # Assignments
    assignments = {}
    prevCost = 0

    for iters in range(maxIters):
        #Step 1: estimate assignments
        totalCost = 0
        for face_id, features in examples.items():
            cost, assignments[face_id] = min((calculateCost(features, mu[k]), k) for k in range(K))
            totalCost += cost

        print "mu = %s, totalCost = %s" % (mu, totalCost)

        # Step 2: estimate centers
        for k in range(K):
            myPoints = [features for face_id, features in examples.items() if assignments[face_id] == k]
            reassignCentroid(mu, k, myPoints)

        if totalCost == prevCost: return (mu, assignments, totalCost)
        prevCost = totalCost

    # print "assignments %s" % assignments
    # print "centers ", mu
    # print totalCost
    return (mu, assignments, prevCost)
    # return 'done'


def main(argv):
    data = read_file(argv[0])
    feature_extracted_data = {}
    for face_id, attr in data.items():
        feature_extracted_data[face_id] = extract_features(attr['attributes'])

    kmeans(feature_extracted_data, 10, 10)
    # arguments: cluster.txt, num clusters, iterations, feature extractor

if __name__ == "__main__":
    main(sys.argv[1:])
