import math
import arff
import sys
import numpy

"""Returns a new list of z-score normalized values"""
def normalize(examples):
    #The sample mean of features
    mean = [sum(x)/float(len(examples)) for x in zip(*examples)]
    stdev = [numpy.std(x, ddof=1) for x in zip(*examples)]
    new = []
    j=0
    for x in examples:
        new.append([])
        for i in xrange(len(x)):
            if (stdev[i] != 0):
                new[j].append(x[i]-mean[i]/stdev[i])
            else:
                new[j].append(0)
        j+=1
    return new

"""Returns k-nearest neighbors of point q"""
def get_knn(k, q, dataset):
    #Store closest neighbors as {y1:dist, y2:dist...}
    closest={}
    for y in dataset:
        #Range is len-1 to ignore class label
        if (q != y):
            dist = eu_dist(q, y)
            if len(closest)<k:
                closest[tuple(y)]=dist
            else:
                max_dist = max(closest.values())
                if dist < max_dist:
                    del closest[max(closest, key=closest.get)]
                    closest[tuple(y)] = dist
    neighbors = []
    for c in closest:
        neighbors.append(list(c))
    return neighbors

"""Takes in the dataset as a list, # of nearest neighbors, 
   and # of expected outliers
   Returns a list of outliers sorted by probability of being an outlier"""
def findOutliers(dataset, minPts, num_outliers):	
    normalized = normalize(dataset)
    #The dataset represented as a dict:
    #{key = data example, value = [k nearest neighbors of q]}
    D = {}
    
    i=0 
    for data in normalized:
        data.append("ex"+str(i))
        i+=1
        for pos in xrange(len(data)-1):
            if math.isnan(data[pos]):
                data[pos] = 0.0
    
    
    found_outliers = []
    while minPts < 20:
        outlier_ranks = []
        outliers = []
        
        for n in normalized:
            D[tuple(n)] = get_knn(minPts, n, normalized)
        #{key = data example, value = outlier ranking}
    
        for q in D:
            outlier_ranks.append((q, LOF(list(q), D[q], D, minPts)))

        outlier_ranks = normalizeRanks(outlier_ranks)
        median=getMedian(outlier_ranks)
    
        #Find top num_outliers outliers with largest ranking
        for q in outlier_ranks:
            if q[1] > median:
                if len(outliers)>num_outliers:
                    min_outlier = min(outliers, key=lambda x:x[1])
                    if q[1] > min_outlier[1]:
                        outliers.remove(min_outlier)
                        outliers.append(q)
                else:
                    outliers.append(q)
        outliers = sorted(outliers, key=lambda x:x[1])
        outliers.reverse()
        found_outliers.append([x for x in outliers])
        minPts+=1
    #{example, outlier rank}
    outlier_avgs={}
    outlier_occurrences={}
    for x in found_outliers:
        for y in x:
            if y[0] in outlier_avgs:
                outlier_avgs[y[0]] += y[1]
                outlier_occurrences[y[0]] += 1
            else:
                outlier_avgs[y[0]] = y[1]
                outlier_occurrences[y[0]] = 1
    outliers=[]
    for x in outlier_avgs:
        outlier_avgs[x] *= outlier_occurrences[x]
    outlier_sorted = sorted(outlier_avgs, key=outlier_avgs.get)
    outlier_sorted.reverse()
    for x in xrange(num_outliers):
        outliers.append((outlier_sorted[x], outlier_avgs[outlier_sorted[x]]))
    return outliers

"""Returns the median of a given list"""
def getMedian(data):
    sorted_r = sorted(data, key=lambda x:x[1])
    if len(data)%2==0:
        first=sorted_r[len(data)/2][1]
        sec = sorted_r[(len(data)/2)-1][1]
        median = (first+sec)/2
    else:
        median = sorted_r[len(data)/2][1]
    return median

"""Takes in a list of outlier ranks, return a new list with normalized values"""
def normalizeRanks(ranks):
    mod_ranks=[]
    median = getMedian(ranks)
    s = [x[1] for x in ranks]
    stdev = numpy.std(s, ddof=1)
    for r in ranks:
        mod_ranks.append((r[0],((r[1]-median)/stdev)))
    return mod_ranks

"""Takes in a point, its k-nearest neighbors, dataset, and 
   Returns the Local Outlier Factor of a point given"""
def LOF(q, nn, D, minPts):
    psum = 0.0
    q_lrd = lrd(q,nn, minPts, D)
    if q_lrd == 0:
        q_lrd = 1.0
    for p in nn:
        psum += lrd(p, D[tuple(p)], minPts, D)/q_lrd
    return psum/minPts

"""Returns the Euclidean distance between given x and y points"""
def eu_dist(x, y):
    power_sum = 0.0
    for j in xrange(len(x)-1):
        power_sum += ((x[j]-y[j])**2)
    return math.sqrt(power_sum)

"""Takes in a point p and its nearest neighbors
   Returns the largest distance between p and its neighbors"""
def k_dist(p, nn):
    max_dist = 0.0
    for n in nn:
        dist = eu_dist(p, n)
        if (dist>max_dist):
            max_dist=dist
    return max_dist

"""Takes in point q, a neighboring point p, and p's k-nearest neighbors
   Returns the reach distance between q and p"""
def reach_dist(q, p, pnn):
    return max(k_dist(p, pnn), eu_dist(q,p))

"""Takes a point, its k-nearest neighbors, # of neighbors, and the dataset
   Returns the local reachability density of q"""
def lrd(q, nn, minPts, D):
    reach_sum=0
    for p in nn:
        reach_sum += reach_dist(q,p,D[tuple(p)])
    if reach_sum >0:
        return minPts/reach_sum
    else:
        return 0.0

"""Writes the given results to a file"""
def output(results):
    f = open(sys.argv[2], "w+")
    ex_pos = len(results[0][0])-1
    for r in results:    
        f.write(r[0][ex_pos] + ": " + str(r[1]) + "\n")
    f.close()

"""minPts: a # of nearest neighbor that define a neighborhood of a point
   Run algorithm on given a dataset and print results to given output file
   from stdin"""
def main(minPts):
    dataset = []
    for q in arff.load(sys.argv[1]):
        dataset.append(list(q))

    for q in dataset:
        #Make sure it is taking out the improper ex
        q.pop()
    if sys.argv[1]=="dataset1.arff":
        expected_outliers = int(math.ceil(len(dataset)*0.02))
    elif sys.argv[1]=="dataset2.arff":
        expected_outliers = int(math.ceil(len(dataset)*0.04))
    elif sys.argv[1]=="synthetic_dataset1.arff":
        expected_outliers = len(dataset)-2
    else:
        expected_outliers = 6
    #Print out the outliers by outlier-ness
    output(findOutliers(dataset, minPts, expected_outliers))

main(10)

