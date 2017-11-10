import numpy as np

def euclideanDist(x1, y1, x2, y2):
    return (x2-x1)**2+(y2-y1)**2
    
def euclideaanDist2(v1, v2):
		tot=0
		for i in range(0,v1.size):
			tot = tot + (v2[i]-v1[i])**2
			
		return tot
		
def generateTestPoints(centroids, numPoints, sigma):
	dim = centroids.shape[1]
	numCentroids = centroids.shape[0]
	points = np.empty([numCentroids, numPoints, dim])
	
	for c in range(0,numCentroids):
	    points[c] = sigma*(np.random.rand(numPoints, dim)-0.5)+centroids[c]
	    	    
#	print(points)
	inputs = points[0]
#	print(inputs)
	for i in range(1, points.shape[0]):
		inputs = np.concatenate((inputs, points[i]))
		
	np.random.shuffle(inputs)
	
	return inputs
	
