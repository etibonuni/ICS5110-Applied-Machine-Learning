import numpy as np

# Initialise prototypes (features,label) m
# Params: 
# 	labelSpec: array[numClasses, 2] containing <class, number of prototypes> tuples
#   limits: array[dim, 2] containing min, max for each dimension
#
# return array[numProtos, dim+1] for prototypes
#
def genPrototypes(labelSpec, limits):
	
	C = len(labelSpec[:,0])    # number of classes
	print("C=", C)
	
	dim = len(limits[:,0])
	print("dim=", dim)

	P=np.zeros((0, dim+1))
	
	firstLoop=True
	
	for c in range(0, C):
		M_c = labelSpec[c,1]
#		print("M_c=", M_c, "for c=", c)
		
		P_c = np.random.rand(M_c, dim+1)	
		
		for d in range(0, dim):
			P_c[:,d] = P_c[:,d]*(limits[d,1]-limits[d,0])+limits[d,0]
		
		P_c[:,-1] = labelSpec[c,0]
		if firstLoop:
			P = P_c
			firstLoop = False
		else:
			P = np.concatenate([P, P_c])
		
#		print("P (c=", c, ")=", P)
	return P
	
# function get nearest prototype
#
def get_nearest_prototype(features,prototype):
#	print("features=", features)
#	print(features.shape[1])
#	print(features.shape)
	K=len(prototype[:,0])
    
	F=np.full((K,features.shape[0]),features)
	diff=F-prototype[:,0:-1]
#	print("diff=", diff)
#    dist = diff[:,0]**2

#    for i in range(1, len(features)):
#    	dist = dist + diff[:,i]**2

	dist = np.sum(np.power(diff, 2), axis=1)    	
	dist=np.sqrt(dist)

#	print("dist=", dist)
#    print(diff)
#    print(dist)
	return dist,dist.argsort()

#f=np.matrix([[1,1],[2,2]])
#p=np.matrix([[0,0,0],[0.5,0.5,1]])
#print(f[0])
#print(get_nearest_prototype(f[0,:][:], p))

#m = np.matrix([[1,2],[3,4]])
#print(m)
#m1 = np.power(m, 2)
#print(m1)
#s = np.sum(m1, axis=1)
#print(s)
#print(np.sqrt(s))
