
# coding: utf-8

# In[ ]:





# # LVQ x2 implementation in  python 
# 

# We first import two useful libraries
# 1. numpy (matrix algebra):  we use np as a shortcut
# 2. plyplot from matplotlib: useful for plotting charts: we use plt as a shortcut
# 3. use tab and shift+tab for help

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import LVQUtils as lu

# In[2]:


# this line plots graphs in line
#get_ipython().magic(u'matplotlib inline')


# ### First we generate a dataset

# In[3]:


# C_g is the array for centroids
# 
M_g = 3
np.random.seed(3)



# In[4]:


theta_deg = np.arange(0.0,181.0,10.0)
theta_rad = theta_deg*(np.pi/180.0)
C_g = np.zeros((len(theta_deg),2))
C_g[:,0]=np.cos(theta_rad)/2.5+0.5
C_g[:,1]=np.sin(theta_rad)/1.5+0.25

plt.figure()
plt.plot(C_g[:,0],C_g[:,1],'o')
plt.show


# In[5]:
simple=False
if (simple):
    # we can also choose the centroids arbitarily
    C_g=np.array([[.25,.25],
                 [.75,.5],
                 [.45,.75]])
    
    
    # In[6]:
    
    
    # Generate data set ( M=3, centroid, constant sigma)
    #
    sigma=0.095
    number=100
    #
    # storing the centroid index (note this may not correspond to teh same number from the k-means algorithm)
    X11=np.concatenate((sigma*np.random.randn(number,2)+C_g[0],np.full((number,1),0.0)),axis=1)
    X22=np.concatenate((sigma*np.random.randn(number,2)+C_g[1],np.full((number,1),1.0)),axis=1)
    X33=np.concatenate((sigma*np.random.randn(number,2)+C_g[2],np.full((number,1),2.0)),axis=1)
    #
    #X=np.concatenate((X1,X2,X3), axis=0)
    X=np.concatenate((X11,X22,X33), axis=0)
else:
    # Generate data set ( M=3, centroid, constant sigma)
    #
    sigma=0.04
    number=20
    number0=200
    #
    # storing the centroid index (note this may not correspond to teh same number from the k-means algorithm)
    #X1=np.concatenate((1.3*sigma*np.random.randn(number0,2)+C_g[0],np.full((number0,1),0.0)),axis=1)
    X=np.zeros([number*len(C_g[:,0])+number0,3])
    for i in range(len(C_g[:,0])):
        #print i
        X[i*number:(i*number)+number,:]=np.concatenate((sigma*np.random.randn(number,2)+C_g[i],np.full((number,1),1.0)),axis=1)
    
    i = len(C_g[:,0])
    #print sigma*np.random.randn(number0,2)+np.array([0.5,0.5]
    X[i*number:(i*number)+number0,:]=np.concatenate((sigma*np.random.randn(number0,2)+np.array([0.5,1.2]),np.full((number0,1),2.0)),axis=1)



#
#X=np.concatenate((X1,X2,X3), axis=0)
#X=np.concatenate((X1,X2,X3,X4,X5,X6,X7,X8), axis=0)
np.random.shuffle(X)
print(X.shape)
#print X
           


# In[6]:


# plot data set and centroids
plt.figure()
col={0:'bo',1:'go', 2:'co'}
for i in range(len(X[:,0])):
    plt.plot(X[i,0],X[i,1],col[int(X[i,2])])

plt.plot(C_g[:,0],C_g[:,1],'ro')
plt.axis([0, 1.5, 0, 1.5])
plt.show()

#split data set into train and test
split = int((len(X[:,0]))*0.7)
print("Split point = ",split)
X_train=np.asarray(X[0:split,:])
print("Train size = ",len(X_train[:,0]))
X_test=np.asarray(X[split:,:])
print("Test size =",len(X_test[:,0]))

requiredAccuracy=0.8
numProtos=50

labelSpec=np.reshape([0, numProtos, 1, numProtos, 2, numProtos], (3,2))
print(labelSpec)

limits=np.reshape([np.min(X[:,0]), np.max(X[:,0]), np.min(X[:,1]), np.max(X[:,1])], (2,2))

print("Limits=", limits)
Prt = lu.genPrototypes(labelSpec, limits)

plt.figure()
col={0:'bo',1:'go', 2:'co'}
for i in range(len(X[:,0])):
    plt.plot(X[i,0],X[i,1],col[int(X[i,2])])
    
plt.plot(Prt[:,0],Prt[:,1],'ro')
plt.axis([0, 1.5, 0, 1.5])
plt.show()

# In[ ]:
def trainLVQ(trainData, initPrt, epochs, rateFunc):
    Prot=initPrt.copy()
    for t in range(0,epochs):
        for f in range(0, len(trainData[:,0])):
            feature=trainData[f,:]
            #print(feature)
        #    print(get_nearest_prototype(feature, Prt))    
            (dist, cls) = lu.get_nearest_prototype(feature[0:-1], Prt)
        #    print("dist=", dist, ", cls=", cls)
    #        print(Prt[cls[0],2])
            prt = Prot[cls[0],0:-1]
            if (Prot[cls[0],-1]==feature[-1]):    
        #        print("Match ->", Prt[cls[0],:], "=", feature)
                newPrt = prt + rateFunc(t) * (feature[0:-1]-prt)
#                print("Match: Old Prt=", prt, ", new Prt=", Prt[cls[0],0:-1], ", rate=", rateFunc(t))
            else:
            #    print("No Match ->", Prt[cls[0],:], "!=", feature)
                newPrt = prt - rateFunc(t) * (feature[0:-1]-prt)                        
                
            Prot[cls[0],0:-1] =     newPrt            

#            prt = Prot[cls[1],0:-1]
#            if (Prot[cls[1],-1]==feature[-1]):    
#        #        print("Match ->", Prt[cls[0],:], "=", feature)
#                newPrt = prt + rateFunc(t) * 0.5 * (feature[0:-1]-prt)
##                print("Match: Old Prt=", prt, ", new Prt=", Prt[cls[0],0:-1], ", rate=", rateFunc(t))
#            else:
#            #    print("No Match ->", Prt[cls[0],:], "!=", feature)
#                newPrt = prt - rateFunc(t) * 0.5 * (feature[0:-1]-prt)                        
#                
#            Prot[cls[1],0:-1] =     newPrt                
    #            print("MisMatch: Old Prt=", prt, ", new Prt=", Prt[cls[0],0:-1], ", rate=", rateFunc(t))            
#        plt.figure()
#        plt.title("t="+str(t))
#        col={0:'bo',1:'go', 2:'co'}
#        for i in range(len(trainData[:,0])):
#            plt.plot(trainData[i,0],trainData[i,1], "kx")#col[int(trainData[i,2])])
#        
#        for i in range(0,len(Prot[:,0])):
#            plt.plot(Prot[i,0],Prot[i,1],col[int(Prot[i,2])])
#            
#        plt.axis([0, 1.5, 0, 1.5])
#        plt.show()
    return Prot
    
trainedPrt = trainLVQ(X_train, Prt, 50, lambda t: 0.5*np.exp(-0.1*t))
#trimmedPrt = trainedPrt.copy()
trimmedPrt = trainedPrt[trainedPrt[:,0]!=Prt[:,0],:]
span = limits[:,1]-limits[:,0]
trimmedPrt = trimmedPrt[(trimmedPrt[:,0]>limits[0,0]-(3*span[0]))]
trimmedPrt = trimmedPrt[(trimmedPrt[:,0]<limits[0,1]+(3*span[0]))]
trimmedPrt = trimmedPrt[(trimmedPrt[:,0]>limits[1,0]-(3*span[1]))]
trimmedPrt = trimmedPrt[(trimmedPrt[:,0]<limits[1,1]+(3*span[1]))]
deadPrt = trainedPrt[trainedPrt[:,0]==Prt[:,0],:]
print("Trimmed prototype set from", len(Prt[:,0]), "points to" , len(trimmedPrt[:,0]), "points")
#print(len(deadPrt[:,0]))
#print( "Trained Prototypes=", trainedPrt)

print(trimmedPrt)
plt.figure()
plt.title("Trained Prototypes")
col={0:'bo',1:'go', 2:'co'}
for i in range(len(X_train[:,0])):
    plt.plot(X_train[i,0],X_train[i,1],col[int(X_train[i,2])])

plt.plot(trimmedPrt[:,0],trimmedPrt[:,1],'ro')
plt.plot(deadPrt[:,0],deadPrt[:,1],'k.')
plt.axis([np.min(trimmedPrt[:,0]), np.max(trimmedPrt[:,0]), np.min(trimmedPrt[:,1]), np.max(trimmedPrt[:,1])])
plt.show()

def runTestSet(X_test, Prots):
    # Do test set
    predictedClasses = X_test[:,-1].copy()
    
    for i in range(0, len(X_test[:,0])):
        (dist,cls) = lu.get_nearest_prototype(X_test[i,0:-1], Prots)
        predictedClasses[i] = Prots[cls[0],-1]
        
    return predictedClasses
    
predictedClasses = runTestSet(X_test, trimmedPrt)

plt.figure()
plt.title("Test set classification")
col={0:'bo',1:'go', 2:'co'}
for i in range(len(X_test[:,0])):
    plt.plot(X_test[i,0],X_test[i,1],col[int(predictedClasses[i])])

plt.plot(trimmedPrt[:,0],trimmedPrt[:,1],'ro')

plt.axis([0, 1.5, 0, 1.5])
plt.show()    

# Calculate accuracy
accuracy = sum(predictedClasses==X_test[:,-1])/len(X_test[:,0])
print("Accuracy(", len(trimmedPrt[:,0]), ")=", accuracy)

unoptimizedAccuracy = accuracy
# Try removing prototypes at random
# Remove 20% at every iteration until accuracy drops to < 95%
prots = trimmedPrt.copy()

trialsPerStep=50

reducedAccuracy=0
bestReducedPrt = trimmedPrt
while (len(prots[:,0]>=1)):
    print("-------------------------------------------------")        
    print("Trying ", trialsPerStep, "reductions of ", len(prots[:,0]), "prototypes...")
    print("reducedAccuracy=", reducedAccuracy)
    
    localBestAcc=0
    localBestPrt=bestReducedPrt
    for i in range(0, trialsPerStep):    

        rndNdx = (np.random.rand(len(prots[:,0])*0.2)*len(prots[:,0])).astype(int)
        #print(rndNdx)
        mask = np.ones(len(prots[:,0]),dtype=bool) #np.ones_like(a,dtype=bool)
        mask[rndNdx] = False
            
        reducedPrt = prots[mask,:]
        
#        print("mask len = ", len(mask))
#        print("mask true=", sum(mask))
#        print("reducedPrt len=", len(reducedPrt[:,0]))
#        print("prots len=", len(prots[:,0]))
        if (len(reducedPrt[:,0]) == len(prots[:,0])):
            break
            
        predictedClasses = runTestSet(X, reducedPrt)
        
#        print("predictedClasses len=", len(predictedClasses))
#        print("X_test len=", len(X_test[:,-1]))
#        print(predictedClasses==X_test[:,-1])
#        print(sum(predictedClasses==X_test[:,-1]))
#        print(reducedPrt)
        acc = sum(predictedClasses==X[:,-1])/len(X[:,0])
        print("trial acc=", acc)
        if (acc >= reducedAccuracy):
            localBestAcc=acc
            localBestPrt = reducedPrt.copy()    
    
    print("localBestAcc=", localBestAcc)
    print("localBestPrt=", localBestPrt)
    
    if (localBestAcc >= reducedAccuracy):
        reducedAccuracy = localBestAcc
        bestReducedPrt = localBestPrt.copy()
    else:
        break    

    prots = bestReducedPrt.copy()
    
    print("Accuracy with ", len(prots[:,0]), "prototypes=", reducedAccuracy)
#    if (len(prots[:,0])<=4):
#        break

prots = bestReducedPrt.copy()        
print("Best params: ",len(prots[:,0]), " at accuracy=", reducedAccuracy)
print(prots)

predictedClasses=runTestSet(X, prots)

plt.figure()
plt.title("Test set classification(optimized prototypes)")
col={0:'bo',1:'go', 2:'co'}
for i in range(len(X[:,0])):
    plt.plot(X[i,0],X[i,1],col[int(predictedClasses[i])])

plt.plot(prots[:,0],prots[:,1],'ro')

plt.axis([0, 3, 0, 3])
plt.show()    
