
# coding: utf-8

# In[ ]:





# # LVQ implementation in  python 
# 

# We first import two useful libraries
# 1. numpy (matrix algebra):  we use np as a shortcut
# 2. plyplot from matplotlib: useful for plotting charts: we use plt as a shortcut
# 3. use tab and shift+tab for help

# In[1]:


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from six.moves import range
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
C_g = np.random.rand(M_g,2)*.6+0.2
print(C_g)


# In[4]:


# we can also fix the centroids on a diagonal
#C_g=np.array([[.25,.25],
#             [.5,.5],
#             [.75,.75]])


# In[5]:


# we can also choose the centroids arbitarily
C_g=np.array([[.25,.25],
             [.75,.5],
             [.45,.75]])


# In[6]:


# Generate data set ( M=3, centroid, constant sigma)
#
sigma=0.095
number=50
#
# storing the centroid index (note this may not correspond to teh same number from the k-means algorithm)
X11=np.concatenate((sigma*np.random.randn(number,2)+C_g[0],np.full((number,1),0.0)),axis=1)
X22=np.concatenate((sigma*np.random.randn(number,2)+C_g[1],np.full((number,1),1.0)),axis=1)
X33=np.concatenate((sigma*np.random.randn(number,2)+C_g[2],np.full((number,1),2.0)),axis=1)
#
#X=np.concatenate((X1,X2,X3), axis=0)
X=np.concatenate((X11,X22,X33), axis=0)
np.random.shuffle(X)
#print X


# In[7]:


# plot data set and centroids
plt.figure()
col={0:'bo',1:'go', 2:'co'}
for i in range(len(X[:,0])):
    plt.plot(X[i,0],X[i,1],col[int(X[i,2])])

plt.plot(C_g[:,0],C_g[:,1],'ro')
plt.axis([0, 1.0, 0, 1.0])
plt.show()


# In[8]:


#split data set into train and test
split = int((number*M_g)*0.7)
print("Split point = ",split)
X_train=np.asarray(X[0:split,:])
print("Train size = ",len(X_train[:,0]))
X_test=np.asarray(X[split:,:])
print("Test size =",len(X_test[:,0]))


# ## LVQ1 starts here

# In[9]:


# Initialise prototypes (features,label) m
# Initialise a learning rate profile
# for each example in the training set do:
#    find the prototype closest to the training example
#    if the prototype label matches the example label:
#        m_i(t+1) = m_i(t) + rate(x(t)-m_i(t))
#    else:
#        m_i(t+1) = m_i(t) + rate(x(t)-m_i(t))




# In[10]:


# Initialise prototypes (features,label) m
#
# Prt is the array for prototypes
#
#np.random.seed(5)
# compute range of values for protypes
min_x0=np.min(X[:,0])
max_x0=np.max(X[:,0])
#print "x0_range = (%5.4f, %5.4f)" %(min_x0, max_x0)
min_x1=np.min(X[:,1])
max_x1=np.max(X[:,1])
#print "x1_range = (%5.4f, %5.4f)" %(min_x1, max_x1)
#a = min(min_x0,min_x1)
#b = max(max_x0,max_x1)
#print "x_range = (%5.4f, %5.4f)" %(a,b)
#
M = 3    # number of prototypes
P_0 = np.random.rand(M,1)*(max_x0-min_x0)+min_x0
P_1 = np.random.rand(M,1)*(max_x1-min_x1)+min_x1
P_label=np.array([[0],[1],[2]])
#
Prt=np.zeros((M,3),dtype=float)
Prt=np.concatenate((P_0,P_1,P_label),axis=1)
print("Initial Prototypes : ")
print(Prt)


# In[11]:


# Initialise a learning rate profile
T=20  # number of epochs
t=np.arange(T)
rate=0.02*np.exp(-0.1*t)
plt.figure()
plt.plot(t,rate)
plt.show()


# In[12]:


# function get nearest prototype
#
def get_nearest_prototype(features,prototype):
    K=len(prototype[:,0])
    F=np.full((K,2),features)
    diff=F-prototype[:,0:-1]
    dist=np.sqrt(diff[:,0]**2+diff[:,1]**2)
#    print(diff)
#    print(dist)
    return dist,dist.argsort()


# In[13]:


# for each example in the training set do:
#    find the prototype closest to the training example
#    if the prototype label matches the example label:
#        m_i(t+1) = m_i(t) + rate(x(t)-m_i(t))
#    else:
#        m_i(t+1) = m_i(t) + rate(x(t)-m_i(t))

print(Prt)
#print(X_train.shape)
t=0

for t in range(0,T):
	for feature in X_train[:,0:3]:
#		print(feature)
	#	print(get_nearest_prototype(feature, Prt))    
		(dist, cls) = lu.get_nearest_prototype(feature[0:-1], Prt)
#		print("dist=", dist, ", cls=", cls)
#		print(Prt[cls[0],2])
		prt = Prt[cls[0],0:-1]
		if (Prt[cls[0],2]==feature[2]):	
#			print("Match ->", Prt[cls[0],:], "=", feature)
			newPrt = prt - rate[t] * (feature[0:-1]-prt)
			Prt[cls[0],0:-1] = 	newPrt
#			print("Match: Old Prt=", prt, ", new Prt=", Prt[cls[0],0:-1], ", rate=", rate[t])
		else:
			newPrt = prt + rate[t] * (feature[0:-1]-prt)						
			Prt[cls[0],0:-1] = 	newPrt			
#			print("MisMatch: Old Prt=", prt, ", new Prt=", Prt[cls[0],0:-1], ", rate=", rate[t])			
#			print("No Match ->", Prt[cls[0],:], "=", feature)

# plot data set and centroids
	plt.figure()
	col={0:'bo',1:'go', 2:'co'}
			
	plt.plot(X_train[:,0],X_train[:,1],"bo")				
	plt.plot(Prt[:,0], Prt[:,1], "ro")	
	plt.axis([0, 1.0, 0, 1.0])
	plt.show()

print(Prt)
print(C_g)
# In[ ]:


# plot data set and centroids
# plot data set and centroids
plt.figure()
col={0:'bo',1:'go', 2:'co'}
for i in range(len(X_test[:,0])):
	(dist, cls) = get_nearest_prototype(X_test[i,0:-1], Prt)
		
	plt.plot(X_test[i,0],X_test[i,1],col[int(Prt[cls[0],2])])

plt.plot(C_g[:,0],C_g[:,1],'ro')
plt.axis([0, 1.0, 0, 1.0])
plt.show()



