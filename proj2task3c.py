import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
np.random.seed(sum([ord(c) for c  in 'abiseria']))

X = [[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]]
Mu = [[6.2,3.2],[6.6,3.7],[6.5,3.0]]
pi1 = 1/3
pi2 = 1/3
pi3 = 1/3
sigma = [[0.5,0],[0,0.5]]



k = multivariate_normal.pdf(X,Mu[0],sigma)
l = multivariate_normal.pdf(X,Mu[1],sigma)
m = multivariate_normal.pdf(X,Mu[2],sigma)
k1 = k*pi1
l1 = l*pi2
m1 = m*pi3
print(k1[0])
print(l1[0])
print(m1[0])
cluster = []
for i in range(3):
	cluster.append([])
for i in range(10):
	r1 = k1[i]/(k1[i]+l1[i]+m1[i]) 
	r2 = l1[i]/(k1[i]+l1[i]+m1[i])
	r3 = m1[i]/(k1[i]+l1[i]+m1[i])
	if(r1>r2):
		if(r1>r3):
			cluster[0].append([i,1,r1])
	if(r2>r1):
		if(r2>r3):
			cluster[1].append([i,2,r2])
	if(r3>r1):
		if(r3>r2):
			cluster[2].append([i,3,r3])
			
print(cluster)

m1,m2,m3=0,0,0
for i in range(len(cluster[0])):
	m1 = m1+cluster[0][i][2]
for i in range(len(cluster[1])):
	m2 = m2+cluster[1][i][2]
for i in range(len(cluster[2])):
	m3 = m3+cluster[2][i][2]
	
pi11 = m1/len(cluster[1])
pi22 = m2/len(cluster[1])
pi33 = m3/len(cluster[1])

mu11,mu22,mu33 = [0,0],[0,0],[0,0]

for i in range(len(cluster[0])):
	mu11 = mu11 + np.dot(cluster[0][i][2],X[cluster[0][i][0]])
	
for i in range(len(cluster[1])):
	mu22 = mu22 + np.dot(cluster[1][i][2],X[cluster[1][i][0]])
	
for i in range(len(cluster[2])):
	mu33 = mu33 + np.dot(cluster[2][i][2],X[cluster[2][i][0]])

mu11 = mu11/m1
mu22 = mu22/m2
mu33 = mu33/m3	
	
print('pi1 = ', pi11)
print('pi2 = ', pi22)
print('pi3 = ', pi33)
print('mu1 = ', mu11)
print('mu2 = ', mu22)
print('mu3 = ', mu33)
	
temp = [0,0]
sigma1 =  [0,0]
sigma2 = [0,0]
sigma3 = [0,0]
for i in range(len(cluster[0])):
	temp = temp + cluster[0][i][2]*np.transpose(X[cluster[0][i][0]] - mu11)*(X[cluster[0][i][0]] - mu11)
sigma1 = temp/m1
temp = [0,0]

for i in range(len(cluster[1])):
	temp = temp + cluster[1][i][2]*np.transpose(X[cluster[1][i][0]] - mu22)*(X[cluster[1][i][0]] - mu22)
sigma2 = temp/m2
temp = [0,0]

for i in range(len(cluster[2])):
	temp = temp + cluster[2][i][2]*np.transpose(X[cluster[2][i][0]] - mu33)*(X[cluster[2][i][0]] - mu33)
sigma3 = temp/m3

print('sigma1 = ', sigma1)
print('sigma2 = ', sigma2)
print('sigma3 = ', sigma3)
cov1 =[[sigma1[0],0],[0,sigma1[1]]]
cov1 =[[sigma2[0],0],[0,sigma2[1]]]
cov1 =[[sigma3[0],0],[0,sigma3[1]]]
print('sigma1 = ', sigma1)
print('sigma2 = ', sigma2)
print('sigma3 = ', sigma3)	

'''s1 =0
s2 = 0
s3 =0
for i in range(10):
	s1 = s1+k1
	s2 = s2+l1
	s3 = s3+m1
Mu1 = 0
Mu2 = 0
Mu3 = 0
for i in range(10):
	Mu1 =  Mu1 + k1[i]*X[i]
	Mu2 =  Mu2 + k2[i]*X[i]
	Mu3 =  Mu3 + k1[i]*X[i]
 	
final1 = Mu1/s1
final2 = Mu2/s2
final3 = Mu3/s3
 
pi11 = s1/10
pi22 = s2/10
pi33 = s3/10
 
print(final1)
print(final2)
print(final3)
print(pi11)
print(pi22)
print(pi33)
'''