# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:12:54 2024

@author: thoma
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:35:47 2024

@author: tt386
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import time as clock
import random


"""
Good-looking seeds:
    
    
The case of a single zealot with a very similar opinion
F,z,N = 0.99, 0.001, 1000    
1,3

3 Looks best



The case of adding more zealots of the same weight, expected tendency to 1
F,z,N = 0.99, 0.01,1000    

1 is best


Up to 0.5 for 1 zealot:
F,z = 0.99, 0.005    



Intermediate case up to 0.5L
F=0.8,z=0.1

1 is best
"""

seed = 1

random.seed(seed)


starttime = clock.time()


F = 0.99#475
#z = 0.1#01

#F= 0.6
#z = 0.3

#Initial population:
N0 = 1000

#Added zealots
Z0 = 1

#Total effective population
N = N0 + Z0


z = Z0/N


#Initial population of individuals in the group
#N = 1000


Z = Z0#int(z*N)

ZS = 1

time = int(1e6)#2e8)

#The number of resistant
n = min(int((z/(1-F) - z) * N),int(N-N*z-1))


#histlist = []

hist = np.zeros(N)


IGNORESTATIONARY = False

def P_Up(n,z,F):
    return (1-n-z-zs)/(1-z-zs) * F*(n+z) / (1-n-z+F*(n+z))


def P_Down(n,z,F):
    return n/(1-z-zs) * (1-n-z) / (1-n-z+F*(n+z))


#########################################
###Simulation Of Complete################
#########################################


for t in range(time):
    if t%(time/10) == 0:
        print("At time",t,"n=",n)
        #print(t)
    randval = random.uniform(0,1)
    
    upprob = (N-Z-ZS-n)/(N-Z-ZS) * F*(n+Z) / (N-Z-n + F*(n+Z))
    
    
    downprob = n/(N-Z-ZS) * (N-n-Z) / (N-n-Z+F*(n+Z))
    """
    upprob = (N-Z-n)/(N-Z) * F*(n+Z) / (N-Z-n-1 + F*(n+Z))
    
    downprob = n/(N-Z) * (N-Z-n) / (N-Z-n + F*(n-1+Z))
    """
    
    tot = upprob+downprob

    if tot==0:
        tot=1
    
    
    if not IGNORESTATIONARY:
        #Use this if allow stationary
        if randval < upprob:
            n += 1
            
        elif randval < upprob + downprob:
            n -= 1
    
    else:
        #Use this if we want to ignore stationary
        if randval < upprob/tot:
            n +=1
            
        else:
            n -= 1
    
  
    
    if n >= N-Z:
        n -=1
    
    
    #if t> 0.9*time:
    #histlist.append((n+Z)/N)
    hist[int(n + Z)] += 1
        
zs =0# 1/N 

"""
def P_Up(n,z,F):
    return (1-n-z-zs)/(1-z-zs) * F*(n+z) / (1-n-z+F*(n+z))


def P_Down(n,z,F):
    return n/(1-z-zs) * (1-n-z) / (1-n-z+F*(n+z))
"""



#########################################
###Analytical Of Complete################
#########################################


def alpha(n,z,F):
    
    #return (1-z-zs-n)/(1-z-zs) * (F*(n+z)-n)/ (1-z-n+F*(n+z))

    return P_Up(n,z,F) - P_Down(n,z,F)

def beta(n,z,F):
    
    #return (1/N)*(1-z-zs-n)/(1-z) * (F*(n+z)+n)/ (1-z-n+F*(n+z))
    return (P_Up(n,z,F) + P_Down(n,z,F))/N

        
PList = []
epsilon = zs#0.0000001
nlist = np.linspace(0,1-z -epsilon,N)

"""
plt.figure()
#plt.semilogy(nlist,alpha(nlist, z, F))
#plt.plot(nlist,beta(nlist, z, F))
plt.plot(nlist,alpha(nlist, z, F)/beta(nlist, z, F))

plt.show()
"""

for n in nlist:

    integral = integrate.quad(lambda n: alpha(n,z,F)/beta(n,z,F),0,n)[0]

    if not IGNORESTATIONARY:
        #P = np.exp(2 * integral )/ beta(n,z,F) #IGNORE THE DIVISION IF USING THE NO STATIONARY STEP VERSION
        
        P = np.exp(2*integral - np.log(beta(n,z,F)))
        
    else: 
        P = np.exp(2 * integral )

    #print(n,P)

    PList.append( P)


PList = np.asarray(PList)
Normalisation = integrate.simpson(PList,nlist)
print(Normalisation)
PList = PList/Normalisation


print("Area under curve:",integrate.simpson(PList,nlist))


print("Mean=",integrate.simpson(PList*(nlist+z),nlist))
print("Deterministic=",z/(1-F))

#Analytical
n = nlist

"""
#Weirdly, ignoring the log(beta) part leads to a highly accurate result
logAna = (4*N*z*F/(1+F)**2) * np.log(n * (F+1)/(z*F) + 1) + -2*N*(1-F**2)*n/(1+F)**2 - np.log(beta(n,z,F))

if IGNORESTATIONARY:
    
    logAna = -2*N*(1-F**2)*n/(1+F)**2 + (4*N*z*F/(1+F)**2) * np.log(n * (F+1)/(z*F) + 1)#- np.log(beta(n,z,F))
Ana = np.exp(logAna)#(n * (F+1)/(z*F) + 1)**(4*N*z*F/(1+F)**2) * np.exp(-2*N*(1-F**2)*n/(1+F)**2)/beta(n,z,F)
Ana /= integrate.simpson(Ana,nlist)
"""






maxval = min(z/(1-F),1)
        
bins= np.arange(N+1)/N-0.5/N 




fig=plt.figure(2)
ax = fig.add_subplot(1,1,1)
#plt.plot(nlist+z,PList,'--k',linewidth = 2)

plt.plot(((nlist+z) * (N0+Z0)-Z0)/N0,PList,'--k',linewidth = 2)

plt.plot([maxval,maxval],[0,max(PList)],'--k',linewidth=0.5)

plt.plot((np.linspace(0,1,N) * (N0 + Z0) - Z0)/N0,hist/sum(hist)*N)



plt.figure(2).set_size_inches(9.00000/2.54, 6.000000/2.54, forward=True)
plt.figure(2).axes[0].set_position([0.1, 0.1, 0.85, 0.85])

ax.set_xticks([0,0.25,0.5,0.75,1])
ax.set_xticklabels(
    ['$0.00$',r'$0.25$',r'$0.50$',r'$0.75$',r'$1.00$'],
    fontsize=7)

ax.set_yticks([0,2,4,6,8,10])
ax.set_yticklabels(
    [r'$0$','$2$',r'$4$',r'$6$',r'$8$',r'$10$'],
    fontsize=7)

ax.set_ylabel(r'probability density',fontsize=8)
ax.set_xlabel(r'proportion of weak opinion $n+z$',fontsize=8)

ax.tick_params(axis='both', which='major', labelsize=20)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.tight_layout()


plt.xlim(0,1)

#plt.title("z %0.3f, F %0.3f, time "%(z,F) + "{:.1e}".format(time) + " ignore_stationary " + str(IGNORESTATIONARY))
#plt.show()

plt.savefig("fig.png")



endtime = clock.time()


print("Time taken:",endtime-starttime)
