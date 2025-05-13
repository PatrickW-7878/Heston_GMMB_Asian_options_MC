import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pricing_asian(s0, v0, r, kappa, theta, sigma, rho, nb_mc, nb_t, T):
    np.random.seed(42)
    dt = T/nb_t
    p1 = 0.0
    sqrt_dt = math.sqrt(dt)
    
    for n in range(0,nb_mc):
        ln_s  = math.log(s0)
        v = v0
        ispot = 0
        for i in range(0,nb_t):
            dw1 = np.random.normal(0,1)*sqrt_dt 
            dw2 = rho*dw1 + math.sqrt(1-rho*rho)*np.random.normal(0,1)*sqrt_dt 
            
            ispot = ispot + math.exp(ln_s)*dt 
            ln_s = ln_s + (r - v/2)*dt + math.sqrt(v)*dw1
            v = v + kappa*(theta-v)*dt + sigma * math.sqrt(v)*dw2
            
        p1 = p1 + np.max([0.0, (1.0/T)*ispot  - math.exp(ln_s) ])
    
    p1 = p1/nb_mc
    
    return p1

####################################################
####################################################
def pricing_heston(s0, v0, r, kappa, theta, sigma, rho, nb_mc, nb_t, T, k):
    np.random.seed(42)
    dt = T/nb_t
    p1 = 0.0
    sqrt_dt = math.sqrt(dt)
    k1 = int(nb_t/k)
    k2 = 1
    for n in range(0,nb_mc):
        ln_s  = math.log(s0)
        v = v0
        ispot = s0
        for i in range(1,nb_t+1):
            dw1 = np.random.normal(0,1)*sqrt_dt 
            dw2 = rho*dw1 + math.sqrt(1-rho*rho)*np.random.normal(0,1)*sqrt_dt 
            
            if( (i/k1)==int(i/k1)):
                k2 = k2 + 1
                ispot = ispot *math.exp(ln_s) 
            ln_s = ln_s + (r - v/2)*dt + math.sqrt(v)*dw1
            v = v + kappa*(theta-v)*dt + sigma * math.sqrt(v)*dw2
            
        p1 = p1 + np.max([0.0, ispot**(1/(k+1))  - math.exp(ln_s) ])
    
    p1 = p1/nb_mc
    
    return p1
####################################################
####################################################

s0 = 1           # initial stock value
v0 = 0.2*0.2     # initial volatility
r = 0.0
kappa = 3.5
theta = 0.2*0.2
sigma = 0.2
rho = -0.5
        
T = 1
nb_mc = 50000
nb_t = 250
k = 5
    
print('Pricing of a GMMB option in the Heston model.')
d2={}
for p in np.linspace(0.9,1.1,num=11):
    sigma2 = p * sigma
    p1 = pricing_heston(s0, v0, r, kappa, theta, sigma2, rho, nb_mc, nb_t, T, k)
    d2[p] = p1

df2 = pd.DataFrame.from_dict(data = d2, orient = 'index', columns=['price'])
df2.plot(y='price', ylabel='price', xlabel = r'$\lambda$', use_index = True, title = r'Sensitivity to $\sigma$ of the GMMB option price in the Heston model')            
plt.savefig("path_dependent_GMMB_option.pdf")

    
print('Pricing of an Asian option in the Heston model.')
d3={}
for p in np.linspace(0.9,1.1,num=11):
    sigma2 = p * sigma
    p1 = pricing_asian(s0, v0, r, kappa, theta, sigma2, rho, nb_mc, nb_t, T)
    d3[p] = p1


df3 = pd.DataFrame.from_dict(data = d3, orient = 'index', columns=['price'])
df3.plot(y='price', ylabel='price', xlabel = r'$\lambda$', use_index = True, title = r'Sensitivity to $\sigma$ of the Asian option price in the Heston model')            
plt.savefig('asian_option.pdf')
        