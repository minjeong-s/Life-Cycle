# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# - checked on 231007 MJ SON

import pandas as pd
import numpy as np
import quantecon as qe
from scipy.stats import norm
from scipy.interpolate import interp1d
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
# import ray
import time

# ## Parameter

# +
beta = 0.96
alpha = 0.36
delta = 0.08

risk_aver_vec = [1,3,5]
rho_vec = [0.0, 0.3, 0.6, 0.9]
sigma_vec = [0.2, 0.4]

risk_aver = 3
rho = 0.6
sigma = 0.2

amin = 0.0
amax = 20
na = 150
naf = na*100
ne = 7
agrid = np.linspace(amin, amax, na)
afgrid = np.linspace(amin, amax, naf)

tol_iter = 1e-3

interp_kind = 'linear'
# The string has to be one of 
#‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
# -

# Markov chain
mc = qe.markov.approximation.tauchen(rho = rho, sigma_u = sigma*((1-rho**2)**(1/2)) , n=ne)
P = mc.P
stationary_dist = mc.stationary_distributions
stationary_dist

state_values = mc.state_values
state_values

labor_endowment = np.exp(mc.state_values)
labor_endowment

egrid = labor_endowment


# ## Function

def u(c):
    if risk_aver == 1:
        u = np.log(c)
    else:
        u = (c**(1-risk_aver)-1)/(1-risk_aver) 
    
    return u


# +
def wage(k,l=1):
    wage = (1-alpha) * k**alpha * l**(-alpha)
    return wage

def interestrate(k, l=1):
    interest = alpha * k**(alpha-1) * l**(1-alpha) - delta
    return interest

def f(k,l):
    return k**alpha * l**(1-alpha)


# -

# @ray.remote
def vfi(r,w, Vguess = np.ones((na, ne))):
    Vdiff_list =[] # Vdiff 확인 위해 임시

    V = Vguess.copy()

    Vdiff = 1
    Iter = 0

    while Vdiff > tol_iter:
        Iter = Iter +1
        Vlast = V.copy()
        
        # Vlast를 afgrid에 맞춰주기 위해서 보간법 사용
        temp1 = interp1d(agrid, Vlast[:,0], kind = interp_kind)(afgrid)
        temp2 = interp1d(agrid, Vlast[:,1], kind = interp_kind)(afgrid)
        temp3 = interp1d(agrid, Vlast[:,2], kind = interp_kind)(afgrid)
        temp4 = interp1d(agrid, Vlast[:,3], kind = interp_kind)(afgrid)
        temp5 = interp1d(agrid, Vlast[:,4], kind = interp_kind)(afgrid)
        temp6 = interp1d(agrid, Vlast[:,5], kind = interp_kind)(afgrid)
        temp7 = interp1d(agrid, Vlast[:,6], kind = interp_kind)(afgrid)

        Vlast_interp = np.array([temp1, temp2, temp3, temp4, temp5, temp6, temp7]).T
        
        # pre-allocation
        V = np.zeros((na, ne))
        sav = np.zeros((na, ne))
        savind = np.zeros((na, ne), dtype = int)
        con = np.zeros((na, ne))
        
        # Loop over assets
        for ia in range(0, na):
            
            # Loop over endowments
            for ie in range(0,ne):
                income = (1+r)*agrid[ia] + w*labor_endowment[ie] 
                # 현재 endowment가 el인지 eh인지에 따라서 다음기 확률이 달라지므로
                
                Vchoice = u(np.maximum(income - afgrid.T, 1.0e-10)) + beta*(Vlast_interp @ P.T)[:,ie]
                V[ia, ie] = np.max(Vchoice)
                savind[ia, ie] = np.argmax(Vchoice)
                sav[ia, ie] = afgrid[savind[ia, ie]]
                con[ia, ie] = income - sav[ia, ie]

        # Value function의 값이 수렴하는지 확인        
        Vdiff = np.max(np.max(abs(V-Vlast)))
        Vdiff_list.append(Vdiff)
        if Display == 1 :
            print('Iteration no. ' + str(Iter), ' max val fn diff is ' + str(Vdiff))     
            
    return V, Vlast, Vdiff_list, savind, sav, con


# @ray.remote
def simulate(Nsim = 50000, Tsim = 500):
    
    # saving interpolation: afgrid에서 afgrid로 매핑하는 건 없기 때문에 만들어주려고
    tem1 = interp1d(agrid, sav[:,0], kind=interp_kind)(afgrid)
    tem2 = interp1d(agrid, sav[:,1], kind=interp_kind)(afgrid)
    tem3 = interp1d(agrid, sav[:,2], kind=interp_kind)(afgrid)
    tem4 = interp1d(agrid, sav[:,3], kind=interp_kind)(afgrid)
    tem5 = interp1d(agrid, sav[:,4], kind=interp_kind)(afgrid)
    tem6 = interp1d(agrid, sav[:,5], kind=interp_kind)(afgrid)
    tem7 = interp1d(agrid, sav[:,6], kind=interp_kind)(afgrid)

    sav_interp = np.array([tem1, tem2, tem3, tem4, tem5, tem6, tem7]).T

    t1 = interp1d(agrid, savind[:,0], kind=interp_kind)(afgrid)
    t2 = interp1d(agrid, savind[:,1], kind=interp_kind)(afgrid)
    t3 = interp1d(agrid, savind[:,2], kind=interp_kind)(afgrid)
    t4 = interp1d(agrid, savind[:,3], kind=interp_kind)(afgrid)
    t5 = interp1d(agrid, savind[:,4], kind=interp_kind)(afgrid)
    t6 = interp1d(agrid, savind[:,5], kind=interp_kind)(afgrid)
    t7 = interp1d(agrid, savind[:,6], kind=interp_kind)(afgrid)

    savind_interp = np.array([t1, t2, t3, t4, t5, t6, t7]).T
    savind_interp = np.around(savind_interp).astype('int')
    savind_interp[savind_interp >= naf] = naf-1 # 인덱스 반올림해서 인덱스 범위 벗어난 경우를 방지하고자 
    
    # pre-allocation
    esim = np.zeros((Nsim, Tsim)) 
    aindsim = np.zeros((Nsim, Tsim), dtype = int) # ind = index

    # Initial asset allocation
    aindsim[:,0] = 0

    # Endowment allocation (Exogenous) 
    for i in range(0, Nsim):
        esim[i,:] = mc.simulate(Tsim).astype('float')
    
    esim = np.exp(esim)

    ### Loop over time periods

    for it in range(0, Tsim): # Tsim
        if Display == 1 & (it+1)%100 == 0: # time period 100 넘길 때마다 알려줌
            print(' Simulating, time period ' + str(it+1))

        # asset choice
        if it < Tsim-1:
            for ie in range(0, ne):
                e = egrid[ie]
                aindsim[esim[:,it] == e, it+1] = savind_interp[aindsim[esim[:,it] == e,it], ie] 

    asim = afgrid[aindsim]
    avg_asset = np.mean(asim[:,-1])
    
    return asim, esim, avg_asset


# +
import scipy.integrate as integrate

def lorenz_and_gini(dist_wealth, agrid, curve=True):
    dist_wealth_normalize = dist_wealth / (dist_wealth.sum())
    
    agg_asset = dist_wealth_normalize @ agrid
    
    wealth_percentile = []
    for i in range(len(agrid)+1):
        value = (dist_wealth_normalize[:i] @ agrid[:i]) / agg_asset
        wealth_percentile.append(value)
    
    if dist_wealth_normalize.cumsum()[0] < 1e-20 :
        pop_cumsum = dist_wealth_normalize.cumsum()
    else:
        pop_cumsum = np.insert(dist_wealth_normalize.cumsum(), 0, 0)
        
    if wealth_percentile[0] < 1e-20:
        wealth_cumsum = wealth_percentile
    else:
        wealth_cumsum = np.insert(wealth_percentile, 0, 0)
    
    if curve == True:
        plt.plot(pop_cumsum, wealth_cumsum)
        plt.plot(wealth_cumsum, wealth_cumsum)
        plt.title('Lorenz curve')
        plt.xlabel('population')
        plt.ylabel('wealth')
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.grid(True)
    else: 
        pass

    ### Gini coefficient
    lorenz = interp1d(pop_cumsum, wealth_cumsum)
    gini = (0.5 - integrate.quad(lorenz, 0, 1)[0])/0.5
    print(f'gini = {gini}')
    
    return gini


# -

# ## Equilibrium r 찾기

1/beta - 1

delta

# +
# assume delta = 0
Display = 0 
phi = 0.8 
iternum = 0
crit = 1
crit_list = []

r_old = 2
r = 0.03

while iternum < 1000 and crit > 1e-2:
    iternum += 1
    crit = abs(r - r_old)/r_old
    crit_list.append(crit)

    k = (r/alpha)**(1/(alpha-1)) # * l 인데 (l = 1)
    w = r*(1-alpha)*k / alpha # / (alpha*L)인데 l = 1

    print(f'iter_num = {iternum}, r = {r}, k = {k}, crit = {crit}')
    
    ## VFI
#     ray.shutdown()
#     ray.init(num_cpus = 8)    
    V, Vlast, Vdiff_list, savind, sav, con =  vfi(r,w) # ray.get(vfi.remote(r))
#     ray.shutdown()
    
    ## Simulation
#     ray.init(num_cpus=8)
    asim, esim, avg_asset =  simulate() #ray.get(simulate.remote())

    k_old = k
    knew = avg_asset
    
    k = k_old*phi + knew*(1-phi)
    
    rnew = alpha* k**(alpha-1) # *L**(1-alpha)인데 l = 1
    
    r_old = r
    
    r = r_old*phi + rnew*(1-phi)
    
    V_guess = V # 이전 r 값에서 수렴시킨 value function 값을 guess로 사용하면 속도가 더 빨라질 수 있으니까

# -



# ## Plot

# Convergence of Value function
plt.plot(Vdiff_list)

## Optimal Decision (saving = capital accumulation)
fig, ax = plt.subplots()
plt.plot(agrid, sav[:,0], label = 'lowest endowment')
plt.plot(agrid, sav[:,6], label = 'highest endowment')
plt.plot(agrid, agrid, label = '$45\,^{\circ}$')
plt.title('Optimal Saving Decision')
plt.xlabel('a')
plt.ylabel('a\'')
fig.legend()

# Consumption Policy Function
fig, ax = plt.subplots()
plt.plot(agrid, con[:,0], label = 'lowest endowment')
plt.plot(agrid, con[:,6], label = 'highest endowment')
plt.title('Consumption Policy Function')
plt.xlabel('a')
plt.ylabel('Consumption')
fig.legend()

[0.54881164, 0.67032005, 0.81873075, 1. , 1.22140276, 1.4918247 , 1.8221188 ]

stationary_dist

# endowment distribution
Tsim = 500
plt.hist(esim[:,Tsim-1] ,bins = len(egrid), edgecolor='black')
plt.ylabel('population')
plt.title('Labor endowment distribution')

# asset distribution
plt.hist(asim[:,-1],bins = 40,facecolor=(.7,.7,.7),edgecolor='black')
plt.xlabel('asset level')
plt.ylabel('population')
plt.title('Asset distribution')

asset_level = np.unique(asim[:,-1], return_counts = True)[0]

pop_dist = np.unique(asim[:,-1], return_counts = True)[1]

lorenz_and_gini(pop_dist, asset_level)


