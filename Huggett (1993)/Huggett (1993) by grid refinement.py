
# # The risk-free rate in heterogeneous-agent incomplete-insurance economies
# - Huggett, 1993, JEDC
# - Infinite horizon model
# - Updated on 23.09.29 by MJ Son
#
# [Steps]
# 1. q is the price of next-period credit balances
# 2. Make a q_grid. Do value function iteration for each q. Check the General Equilibrium Condition (steady state distribution, market clear)
# 3. Fix at the value of q which satisfies the general equilibrium condition
# 4. Calculate policy function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import quantecon as qe
# import ray

# ## Parameter

# +
risk_aver = 1.5
risk_aver_vec = [1.5, 3.0]

amax = 10 # not clear exactly what value Huggett used, I set this value based on axes of Figure1
amin = -8 # borrowing limit
amin_vec = [-2, -4, -6, -8]
beta = 0.99322 # discount factor

eh = 1.0 # high endowment
el = 0.1 # low endowment
pieheh = 0.925 # probability of eh given eh
piehel= 0.5 # probability of eh given el
egrid = [el, eh]
ne = 2

### Computation
max_iter = 3000
tol_iter = 1e-2


### Grid
# In the paper, "between 150 and 350 evenly spaced gridpoints are used on the set A"
# "The gridsize is between 0.03 and 0.1 units of credit balances"
na = 200
naf = na * 100

# asset
agrid = np.linspace(amin, amax, na)
afgrid = np.linspace(amin, amax, naf)

# q: price (reciprocal of (intereset rate+1))
Display=1
# -

# ## Utility function

u = lambda c: (c**(1-risk_aver))/(1-risk_aver)

# ## Stationary distribution of Markov chain

# +
## Stationary Distribution of Markov Chain by using qe module

# 1. State description
state_values = ["0.1", "1.0"]

# 2. Transition Probabilities
P = np.array([[1-piehel, piehel],[1-pieheh, pieheh]])

# 3. Initial distribution
x0 = np.array([1,0]) # population measure = 1

# 4. Stationary Distribution
mc = qe.markov.MarkovChain(P, state_values)
stat_dist = mc.stationary_distributions
print(stat_dist)


# -

# ## VFI function
# - VFI calculation finds the individual’s optimal decisions at a given q.
# - By simulating, can get aggregate values(distribution, agg.consumption, agg. saving) for a given q
#

# @ray.remote
def vfi(q, Vguess = np.ones((na, ne))):
    Vdiff_list =[] # Vdiff 확인 위해 임시

    V = Vguess.copy()

    Vdiff = 1
    Iter = 0

    while Vdiff > tol_iter:
        Iter = Iter +1
        Vlast = V.copy()
        
        # Use interpolation to match Vlast to afgrid
        temp1 = interp1d(agrid, Vlast[:,0], kind = 'linear')(afgrid)
        temp2 = interp1d(agrid, Vlast[:,1], kind = 'linear')(afgrid)
        Vlast_interp = np.array([temp1, temp2]).T
        
        # pre-allocation
        V = np.zeros((na, ne))
        sav = np.zeros((na, ne))
        savind = np.zeros((na, ne), dtype = int) # index
        con = np.zeros((na, ne))
        
        # Loop over assets
        for ia in range(0, na):
            
            # Loop over endowments
            for ie in range(0,ne):
                income = agrid[ia] + egrid[ie] 
                 # the next period probability varies depending on whether the current endowment is el or eh
                
                Vchoice = u(np.maximum(income - q*afgrid.T, 1.0e-10)) + beta*(Vlast_interp @ P.T)[:,ie]
                V[ia, ie] = np.max(Vchoice)
                savind[ia, ie] = np.argmax(Vchoice)
                sav[ia, ie] = afgrid[savind[ia, ie]]
                con[ia, ie] = income - q*sav[ia, ie]

        # check the convergence of Value function        
        Vdiff = np.max(np.max(abs(V-Vlast)))
        Vdiff_list.append(Vdiff)
        if (Display == 1) & (Iter%100 == 0): # Notifies you every time it exceeds 100
            print('Iteration no. ' + str(Iter), ' max val fn diff is ' + str(Vdiff))     
            
    return V, Vlast, Vdiff_list, savind, sav, con


# ## Simulation function

# @ray.remote
def simulate(Nsim = 50000, Tsim = 500):
    
    # saving interpolation: Since there is no mapping from afgrid to afgrid, I create one.
    tem1 = interp1d(agrid, sav[:,0], kind='linear')(afgrid)
    tem2 = interp1d(agrid, sav[:,1], kind='linear')(afgrid)
    sav_interp = np.array([tem1, tem2]).T

    t1 = interp1d(agrid, savind[:,0], kind='linear')(afgrid)
    t2 = interp1d(agrid, savind[:,1], kind='linear')(afgrid)
    savind_interp = np.array([t1, t2]).T
    savind_interp = np.around(savind_interp).astype('int')

    # pre-allocation
    esim = np.zeros((Nsim, Tsim)) 
    aindsim = np.zeros((Nsim, Tsim), dtype = int) # ind = index

    # Initial asset allocation
    aindsim[:,0] = 0

    # Endowment allocation (Exogenous) 
    for i in range(0, Nsim):
        esim[i,:] = mc.simulate(Tsim, init="0.1").astype('float')


    ### Loop over time periods

    for it in range(0, Tsim): # Tsim
        if Display == 1 and (it+1)%100 == 0: 
            print(' Simulating, time period ' + str(it+1))

        # asset choice
        if it < Tsim-1:
            for ie in range(0, ne):
                e = egrid[ie]
                aindsim[esim[:,it] == e, it+1] = savind_interp[aindsim[esim[:,it] == e,it], ie] 

    asim = afgrid[aindsim]
    agg_asset = np.mean(asim, axis=0)[-1] 
    
    return asim, esim, agg_asset

# ### To find the equilibrium q: grid refinement
#
# - With a given q grid, calculate an aggregate asset list.
# - Find the point where the sign of the aggregate asset changes (-,+).
# - Refine the grid to that range. Repeat the process again and again.
# - The aggregate asset converges to 0. We can find the value of an equilibrium q.

# +
nq = 10
adjust = 0.05

tol = 1e-1
agg_asset = 1
Display = 0
# q 
qmin = 0.99
qmax = 1.03
gridnum = 0

while abs(agg_asset) > tol:
    agg_asset_list = []
    qgrid = np.linspace(qmin, qmax, nq)
    print('------------------------------------')
    print(f'{gridnum}th Grid Updated {[qmin, qmax]}')
    
    iternum = 0
    for q in qgrid:
        # Value function iteration 
#         ray.shutdown()
#         ray.init(num_cpus=10)
        V, Vlast, Vdiff_list, savind, sav, con = vfi(q) # ray.get(vfi.remote(q))
#         ray.shutdown()

        # Simulation
#         ray.init(num_cpus=10)
        asim, esim, agg_asset = simulate() #ray.get(simulate.remote())
        
        agg_asset_list.append(agg_asset)
        print(f'iternum = {iternum}, Running on q = {q}, agg_asset = {agg_asset}')

        if abs(agg_asset) <= tol:
            print('You got it !!!')
            break 
        
        if iternum > 0 and agg_asset_list[iternum-1]*agg_asset_list[iternum] < 0:
            break
            
        iternum += 1
            
    # update grid
    idxmin = np.abs(agg_asset_list).argmin()
    print(f'min_diff = {np.abs(agg_asset_list).min()}')
    agg_asset_list = np.array(agg_asset_list)
    if all(agg_asset_list < 0):
        qmin = qgrid[-1]
        qmax = qgrid[-1]*(1+adjust)

    elif all(agg_asset_list > 0):
        qmin = qgrid[0]*(1-adjust)
        qmax = qgrid[0]
    
    elif idxmin > 0 and agg_asset_list[idxmin]*agg_asset_list[idxmin-1] < 0:
        qmin = qgrid[idxmin-1]
        qmax = qgrid[idxmin]

    elif agg_asset_list[idxmin]*agg_asset_list[idxmin+1] < 0:
        qmin = qgrid[idxmin]
        qmax = qgrid[idxmin+1]    
    

    gridnum += 1     
   

# -

# ## optimal choice at Equilibrium q

equil_q = 0.9941152263374485
V, Vlast, Vdiff_list, savind, sav, con = vfi(equil_q) 
asim, esim, agg_asset = simulate() 

agg_asset

# Convergence of Value Function
plt.plot(Vdiff_list)

# Optimal Decision
fig, ax = plt.subplots()
plt.plot(agrid, sav[:,0], label = 'unemploymed')
plt.plot(agrid, sav[:,1], label = 'employed')
plt.plot(agrid, agrid, label = '$45\,^{\circ}$')
plt.title('Optimal Saving Decision')
plt.xlabel('a')
plt.ylabel('a\'')
fig.legend()

# Consumption Policy Function
fig, ax = plt.subplots()
plt.plot(agrid, con[:,0], label = 'Unemployed')
plt.plot(agrid, con[:,1], label = 'Employed')
plt.title('Consumption Policy Function')
plt.xlabel('a')
plt.ylabel('Consumption')
fig.legend()

# Endowment distribution
Tsim = 500
plt.hist(esim[:,Tsim-1],bins = len(egrid), edgecolor='black')
plt.ylabel('population')
plt.title('Endowment distribution')

## convergence check
# asset market clear check 
plt.plot(range(0,Tsim),np.mean(asim,axis = 0),'k-',linewidth=1.5)
plt.xlabel('Time Period')
plt.title('Mean Asset Convergence')
plt.show()

# # asset distribution
plt.hist(asim[:,-1],bins = 40,facecolor=(.7,.7,.7),edgecolor='black')
plt.xlabel('asset level')
plt.ylabel('population')
plt.title('Asset distribution')

# ## Lorenz curve and Gini coefficient

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

asset_level = np.unique(asim[:,-1], return_counts = True)[0]

pop_dist = np.unique(asim[:,-1], return_counts = True)[1]

asset_level += abs(amin)

lorenz_and_gini(pop_dist, asset_level)


