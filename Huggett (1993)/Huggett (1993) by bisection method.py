
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
sigma_vec = [1.5, 3.0]
sigma = sigma_vec[0]

amax = 12 # not clear exactly what value Huggett used, I set this value based on axes of Figure1
amin_vec = [-2, -4, -6, -8]
amin = amin_vec[-1] # borrowing limit

beta = 0.99322 # discount factor

eh = 1.0 # high endowment
el = 0.1 # low endowment
pi_hh = 0.925 # probability of eh given eh
pi_ll= 0.5 # probability of eh given el
egrid = [el, eh]
nb_e = len(egrid)

# +
### Computation
max_iter = 3000
tol_iter = 1e-2


### Grid
# In the paper, "between 150 and 350 evenly spaced gridpoints are used on the set A"
# "The gridsize is between 0.03 and 0.1 units of credit balances"
nb_agrid = 200
nb_afgrid = nb_agrid * 100

# asset
agrid = np.linspace(amin, amax, nb_agrid)
afgrid = np.linspace(amin, amax, nb_afgrid)

# q: price (reciprocal of (intereset rate+1))
Display=1
# -

# ## Utility function

u = lambda c: (c**(1-sigma))/(1-sigma)

# ## Stationary distribution of Markov chain

# +
## Stationary Distribution of Markov Chain by using qe module

# 1. State description
state_values = ["0.1", "1.0"]

# 2. Transition Probabilities
P = np.array([[1-pi_ll, pi_ll],[1-pi_hh, pi_hh]])

# 3. Initial distribution
x0 = np.array([1,0]) # population measure = 1

# 4. Stationary Distribution
mc = qe.markov.MarkovChain(P, state_values)
stat_dist = mc.stationary_distributions
print(stat_dist)
# -

income_temp = np.zeros(((nb_agrid, nb_e)))
income_temp[:,0] = np.maximum(agrid + egrid[0],1e-100)
income_temp[:,1] = np.maximum(agrid + egrid[1],1e-100)

V = u(income_temp) # will be used as an initial guess of V 

# +
income = agrid[1] + egrid[1] 

q = 1/beta*1.002
temp1 = interp1d(agrid, V[:,0], kind = 'cubic')(afgrid)
temp2 = interp1d(agrid, V[:,1], kind = 'cubic')(afgrid)
V_f = np.array([temp1, temp2]).T

con_tem = np.maximum(income - q*afgrid, 1E-100)

u(con_tem)+beta*(V_f @ P[1,:].T)

# +
diff_list =[] # to check Vdiff
diff = 1
Iter = 0
#q = 1/beta*1.002
q = 0.9941152263374485
# pre-allocation
V = np.ones((nb_agrid, nb_e))
apr = np.zeros((nb_agrid, nb_e))
apr_arg = np.zeros((nb_agrid, nb_e), dtype = int) #???
con = np.zeros((nb_agrid, nb_e))

while diff > tol_iter:
    Iter = Iter +1
    Vlast = V.copy()
    
    # Use interpolation to match Vlast to afgrid
    temp1 = interp1d(agrid, Vlast[:,0], kind = 'cubic')(afgrid)
    temp2 = interp1d(agrid, Vlast[:,1], kind = 'cubic')(afgrid)
    V_f = np.array([temp1, temp2]).T
    
    # Loop over assets
    for ia in range(0, nb_agrid):
        # Loop over endowments
        for ie in range(0,nb_e):
            income = agrid[ia] + egrid[ie] 
            # the next period probability varies depending on whether the current endowment is el or eh,
            Vchoice = u(np.maximum(income - q*afgrid.T, 1.0e-10)) + beta*(V_f @ P.T)[:,ie]
            V[ia, ie] = np.max(Vchoice)
            apr_arg[ia, ie] = np.argmax(Vchoice)
            apr[ia, ie] = afgrid[apr_arg[ia, ie]]
            con[ia, ie] = income - q*apr[ia, ie]

    # check the convergence of Value function        
    diff = np.max(np.max(abs(V-Vlast)))
    diff_list.append(diff)
    if (Display == 1) & (Iter%100 == 0): # Notifies you every time it exceeds 100
        print('Iteration:' + str(Iter), ' Diff:' + str(diff))  
# -

plt.plot(agrid, apr, agrid, agrid)


# ## VFI function
# - VFI calculation finds the individual’s optimal decisions at a given q.
# - By simulating, can get aggregate values(distribution, agg.consumption, agg. saving) for a given q
#

# @ray.remote
def vfi(q, Vguess = np.ones((nb_agrid, nb_e))):
    diff_list =[] 

    V = Vguess.copy()

    diff = 1
    Iter = 0
    apr = np.zeros((nb_agrid, nb_e))
    apr_arg = np.zeros((nb_agrid, nb_e), dtype = int) # index
    con = np.zeros((nb_agrid, nb_e))

    while diff > tol_iter:
        Iter = Iter +1
        Vlast = V.copy()
        
        # Use interpolation to match Vlast to afgrid
        temp1 = interp1d(agrid, Vlast[:,0], kind = 'linear')(afgrid)
        temp2 = interp1d(agrid, Vlast[:,1], kind = 'linear')(afgrid)
        Vlast_interp = np.array([temp1, temp2]).T
        
        # pre-allocation
        #V = np.zeros((na, ne))

        
        # Loop over assets
        for ia in range(0, nb_agrid):
            
            # Loop over endowments
            for ie in range(0,nb_e):
                income = agrid[ia] + egrid[ie] 
                
                # the next period probability varies depending on whether the current endowment is el or eh,
                
                Vchoice = u(np.maximum(income - q*afgrid.T, 1.0e-10)) + beta*(Vlast_interp @ P.T)[:,ie]
                V[ia, ie] = np.max(Vchoice)
                apr_arg[ia, ie] = np.argmax(Vchoice)
                apr[ia, ie] = afgrid[apr_arg[ia, ie]]
                con[ia, ie] = income - q*apr[ia, ie]

        # check the convergence of Value function            
        diff = np.max(np.max(abs(V-Vlast)))
        diff_list.append(diff)
        if (Display == 1) & (Iter%100 == 0): # Notifies you every time it exceeds 100
            print('Iteration no. ' + str(Iter), ' max val fn diff is ' + str(diff))     
            
    return V, Vlast, diff_list, apr_arg, apr, con

q = 0.9941152263374485
Vguess = V

V, Vlast, diff_list, apr_arg, apr, con = vfi(q)

# ## Simulation

Nsim = 1
Tsim = 1_000_000#_000_000
# pre-allocation
esim = np.zeros(Tsim)
a_sim = np.zeros(Tsim) # ind = index
apr_sim = np.zeros(Tsim)
apr_arg_sim = np.zeros(Tsim)

esim=mc.simulate(Tsim, init="0.1").astype('float')

estate = (esim==1)
plt.hist(esim,100)
plt.show()

for t in range(Tsim-1):
    if estate[t]==True:
        apr_sim[t] = interp1d(agrid, apr[:,1],kind='linear')(a_sim[t])
        a_sim[t+1]= apr_sim[t]
    else:
        apr_sim[t] = interp1d(agrid, apr[:,0],kind='linear')(a_sim[t])
        a_sim[t+1]= apr_sim[t]
avg_a = np.mean(a_sim[-int(Tsim/3):])

plt.hist(a_sim,100)
plt.show()

avg_a = np.mean(a_sim[-int(Tsim/3):])
print(avg_a)

Tsim = 1_000_000#_000_000
# pre-allocation
esim = np.zeros(Tsim)
a_sim = np.zeros(Tsim) # ind = index
apr_sim = np.zeros(Tsim)
apr_arg_sim = np.zeros(Tsim)
esim=mc.simulate(Tsim, init="0.1").astype('float')
estate = (esim==1)


def simulate(Tsim = 1000_000):
    esim=mc.simulate(Tsim, init="0.1").astype('float')
    estate = (esim==1)
    for t in range(Tsim-1):
        if estate[t]==True:
            apr_sim[t] = interp1d(agrid, apr[:,1],kind='linear')(a_sim[t])
            a_sim[t+1]= apr_sim[t]
        else:
            apr_sim[t] = interp1d(agrid, apr[:,0],kind='linear')(a_sim[t])
            a_sim[t+1]= apr_sim[t]
    avg_a = np.mean(a_sim[-int(Tsim/3):])
    return a_sim, estate, esim, avg_a


a_sim, estate, esim, avg_a= simulate()

avg_a

plt.hist(a_sim,100)
plt.show()

# ### Find equilibrium q with BISECTION METHOD

# +
tol = 1e-2
avg_a = 1
Display = 0
# q 
qmin = beta
qmax = 1
i = 0
avg_a_list = []
q_list = []
q0 = (qmin+qmax)/2

while abs(avg_a) > tol:
    V, Vlast, diff_list, apr_arg, apr, con = vfi(q0)
    a_sim, estate, esim, avg_a = simulate()
    i += 1
    
    q_list.append(q0)
    avg_a_list.append(avg_a)
    print('Iteration no.: ' + str(i), 'Value of q: ' + str(q0), 'Avg asset: ' + str(avg_a))     
    if avg_a > 0:
        qmin = q0
    else:
        qmax = q0
    q0 = (qmin + qmax) / 2   
    
# -

# ## Optimal choices at the equilibrium q

equil_q = q0
V, Vlast, diff_list, savind, sav, con = vfi(equil_q) 
a_sim, estate, esim, avg_a = simulate() 

# Convergence of Value Function
plt.plot(diff_list)

# Optimal Decision
# consumption smoothing: save more with eh, dissave with el. So eh is always above than el.
fig, ax = plt.subplots()
plt.plot(agrid, apr[:,0], label = 'unemploymed')
plt.plot(agrid, apr[:,1], label = 'employed')
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
# measure of people at el and eh
plt.hist(esim,bins = 100)
plt.ylabel('population')
plt.title('Endowment distribution')

## convergence check
# asset market clear check 
plt.plot(avg_a_list)
plt.xlabel('Time Period')
plt.title('Mean Asset Convergence')
plt.show()

# # asset distribution
plt.hist(a_sim,bins = 40,facecolor=(.7,.7,.7),edgecolor='black')
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

asset_level = np.unique(a_sim, return_counts = True)[0]

pop_dist = np.unique(a_sim, return_counts = True)[1]

asset_level += abs(amin)

lorenz_and_gini(pop_dist, asset_level)


