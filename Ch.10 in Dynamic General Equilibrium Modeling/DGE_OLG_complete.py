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

# ## Replication of the model in Ch10: The OLG Model with Income Uncertainty
# - Book: Dynamic General Equilibriumm Modeling (3rd ed.) by Burkhard Heer , Alfred Maußner
# - Code written by MJ Son

# +
import pandas as pd
import numpy as np
import quantecon as qe
from scipy.stats import norm
from scipy.interpolate import interp1d
import time
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# abbreviations
exp = np.e
log= math.log

# -

# ## Function

# +
def wage(Ktilde, Ltilde): # w: real wage tilde, that is, W/A
    return (1-alpha) * Ktilde**alpha * Ltilde**(-alpha)

def interest(Ktilde,Ltilde): # r
    return alpha * Ktilde**(alpha-1) * Ltilde**(1-alpha)

def production(Ktilde,Ltilde): # production function
    return Ktilde**alpha * Ltilde**(1-alpha)


# -

def u(c,l):
    if eta == 1:
        y = gamma*log(c) + (1-gamma)*log(1-l)
    else:
        y = ( (c**gamma)*((1-l)**(1-gamma)) )**(1-eta) / (1-eta)
    return y



# +
# first order condition에서 l에 대해서 푼 것

def optimal_labor(a0,a1):
    
    labor = np.array([])
    
    w0 = (1-taun-taup)*w*ef[it]*perm[iperm]*ye1[iy]
    labor_value = gamma - ((1+(1-tauk)*(r-delta))*a0+trtilde-ygrowth*a1)*(1-gamma)/w0
    
    labor = np.append(labor, np.array([labor_value]))
    
    labor[labor < 0.0] = 0.0
    labor[labor > 0.6] = 0.6

    
    return labor
# -

# ## Parameterization and import data

# +
# asset grids
kmin = 0
kmax = 15 # upper limit of capital grid
na = 2000 # number of grid points over assets a in [kmin, kmax]
agrid = np.linspace(kmin, kmax, na) # asset grid policy function


labormax = 0.6 # maximum labor supply
phi = 0.8 # update aggregate variables in outer iteration over K, L, tr, taup, taun
tol = 0.0001 # percentage deviation of final solution K and L
neg = -1e10      # initial value for value function 
nq = 30 # number of outer iterations
# -

# Import data 
# : survival probabilities and age-efficiency component from Excel file 'survival_probs.xlsx'
data = pd.read_excel('survival_probs.xlsx')
df = pd.DataFrame(data, columns = ['sp1','ef'])
display(df)
arr = np.array(df)
sp1 = arr[:,0]
ef = arr[0:45, 1]

# +
# Demographics
nage = 70 # maximum age
nw = 45 # number of working years
Rage = 46 # first period of retirement
nr = nage - Rage + 1 # number of retirement years: 25
popgrowth0 = 0.00754 # population growth rate

# preferences
beta1 = 1.011 # discount factor
gamma=0.33  # weight of consumption in utility
lbar = 0.25 # steady-state labor supply
eta = 2.0 # 1/IES

# production
ygrowth = 1.02 # annual growth factor
alpha = 0.35 # production elasticity of capital
delta = 0.083 # depreciation rate
rbbar = 1.04 # initial annual real interest rate on bonds

# fiscal policy and social security
taulbar = 0.28 # both taun+taup = 0.28
tauk = 0.36 # capital income tax rate
tauc = 0.05 # consumption tax rate
taup = 0.124 # initial guess for social security contribution rate
replacement_ratio=0.352 # gross replacement ratio US
by = 0.63 # debt-output ratio
gy = 0.18 # government consumption-output ratio

# +
# measure of living persons
# compute the stationary age distribution in the population 
# the mass of the one-year-old, mu1 = mass[0]
mass = np.ones(nage)

for i in range(nage-1):
    mass[i+1] = mass[i]*sp1[i]/(1+popgrowth0)

mass = mass/mass.sum() # normalize
# -

plt.xlabel('age')
plt.ylabel('measure')
plt.plot(range(21,91), mass)
# The measure of the cohorts in our calibration declines monotonically with age s

# +
# productivity of workers
# discretization of the individual idiosyncratic productivity 

nperm = 2 # number of permanent productivities
perm = np.zeros(2)
perm[0] = 0.57 # e1, highschool
perm[1] = 1.43 # e2, college

rho = 0.96 # autoregressive parameter
sigma_y1 = 0.38 # variance for 20-year old, log earnings
sigma_e = 0.045 # earnings disturbance term variance
ny = 5 # number of productivity types 
m = 1 # multiple of standard deviation determining the width of productivity grid (calibrated to replicate Gini wages = 0.37)

# compute productivity grid 
sy = np.sqrt(sigma_e/(1-rho**2)) # standard deviation of earnings
ye = np.linspace(-m*sy, m*sy, ny)

# transition matrix using Tauchen's approximation 
mc = qe.markov.approximation.tauchen(rho, sigma_e, 0.0, m, ny)
py = mc.P # transition matrix is stored in object 'P'


# +
# mass of the workers
muy = np.zeros(ny)
w = ye[1] - ye[0] # 간격

# first year mass distribution
muy[0] = norm.cdf( (ye[0]+w/2)/np.sqrt(sigma_y1) )
muy[ny-1] = 1-norm.cdf( (ye[ny-1]-w/2)/np.sqrt(sigma_y1) )

for i in range(1, ny-1):
    muy[i] = norm.cdf((ye[i]+w/2)/np.sqrt(sigma_y1)) - norm.cdf((ye[i]-w/2)/np.sqrt(sigma_y1))
    
ye1 = np.exp(ye)
# -

# ## Outer loop

# +
# initial guesses
rtilde = 0.03 # interest rate r/A
Ltilde = 0.3 # A*L effective labor
mean_labor = 0.3 # l bar : mean working hours
Ktilde = (alpha/(rtilde + delta))**(1/(1-alpha)) * Ltilde 
print(f'Ktilde = {Ktilde}')

Ltilde_old = 100
Ktilde_old = 100 # initialization so that Kold-Ktilde is larger than tolerance level for outer loop
omega = Ktilde*1.2
trtilde = 0.01
w = wage(Ktilde, Ltilde)
r = interest(Ktilde, Ltilde)
rb = (1-tauk)*(r-delta) # interest rate on bonds

pen = replacement_ratio*w*mean_labor # individual pension 
taup = pen*sum(mass[nw:nage])/(w*Ltilde) # balanced budget social security # ??
taun = taulbar - taup
bequests = 0

# +
# %%time
iternum = 0
crit = 1
crit_list = []

while iternum < 100 and crit > 0.001:
    iternum += 1
    print(f'iter_num = {iternum}')
    crit = max(abs((Ktilde - Ktilde_old)/ Ktilde_old),abs((Ltilde - Ltilde_old)/ Ltilde_old))
    print(crit)
    crit_list.append(crit)
    
    w = wage(Ktilde, Ltilde)
    r = interest(Ktilde, Ltilde)
    rb = (1-tauk)*(r-delta)
    
    ##### VFI for retirees
    Vr = np.zeros((na, nr)) 
    savr = np.zeros((na, nr)) 
    conr = np.zeros((na, nr)) 
    savindr = np.zeros((na, nr), dtype=int)
    
    # values for the last period of life is pre-determined
    # loop over different asset levels in the last period
    for ia in range(na):
        c = ( agrid[ia]*(1+(1-tauk)*(r-delta)) + pen + trtilde )/(1+tauc)
        Vr[ia, nr-1] = u(c,0)
        conr[ia, nr-1] = c
        savr[ia, nr-1] = 0
        savindr[ia, nr-1] = np.argwhere(agrid == 0)

    for it in range(nr-2, -1, -1):
        print('Solving at age (retiree): '+str(it+Rage))

        # loop over assets
        for ia in range(0,na):
            income = agrid[ia]*(1+(1-tauk)*(r-delta)) + pen + trtilde
            c = (income - ygrowth*agrid)/(1+tauc)
            Vchoice = u(np.maximum(c, 1.0e-10),0) + ygrowth**(gamma*(1-eta))*beta1*sp1[it+Rage-1]*Vr[:,it+1]
            Vr[ia,it] = np.max(Vchoice)
            savindr[ia, it] = np.argmax(Vchoice)
            savr[ia, it] = agrid[savindr[ia,it]]
            conr[ia, it] = (income - ygrowth*savr[ia,it])/(1+tauc)
            
    ##### VFI for worker
    Vw = np.zeros((nw, nperm, na, ny))
    savw = np.zeros((nw, nperm, na, ny))
    conw = np.zeros((nw, nperm, na, ny))
    savindw = np.zeros((nw, nperm, na, ny), dtype = int)
    laborw = np.zeros((nw, nperm, na, ny))
    
    # loop over time period, education, different asset, and idiosyncratic income

    for it in range(nw-1, -1, -1):
        print('Solving at age: '+str(it+1))

        for iperm in range(nperm):
            for ia in range(na):
                for iy in range(ny):

                    if it == nw-1:
                        labor_income = (1-taun-taup)*w*ef[it]*perm[iperm]*ye1[iy]*optimal_labor(agrid[ia], agrid)
                        income = labor_income + agrid[ia]*(1+(1-tauk)*(r-delta)) + trtilde
                        c = (income - ygrowth*agrid)/(1+tauc)
                        Vchoice = u(np.maximum(c, 1.0e-10),optimal_labor(agrid[ia],agrid)) + ygrowth**(gamma*(1-eta))*beta1*sp1[it]*Vr[:,0]
                        Vw[it, iperm, ia, iy] = np.max(Vchoice)
                        savindw[it, iperm, ia, iy] = np.argmax(Vchoice)
                        savw[it, iperm, ia, iy] = agrid[savindw[it, iperm, ia, iy]]

                        laborw[it, iperm, ia, iy] = optimal_labor(agrid[ia], savw[it, iperm, ia, iy])

                        labor_income = (1-taun-taup)*w*ef[it]*perm[iperm]*ye1[iy]*optimal_labor(agrid[ia], savw[it, iperm, ia, iy])
                        income = labor_income + agrid[ia]*(1+(1-tauk)*(r-delta)) + trtilde
                        conw[it, iperm, ia, iy] = (income - ygrowth*savw[it, iperm, ia, iy])/(1+tauc)


                    else:
                        labor_income = (1-taun-taup)*w*ef[it]*perm[iperm]*ye1[iy]*optimal_labor(agrid[ia], agrid)
                        income = labor_income + agrid[ia]*(1+(1-tauk)*(r-delta)) + trtilde
                        c = (income - ygrowth*agrid)/(1+tauc)
                        Vchoice = u(np.maximum(c, 1.0e-10),optimal_labor(agrid[ia],agrid)) + ygrowth**(gamma*(1-eta))*beta1*sp1[it]*(Vw[it+1][iperm] @ py.T)[:,iy]
                        Vw[it, iperm, ia, iy] = np.max(Vchoice)
                        savindw[it, iperm, ia, iy] = np.argmax(Vchoice)
                        savw[it, iperm, ia, iy] = agrid[savindw[it, iperm, ia, iy]]


                        laborw[it, iperm, ia, iy] = optimal_labor(agrid[ia], savw[it, iperm, ia, iy])

                        labor_income = (1-taun-taup)*w*ef[it]*perm[iperm]*ye1[iy]*optimal_labor(agrid[ia], savw[it, iperm, ia, iy])
                        income = labor_income + agrid[ia]*(1+(1-tauk)*(r-delta)) + trtilde
                        conw[it, iperm, ia, iy] = (income - ygrowth*savw[it, iperm, ia, iy])/(1+tauc)
                        
    ##### Wealth distribution
    
    ### distribution of wealth among workers: dw
    dw = np.zeros((nw, nperm, na, ny))

    # measure at age 1 
    # all agents have zero wealth
    for iy in range(ny):
        dw[0, 0, 0, iy] = 1/2 * mass[0] * muy[iy]
        dw[0, 1, 0, iy] = 1/2 * mass[0] * muy[iy]

    for it in range(0, nw-1):
        for iperm in range(nperm):
            for iy in range(ny):
                for ia in range(na):

                    next_asset = savindw[it, iperm, ia, iy]

                    for ay in range(ny):
                        addmass = dw[it, iperm, ia, iy]* (sp1[it]/(1+popgrowth0)) *py[iy,ay]
                        dw[it+1, iperm, next_asset, ay] = dw[it+1, iperm, next_asset, ay] + addmass
                        
    temp = []
    for it in range(0,nw-1):
        print(abs(np.sum(dw[it], axis=(0,1,2)) - mass[it]) < 1e-15)
        value = np.sum(dw[it], axis=(0,1,2)) - mass[it]
        temp.append(value)
        
    ### distribution of wealth among retirees: dr
    dr = np.zeros((nr, na))
    dr[0,:] = np.sum(dw[44,:,:,:], axis = (0,2))
    for it in range(nr-1):
        for ia in range(na):

            next_asset = savindr[ia,it]
            addmass = dr[it, ia]* sp1[Rage+it-1]/(1+popgrowth0) 
            dr[it+1, next_asset] = dr[it+1, next_asset] + addmass
            
    for it in range(25):
        print(abs( np.sum(dr, axis=1)[it] - mass[it+45]) < 1e-3)
        
    ### distribution of wealth over age
    dist_wealth_age_withinage_worker = np.sum(dw, axis = (1,3))
    dist_wealth_withinage = np.vstack((dist_wealth_age_withinage_worker, dr))
    dist_meanwealth_age = []

    for it in range(nage):
            mean_asset = dist_wealth_withinage[it]@agrid
            dist_meanwealth_age.append(mean_asset)
            
    ### distribution of wealth
    dist_wealth = np.sum(dist_wealth_withinage, axis = 0)
    
    ### distribution of earnings
    dist_earnings = np.sum(dw, axis = (0,1,2))
    
    ##### Aggregate
    ### Aggregate capital
    Ytilde = production(Ktilde, Ltilde)
    Btilde = by*Ytilde
    omega_new = dist_wealth @ agrid
    omega = phi*omega + (1-phi)*omega_new
    
    Ktilde_new = omega_new - Btilde
    Ktilde_old = Ktilde
    Ktilde = phi*Ktilde_old + (1-phi)*Ktilde_new

    ### Aggregate effective labor supply
    agg_labor = 0
    for it in range(nw):
        for iperm in range(nperm):
            for ia in range(na):
                for iy in range(ny):
                    add = laborw[it, iperm, ia, iy] * dw[it, iperm, ia, iy]
                    agg_labor += add

    Ltilde_new = agg_labor
    Ltilde_old = Ltilde
    
    Ltilde = phi*Ltilde_old + (1-phi)*Ltilde_new

    ### Total transfer
    # aggregate consumption
    agg_con = 0

    for it in range(nw):
        for iperm in range(nperm):
            for ia in range(na):
                for iy in range(ny):
                    add = conw[it, iperm, ia, iy] * dw[it, iperm, ia, iy]
                    agg_con += add

    for it in range(nr):
        for ia in range(ia):
            add = conr[ia, it] * dr.T[ia,it]
            agg_con += add
            
    total_tax = taun*w*Ltilde + tauk*(r-delta)*Ktilde + tauc*agg_con
    
    #aggregate bequest
    agg_beq = 0

    for it in range(nw):
        for iperm in range(nperm):
            for ia in range(na):
                for iy in range(ny):
                    add = (1-sp1[it])*(1+(1-tauk)*(r-delta))*savw[it, iperm, ia, iy]*dw[it, iperm, ia, iy]
                    agg_beq += add

    for it in range(nr):
        for ia in range(na):
            add = (1-sp1[Rage + it -1])*(1+(1-tauk)*(r-delta))*savr[ia, it]*dr.T[ia, it]
            agg_beq += add
            
    Gtilde = gy*Ytilde

    trtilde_new = total_tax + agg_beq + (ygrowth*(1+popgrowth0) - (1+(1-tauk)*(r-delta)))*Btilde - Gtilde # government budget constraint 사용
    trtilde = phi*trtilde + (1-phi)*trtilde_new
    
    mean_labor = 0

    for it in range(nw):
        for iperm in range(nperm):
            for ia in range(na):
                for iy in range(ny):
                    add = laborw[it, iperm, ia, iy]/(sum(mass[0:nw]))*dw[it, iperm, ia, iy]
                    mean_labor += add
                    
    pennew = replacement_ratio*w*mean_labor # individual이 받는 pension amount
    pen = phi*pen + (1-phi)*pennew
    
    taupnew = pen*sum(mass[Rage:])/(w*Ltilde) # 앞에 term이 total pension, aggregate pension amount
    taup = phi*taup + (1-phi)*taupnew
    
    taunnew = taulbar-taup
    taun = phi*taun + (1-phi)*taunnew
# -

# ## Plot

# Asset choice of retirees at the first period of retirement
fig, ax = plt.subplots()
plt.plot(agrid, savr[:,0], label = 'asset choice') # at age 46, first period of retirement
plt.plot(agrid, agrid, label = '45$\degree$')
plt.title('Asset choice')
plt.xlabel('asset level at age 46 (first year of retirement)')
plt.ylabel('next-period asset choice')
fig.legend()

# Consumption choice of retirees at the first period of retirement
fig, ax = plt.subplots()
plt.plot(agrid, conr[:,0])
# plt.plot(agrid, agrid, label = '45$\degree$')
plt.title('Consumption choice')
plt.xlabel('asset level at age 46 (first year of retirement)')
plt.ylabel('consumption choice')
fig.legend()

# Loabor choice of workers at the first period
fig, ax = plt.subplots()
plt.plot(agrid, laborw[0][1][:,4]) # period, education level, productivity (shock) level
# plt.plot(agrid, agrid, label = '45$\degree$')
plt.title('labor choice')
plt.xlabel('asset level at age 21 (first year)')
plt.ylabel('labor choice')
fig.legend()

# +
# Labor choice of skilled worker and unskilled worker

fig, ax = plt.subplots()
# period, education level, productivity (shock) level
plt.plot(agrid, laborw[0][0][:,3], label = 'unskilled')  # unskilled labor
plt.plot(agrid, laborw[0][1][:,3], label = 'skilled') # skilled labor
# plt.plot(agrid, agrid, label = '45$\degree$')
plt.title('labor choice')
plt.xlabel('asset level at age 21 (first year)')
plt.ylabel('labor choice')
fig.legend()
# -

# Asset choice of worker at the first period
fig, ax = plt.subplots()
plt.plot(agrid, savw[0][0][:,0]) # period, education level, productivity (shock) level
plt.plot(agrid, agrid, label = '45$\degree$')
plt.title('Asset choice')
plt.xlabel('asset level at age 21 (first year)')
plt.ylabel('next-period asset choice')
fig.legend()

# consumption choice of worker at the first period
fig, ax = plt.subplots()
plt.plot(agrid, conw[0][0][:,0]) # period, education level, productivity (shock) level
# plt.plot(agrid, agrid, label = '45$\degree$')
plt.title('Consumption choice')
plt.xlabel('asset level at age 21 (first year)')
plt.ylabel('consumption choice')
fig.legend()

# Wealth-Age Profile
plt.plot(range(1,nage+1), dist_meanwealth_age)
plt.title('mean wealth over age (distribution of wealth over age)')
plt.xlabel('age')
plt.ylabel('mean welath')

# +
# wealth distribution

plt.bar(agrid, dist_wealth, width = 0.2)

plt.title('wealth distribution')
plt.xlabel('asset level')
plt.ylabel('population')
plt.xlim(0, 10)
plt.ylim(0, 0.2)

# +
# wealth distribution zoom in
plt.bar(agrid, dist_wealth, width = 0.2)

plt.title('wealth distribution')
plt.xlabel('asset level')
plt.ylabel('population')
plt.xlim(0, 15)
plt.ylim(0, 0.01)
# -

# distribution of earnings
plt.plot(ye1 ,dist_earnings)
plt.title('distribution of earnings')
plt.xlabel('level of earnings')
plt.ylabel('distribution')

# Proportion of people with zero assets by age 
plt.plot(list(range(1, nage+1)),dist_wealth_withinage[:,0])
plt.xlabel('Age')
plt.ylabel('percentage')
plt.title('Percentage of zero asset holder(age)')
plt.ylim(0,0.025)
plt.xlim(1,nage)
plt.grid(True)

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

# inequality in wealth distribution
lorenz_and_gini(dist_wealth, agrid, curve = True)

# inequality in earnings
lorenz_and_gini(dist_earnings, ye1, curve = True)

# +
# Wealth gini for each age
gini_age = []
for i in range(1,nage):
    gini = lorenz_and_gini(dist_wealth_withinage[i,:],agrid,curve=False)
    gini_age.append(gini)
    
gini_age
# -

# The reason it appears as a U-shape is because there are many people with zero wealth among the young and the old.
plt.plot(range(1,nage),gini_age)
plt.title('Gini coefficient based on each age')
plt.xlabel('age')
plt.ylabel('gini')
plt.grid(True)


