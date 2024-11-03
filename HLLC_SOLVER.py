#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 18:40:23 2024

@author: krishanu
"""

import matplotlib
from matplotlib import pyplot 
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16




gamma = 1.4     #Гамма, она же каппа
#Различные формулы из гаммы
g_m_1_over_2g = (gamma-1)/2/gamma      #g1
g_p_1_over_2g = (gamma+1)/2/gamma      #g2
g_m_1_over_2g_inv = 1/g_m_1_over_2g    #g3
g_m_1_over_2_inv = 2/(gamma-1)         #g4
g_p_1_over_2_inv = 2/(gamma+1)         #g5
g_m_1_over_g_p_1 = (gamma-1)/(gamma+1) #g6
g_m_1_over_2 = (gamma-1)/2             #g7
g_m_1 = gamma-1  

tol = 1e-8




def sound_speed(d,p):
    return (gamma*(p/d))**0.5



def sound_speed_w(W):
    d = W[0]
    p = W[2]
    return sound_speed(d,p) 


def guess_p(W_l,W_r,label):
    c_l = sound_speed(W_l[0],W_l[2])
    c_r = sound_speed(W_r[0],W_r[2])

    
    W_aver = 0.5*(W_l+W_r)
    p_pv = W_aver[2] - 0.5*(W_r[1]-W_l[1])*W_aver[0]*0.5*(c_l+c_r)
    p_0 = max(tol,p_pv)
    
    if label == 'TR':
        return ((c_l + c_r - 0.5*g_m_1*(W_r[1] - W_l[1]))/
                ((c_l/W_l[2]**g_m_1_over_2g) + (c_r/W_r[2]**g_m_1_over_2g) ))**g_m_1_over_2g_inv
    
    elif label == 'PV':

        return p_0
    
    elif label == 'TS':
        A_k = lambda x : g_p_1_over_2_inv/x
        B_k = lambda x : g_m_1_over_g_p_1*x
        p_ts = ((A_k(W_l[0])/(p_pv + B_k(W_l[2])))**0.5*W_l[2] + (A_k(W_r[0])/(p_pv + B_k(W_r[2])))**0.5*W_r[2] \
                - (W_r[1]-W_l[1])) /\
        ((A_k(W_l[0])/(p_pv + B_k(W_l[2])))**0.5 + (A_k(W_r[0])/(p_pv + B_k(W_r[2])))**0.5)  
        return max(tol,p_ts)
    else:
        return W_aver[2]
        


def init(case):
    if case == 'sod':
        W_l = np.array([1, 0.75, 1])
        W_r = np.array([0.125, 0, 0.1])
        t = 0.25
    elif case == '123':
        W_l = np.array([1, -2, 0.4])
        W_r = np.array([1, 2, 0.4])
        t = 0.15
    elif case == 'left-woodward':
        W_l = np.array([1, 0, 1000])
        W_r = np.array([1, 0, 0.1])
        
        t = 0.012
    else : print('Unknown case!')
    return W_l, W_r, t


W_l,W_r,_ = init('sod')


def U_to_W(U):
    W = np.zeros_like(U)
    W[0] = U[0]
    W[1] = U[1]/U[0]
    W[2] = g_m_1*(U[2] - 0.5*U[1]**2/U[0])
    return W



def W_to_U(W):
    U = np.zeros_like(W)
    U[0] = W[0]
    U[1] = W[1]*W[0]
    U[2] = 0.5*W[1]**2*W[0]+W[2]/ g_m_1
    return U


def flux(W):
    F = np.zeros_like(W)
    F[0] = W[1]*W[0]
    F[1] = W[1]**2*W[0] + W[2]
    F[2] = W[1]*(0.5*W[1]**2*W[0]+W[2]/ g_m_1 + W[2])
    return F



def q(p,p_star):
    if p_star > p:
        return (1 + g_p_1_over_2g*(p_star/p - 1))**0.5
    else :
        return 1




def get_speeds(W_l,W_r,p_star):
    S_l = W_l[1] - sound_speed_w(W_l)*q(W_l[2],p_star)
    S_r = W_r[1] + sound_speed_w(W_r)*q(W_r[2],p_star)
    
    S_star = (W_r[2] - W_l[2] + \
              W_l[0]*W_l[1]*(S_l - W_l[1]) - \
              W_r[0]*W_r[1]*(S_r - W_r[1]))/( W_l[0]*(S_l - W_l[1]) - \
              W_r[0]*(S_r - W_r[1]) )
    return np.asarray((S_l,S_r,S_star))

def F_HLLC(W,S_star,S):
    D = np.asarray([0,1,S_star])
    F_star = (S_star*(S*W_to_U(W) - flux(W)) + \
              S*(W[2] + W[0]*(S - W[1])*(S_star - W[1]))*D)/\
    (S - S_star)
    return F_star



def hllc_flux(W_l,W_r):
    p_star = guess_p(W_l,W_r,'TR')
    S_l,S_r,S_star = get_speeds(W_l,W_r,p_star)
    if 0 <= S_l:
        return flux(W_l)
    if S_l <= 0 <= S_star:
        return F_HLLC(W_l,S_star,S_l)
    if S_star <= 0 <= S_r:
        return F_HLLC(W_r,S_star,S_r)
    if 0 >= S_r:
        return flux(W_r)




N_points = 101
x = np.linspace(0,1,N_points)
decay_pos = 0.3
W_correct= np.zeros((N_points,3))



W_l,W_r,t = init('sod')



dx = 1./(N_points - 1)
sigma = 0.9

W = np.zeros((N_points,3))
fluxes = np.zeros((N_points - 1,3))
U = np.zeros_like(W) 


W[np.where(x<decay_pos),:] = W_l
W[np.where(x>=decay_pos),:] = W_r



t_ = 0
#dt = 0.0002
U_n = np.copy(U)
while t_<t:
    U = W_to_U(W.T).T
    hllc_fluxes = np.zeros((N_points - 1,3))
    speeds = np.zeros(N_points - 1)
    for i,x_ in enumerate(x[:-1]):
        
        hllc_fluxes[i] =hllc_flux(W[i],W[i+1])
        c_l = sound_speed(W[i][0],W[i][2])
        c_r = sound_speed(W[i+1][0],W[i+1][2])
        speeds[i] = max(abs(W[i][1])+c_l,abs(W[i+1][1])+  c_r)

    
    dt = sigma*dx/max(speeds)
    U_n[1:-1,:] = U[1:-1,:] + dt/dx*(hllc_fluxes[:-1,:]-hllc_fluxes[1:,:]) 
    U_n[0,:] = U_n[1,:]
    U_n[-1,:] = U_n[-2,:]
    W = U_to_W(U_n.T).T
    t_=t_+dt


%store -r


fig,axs = pyplot.subplots(1,3,figsize=(18,6)
                         )
for ax, W_, W_r_, W_c, y_label in zip(axs, W.T, W_roe.T, W_correct.T, (r'$\rho$',r'$u$',r'$p$') ):
#for ax, W_,  W_c, y_label in zip(axs, W.T,  W_correct.T, (r'$\rho$',r'$u$',r'$p$') ):
    ax.plot(x,W_,'o-',label='HLLC')
    #ax.plot(x,W_r_,'o',label='Roe')
    ax.plot(x,W_c,label='Exact')
    ax.set_ylabel(y_label)
    ax.set_xlabel('x')
    scale_y = 1.1*abs(max(W_c)-min(W_c))
    ax.set_ylim(0.5*(max(W_c)+min(W_c) - scale_y), 0.5*(max(W_c)+min(W_c) + scale_y))
    ax.legend(loc='best')
    ax.grid()
    #plt.savefig('HLLC.png',dpi = 400)



# Setup grid
N_points = 101
x = np.linspace(0, 1, N_points)
decay_pos = 0.3

# Initialize states
W_l, W_r, t = init('sod')
W = np.zeros((N_points, 3))
W[x < decay_pos, :] = W_l
W[x >= decay_pos, :] = W_r

# Plot the initial state
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for ax, W_, y_label in zip(axs, W.T, (r'$\rho$', r'$u$', r'$p$')):
    ax.plot(x, W_, 'o-', label='Initial State')
    ax.set_ylabel(y_label)
    ax.set_xlabel('x')
    ax.legend(loc='best')
    ax.grid()

plt.suptitle('Initial State of the Flow')
plt.show()



















