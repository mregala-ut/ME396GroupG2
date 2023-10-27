#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 

@author: Matthew Regala, Ashton Siegel, Aaron Park
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# User Defined
alpha_0_deg = 10
velocity_initial = 0
clalpha_defined = 2 * np.pi
cd_0 = 0.008
chord = 0.30  # m
span = 1.00  # m
#m = 798 # mass kg
#d_w = 3.6 # wheelbasse m
#r = 0.23 # tire radius m
#p_e = 250 # starting brake horsepower Hp
#p_max = 850 # maximum Hp
#t_e = 0.98 # transmission efficiency
#s_f = 1.7 # static friction coefficient between tire and asphalt
#g = 9.81 # gravity m / s^2
#c_dc = 0.85 # car drag coefficient
#A_c = 1.30 # frontal area of car m^2
#c_rr = 0.012 # rolling resistance coefficient

# Parameters & Calculations
N = 20  # Number of nodes
air_density = 1.293  # kg/m^3
AR = span / chord  # Aspect Ratio
e = 0.20
clalpha = clalpha_defined * AR / (2 + AR)
e = 0.20
e_oswald = 0.7
elastic_offset = chord * e  # m
JG = 1000 * (chord * 3/4 + e) * span / 2 / np.deg2rad(1)
alpha_0_rad = np.deg2rad(10)

theta_values = []
q_inf_values = []
lift_distributions = []
drag_distributions = []
lift_areas = []
drag_areas = []
#optimal_power = []

def FEM_module(velocity):
    '''
    

    Parameters
    ----------
    velocity : nueric
        This function takes an input velocity, and calculates the resultant
        angle of attacks, lift and drag distributions, and total lift and drag.
        These values are appended to theta_values,lift_distributions,
        drag_distributions, lift_areas, drag_areas.

    Returns
    -------
    None.
    
    Author: Matthew Regala

    '''
    q_inf = 1/2 * air_density * velocity**2
    q_inf_values.append(q_inf)
    beta = q_inf * clalpha * elastic_offset

    # xi (master element coordinate)
    # x_i: nodal coordinate in physical space - left element end
    # x_j: nodal coordinate in physical space - right element end
    Xi, x, x_i, x_j = sp.symbols('Xi x x_i x_j')
    # element length
    h = 1 / N
    # shape functions
    psi_1 = 1/2 * (1 - Xi)
    psi_2 = 1/2 * (1 + Xi)
    # derivatives of shape functions
    dpsi_1 = sp.diff(psi_1, Xi)
    dpsi_2 = sp.diff(psi_2, Xi)
    # element matrix
    ke11 = h/2 * sp.integrate(JG * 2/h * dpsi_1 * 2/h * dpsi_1, (Xi, -1, 1)) + beta * h / 2 * sp.integrate(psi_1**2, (Xi, -1, 1))
    ke12 = h/2 * sp.integrate(JG * 2/h * dpsi_1 * 2/h * dpsi_2, (Xi, -1, 1)) + beta * h / 2 * sp.integrate(psi_1**2, (Xi, -1, 1))
    ke21 = h/2 * sp.integrate(JG * 2/h * dpsi_2 * 2/h * dpsi_1, (Xi, -1, 1)) + beta * h / 2 * sp.integrate(psi_1**2, (Xi, -1, 1))
    ke22 = h/2 * sp.integrate(JG * 2/h * dpsi_2 * 2/h * dpsi_2, (Xi, -1, 1)) + beta * h / 2 * sp.integrate(psi_1**2, (Xi, -1, 1))
    # assembly of global stiffness K
    K_Total = np.zeros((N + 1, N + 1))
    for i in range(N):
        K_Total[i, i] += ke11
        K_Total[i, i+1] += ke12
        K_Total[i+1, i] += ke21
        K_Total[i+1, i+1] += ke22

    # Define symbolic variables
    unknown_values = sp.symbols('unknown_values0:%d' % (N - 1))
    # Construct the full theta array
    theta = [alpha_0_rad] + list(unknown_values) + [alpha_0_rad]
      
    
    # Define a list to store equations
    eqn = [None] * N
    
    # Calculate the equations for each row of K_Total
    # K_Total = sp.Matrix()  # Replace this with your actual K_Total matrix
    for i in range(N):
        K_row = K_Total[i]
        # Create an equation with a mix of numerical and symbolic data
        equation = sum(K_row[j] * theta[j] for j in range(len(theta)))
        eqn[i] = sp.Eq(equation, 0)
    
    # Solve for unknown_values
    solutions = sp.solve(eqn[1:N], unknown_values)
    
    # Reconstruct the theta array with the solved values
    theta = [alpha_0_rad] + [solutions[value] for value in unknown_values] + [alpha_0_rad]
    theta_values.append(np.array(theta).astype(np.float64))
    
    # Calculate lift distribution for this velocity
    lift_distribution = np.array([q_inf * chord * clalpha * t for t in theta]).astype(np.float64)
    lift_distributions.append(lift_distribution)
    
    # Calculate drag distribution for this velocity
    drag_distro_elastic = np.array([q_inf * chord * (cd_0 + (clalpha * t)**2 / np.pi / e_oswald / AR) for t in theta])
    drag_distributions.append(drag_distro_elastic)
    
    # Calculate and store area under curves
    lift_area = np.trapz(lift_distribution, np.linspace(0, span, N + 1))
    lift_areas.append(lift_area)
    drag_area = np.trapz(drag_distro_elastic,np.linspace(0, span, N + 1))
    drag_areas.append(drag_area)

if __name__ == '__main__':
    velocities = range(0,101,10)
    for velocity in velocities:
        FEM_module(velocity)
    plt.plot(velocities,lift_areas,label = 'Downforce')
    plt.plot(velocities,drag_areas,label = 'Drag')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Aerodynamic Force (N)')
    plt.title('Aerodynamic Forces on Rear Wings')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

#def ForcingFunction()
    ##Power for initial condition
    #if velocity != 0:
        #p_e = (velocity*s_f/(745.7*t_e))*(lift_area+0.5*m*g)
    ##Power limiter
    #if p_e > p_max:
        #p_e = p_max
    #optimal_power.append(p_e)
    ##Initial acceleration based on friction force without lift inserted
    #if velocity == 0:
        #acceleration=(1/m)*0.5*s_f*(0.5*m*g)
    #else:
        #acceleration = (1/m)*((745.7*p_e*t_e/velocity)-0.5*(c_dc*A_c*air_density*(velocity**2))-drag_area-c_rr*(lift_area + m*g))
    #return acceleration
    
# if __name__ == '__main__':
#     velocity = [0]
#     distance = [0]
#     acceleration = []
#     max_distance = 1000
#     while distance[-1] <= max_distance
#         FEM_module(velocity)
#         acceleration.append(ForcingFunction())
#         velocity.append(acceleration * time_step)
#         distance.append(velocity * time_step)
    
#     # output/gui
#     plt.plot(velocities,lift_areas,label = 'Downforce')
#     plt.plot(velocities,drag_areas,label = 'Drag')
#     plt.xlabel('Velocity (m/s)')
#     plt.ylabel('Aerodynamic Force (N)')
#     plt.title('Aerodynamic Forces on Rear Wings')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
