#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 2024

@author: Matthew Regala
"""

# Matthew Regala
# Code to Describe Aeroelastic Torsion of Elastic Wing

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


# Parameters
air_density = 1.293  # kg/m^3
chord = 0.30  # m
L = 1.00  # m
AR = L / chord  # Aspect Ratio
e = 0.20
velocities = list(range(0, 151, 5))

minimumJG = 1000 * (chord * 3/4 + e) * L / 2 / np.deg2rad(1)
JG = minimumJG

alpha_0_deg = 10
alpha_0_rad = np.deg2rad(10)

clalpha = 2 * np.pi * AR / (2 + AR)
cd_0 = 0.008
e_oswald = 0.7
alpha_values = alpha_0_rad
elastic_offset = chord * e  # m

N = 20  # Set a single N value
theta_values = []
h_values = []
q_inf_values = []
lift_distributions = []
lift_distro_rigid = []
drag_distributions = []
drag_distro_rigid = []
lift_areas_elastic = []
drag_areas_elastic = []
lift_areas_rigid = []
drag_areas_rigid = []

for velocity in velocities:
    q_inf = 1/2 * air_density * velocity**2
    q_inf_values.append(q_inf)
    beta = q_inf * clalpha * elastic_offset

    # xi (master element coordinate)
    # x_i: nodal coordinate in physical space - left element end
    # x_j: nodal coordinate in physical space - right element end
    Xi, x, x_i, x_j = sp.symbols('Xi x x_i x_j')
    # element length
    h = 1 / N
    # coordinate map
    x = 1/2 * (Xi * h + x_i + x_j)

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
    lift_distro_elastic = np.array([q_inf * chord * clalpha * t for t in theta]).astype(np.float64)
    lift_distributions.append(lift_distro_elastic)
    lift_distro_rigid.append([q_inf * chord * clalpha * alpha_0_rad]*(N+1))
    
    # Calculate drag distribution for this velocity
    drag_distro_elastic = np.array([q_inf * chord * (cd_0 + (clalpha * t)**2 / np.pi / e_oswald / AR) for t in theta])
    drag_distributions.append(drag_distro_elastic)
    drag_distro_rigid.append([1/2*air_density*velocity**2*chord*(cd_0+(clalpha*np.deg2rad(10))**2/np.pi/e_oswald/AR)] * (N+1))
    
    # Calculate and store area under curves
    lift_areas_elastic.append(np.trapz(lift_distro_elastic, np.linspace(0, L, N + 1)))
    drag_areas_elastic.append(np.trapz(drag_distro_elastic,np.linspace(0, L, N + 1)))
    lift_areas_rigid.append(q_inf * chord * L * clalpha * alpha_0_rad)
    drag = [L * 1/2*air_density*velocity**2*chord*(cd_0+(clalpha*np.deg2rad(10))**2/np.pi/e_oswald/AR)]
    drag_areas_rigid.append(drag)

## Plot Force Distributions
    
plt.figure()
plt.plot(velocities,lift_areas_rigid,label='Total Downforce - Rigid',color='b')
plt.plot(velocities,lift_areas_elastic,'--',label='Total Downforce - Elastic',color='b')
plt.plot(velocities,drag_areas_rigid,label='Total Drag - Rigid',color='r')
plt.plot(velocities,drag_areas_elastic,'--',label='Total Drag - Elastic',color='r')
plt.grid(True, which='both')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Aerodynamic Force (N)')
plt.title('Aerodynamic Forces on Rear Wings')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# Plot Lift Distribution, Drag Distribution, and Angle of Attack
selected_velocities = [50, 75, 100, 125, 150]
colors = ['b', 'r', 'g', 'm', 'c']
x = np.linspace(0, L, N + 1)

# Lift Distribution
plt.figure()

for i, velocity in enumerate(velocities):
    if velocity in selected_velocities:
        plt.plot(x, lift_distributions[i], label=f'{velocity} m/s (Flexible)', linewidth=1.5, color=colors[selected_velocities.index(velocity)])
        plt.plot(x, lift_distro_rigid[i], '--', label=f'{velocity} m/s (Rigid)', linewidth=1.5, color=colors[selected_velocities.index(velocity)])

plt.grid(True, which='both')
plt.xlabel('Span of Rear Wing')
plt.ylabel('Lift Distribution (N/m)')
plt.title('Lift Distribution of a Rear Wing at Different Velocities')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Drag Distribution
plt.figure()

for i, velocity in enumerate(velocities):
    if velocity in selected_velocities:
        plt.plot(x, drag_distributions[i], label=f'{velocity} m/s (Flexible)', linewidth=1.5, color=colors[selected_velocities.index(velocity)])
        plt.plot(x, drag_distro_rigid[i], '--', label=f'{velocity} m/s (Rigid)', linewidth=1.5, color=colors[selected_velocities.index(velocity)])

plt.grid(True, which='both')
plt.xlabel('Span of Rear Wing')
plt.ylabel('Drag Distribution (N/m)')
plt.title('Drag Distribution of a Rear Wing at Different Velocities')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Angle of Attack
plt.figure()

for velocity, theta_distribution in zip(velocities, theta_values):
    if velocity in selected_velocities:
        x = np.linspace(0, L, len(theta_distribution))
        plt.plot(x, np.rad2deg(theta_distribution), label=f'{velocity} m/s')

plt.grid(True, which='both')
plt.xlabel('Span of Rear Wing (m)')
plt.ylabel('Angle of Attack (degrees)')
plt.title('Angle of Attack for a Flexible Rear Wing at Different Velocities')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

## Plot Force Distributions
    
plt.figure()
plt.plot(velocities,lift_areas_rigid,label='Total Downforce - Rigid',color='b')
plt.plot(velocities,lift_areas_elastic,'--',label='Total Downforce - Elastic',color='b')
plt.plot(velocities,drag_areas_rigid,label='Total Drag - Rigid',color='r')
plt.plot(velocities,drag_areas_elastic,'--',label='Total Drag - Elastic',color='r')
plt.grid(True, which='both')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Aerodynamic Force (N)')
plt.title('Aerodynamic Forces on Rear Wings')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()