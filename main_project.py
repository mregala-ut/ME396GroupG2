#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 

@author: Matthew Regala, Ashton Siegel, Aaron Park
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib import cm

# User Defined
alpha_0_deg = 10
velocity_initial = 0
clalpha_defined = 2 * np.pi
cd_0 = 0.008
chord = 0.30  # m
span = 1.00  # m
# max_distance = 1000 # m
airfoil = 'NACA0012.txt'
airfoil_center = (.25,0)

m = 798 # mass kg
d_w = 3.6 # wheelbasse m
r = 0.23 # tire radius m
p_e = 0 # engine brake horsepower Hp
p_max = 850 # maximum brake horsepower Hp
t_e = 0.98 # transmission efficiency
s_f = 1.7 # static friction coefficient between tire and asphalt
g = 9.81 # gravity m / s^2
c_dc = 0.85 # car drag coefficient
A_c = 1.30 # frontal area of car m^2
c_rr = 0.012 # rolling resistance coefficient

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

theta_values = [alpha_0_rad*np.ones(N+1)]
q_inf_values = [0]
lift_distributions = [0*np.ones(N+1)]
drag_distributions = [0*np.ones(N+1)]
lift_areas = [0]
drag_areas = [0]
optimal_power = [0]

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

# if __name__ == '__main__':
#     velocities = range(0,101,10)
#     for velocity in velocities:
#         FEM_module(velocity)
#     plt.plot(velocities,lift_areas,label = 'Downforce')
#     plt.plot(velocities,drag_areas,label = 'Drag')
#     plt.xlabel('Velocity (m/s)')
#     plt.ylabel('Aerodynamic Force (N)')
#     plt.title('Aerodynamic Forces on Rear Wings')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

def ForcingFunction(velocity,p_e):
    #Power for initial condition
    if velocity != 0:
        p_e = (velocity*s_f/(745.7*t_e))*(lift_areas[-1]+0.5*m*g)
    #Power limiter
    if p_e > p_max:
        p_e = p_max
    optimal_power.append(p_e)
    #Initial acceleration based on friction force without lift inserted
    if velocity == 0:
        acceleration=(1/m)*0.5*s_f*(0.5*m*g)
    else:
        acceleration = (1/m)*((745.7*p_e*t_e/velocity)-0.5*(c_dc*A_c*air_density*(velocity**2))-drag_areas[-1]-c_rr*(lift_areas[-1] + m*g))
    return acceleration

def output():
    vals = []
    for line in open(airfoil):
        n = line.strip().split(' ')  # intake and split by csv
        # add x,y to list as tuple. crop \n from y
        vals.append((float(n[0]), float(n[-1][:-1])))
    
    # delta_angle = [theta - theta_values[0] for theta in theta_values]
    init_angle = alpha_0_rad

    def toPlot(angle):  #init angle, delta_angle for one time slice      
        # calculation and output
        x = np.linspace(-span/2/chord,span/2/chord,len(angle)) # point coordinates
        y = [num[0] for num in vals]
        z = [num[1] for num in vals]
        slope = np.zeros((len(angle),1)) # slope determines strain
        for i in range(len(slope)-2):
            slope[i+1] = (angle[i+2]-angle[i])/(x[i+2]-x[i])
        slope[0] = (angle[1]-angle[0])/(x[1]-x[0])
        slope[-1] = (angle[-1]-angle[-2])/(x[-1]-x[-2])
        
        
        def angle_transform(x,y,section_angle,airfoil_center,slope):
            from numpy import cos, sin
            def transform(x,y,point_angle):
                theta = point_angle*-1 #np.deg2rad(angle)*-1
                rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
                a = np.array([x,y])
                a_ = np.dot(rot,a)
                return a_[0],a_[1]
            
            x0 = x - airfoil_center[0] # coordinate shift
            y0 = y - airfoil_center[1]
            r = np.sqrt(x0**2+y0**2) # radius from rotation axis
            
            x1,y1 = transform(x0,y0,init_angle*-1) # initial
            x2,y2 = transform(x0,y0,section_angle*-1) # final
            
            xout = x2 + airfoil_center[0] # coordinate shift
            yout = y2 + airfoil_center[1]
            
            
            displacement = np.sqrt((y2-y1)**2+(x2-x1)**2)
            strain = r*slope[0]
            
            return xout,yout,displacement,strain
        
        
        n = len(x)
        m = len(y)
        xp = np.zeros((n,m))
        yp = np.zeros((n,m))
        zp = np.zeros((n,m))
        dp = np.zeros((n,m))
        sp = np.zeros((n,m))
        for i in range(n): # for each airfoil section:
            for j in range(m): # for each airfoil coordinate point, 
                # add (x,y',z',displacement color,strain color)
                y_,z_,d_,s_ = angle_transform(y[j],z[j],angle[i],airfoil_center,slope[i])
                
                xp[i][j] = x[i]
                yp[i][j] = y_
                zp[i][j] = z_
                dp[i][j] = np.abs(d_)
                sp[i][j] = np.abs(s_)
        return xp,yp,zp,dp,sp
    
    xp = []
    yp = []
    zp = []
    dp = []
    sp = []
    for theta in theta_values:
        xt,yt,zt,dt,st = toPlot(theta)
        xp.append(xt)
        yp.append(yt)
        zp.append(zt)
        dp.append(dt)
        sp.append(st)
    del xt
    del yt
    del zt
    del dt
    del st
    

    
    def update_dp(i,zp,plot):

        # colormap
        norm1 = Normalize(vmin=0,vmax=dp[-1].max())
        cmap1 = plt.get_cmap('jet')
        colors1 = cmap1(norm1(dp[i]))
        plot[0].remove()
        plot[0] = ax1.plot_surface(xp[i],yp[i],zp[i],facecolors=colors1)
        # lift.set_data(xp[i],0.25*np.ones(len(xp[i])),lift_distributions[0])
        ax1.axis('equal')
        # mappable = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
        # mappable.set_array([])  # This line is needed to make the colorbar work
        # colorbar = plt.colorbar(mappable, label='displacement')
        ax1.set_title("Diaplacement")
        ax1.set_xlabel('span')
        ax1.set_ylabel('chord')
        ax1.set_zlabel('height')

        
    # def update_sp(i,zp,plot):
    #     # colormap
    #     norm2 = Normalize(vmin=0,vmax=sp[-1].max())
    #     cmap2 = plt.get_cmap('jet')
    #     colors2 = cmap2(norm2(sp[i]))
    #     plot[0].remove()
    #     plot[0] = ax2.plot_surface(xp[i],yp[i],zp[i],facecolors=colors2)
    #     ax2.axis('equal')
    #     # mappable = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
    #     # mappable.set_array([])  # This line is needed to make the colorbar work
    #     # colorbar = plt.colorbar(mappable, label='displacement')
    #     ax2.set_title("Strain")
    #     ax2.set_xlabel('span')
    #     ax2.set_ylabel('chord')
    #     ax2.set_zlabel('height')
        
        
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    plot1 = [ax1.plot_surface(xp[0],yp[0],zp[0])]
    # lift, = ax1.plot(xp[0],0.25*np.ones(len(xp[0])),lift_distributions[0],color = 'black')
    anim = FuncAnimation(fig1, update_dp, fargs=(zp,plot1), frames=np.arange(0,len(theta_values)), interval=250,init_func=lambda: None)
    anim.save('NACA0012_displacement_main.gif', dpi=80, writer='pillow')
    
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(projection='3d')
    # plot2 = [ax2.plot_surface(xp[0],yp[0],zp[0])]
    # anim = FuncAnimation(fig2, update_sp, fargs=(zp,plot2), frames=np.arange(0,len(theta_values)), interval=250,init_func=lambda: None)
    # anim.save('NACA0012_strain_main.gif', dpi=80, writer='pillow')
    

    
if __name__ == '__main__':
    time = [0]
    velocity = [velocity_initial]
    distance = [0]
    acceleration = [0]
    max_distance = 1000
    time_step = 1 # s
    while distance[-1] <= max_distance:
        time.append(time[-1] + time_step)
        FEM_module(velocity[-1])
        acceleration.append(ForcingFunction(velocity[-1],p_e))
        velocity.append(velocity[-1] + acceleration[-1] * time_step)
        distance.append(distance[-1] + velocity[-1] * time_step)
    
    # output/gui
    plt.plot(velocity,lift_areas,label = 'Downforce')
    plt.plot(velocity,drag_areas,label = 'Drag')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Aerodynamic Force (N)')
    plt.title('Aerodynamic Forces on Rear Wings')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    fig1,axs = plt.subplots(3)
    fig1.suptitle('Vehicle Motion')
    axs[0].plot(time,distance,label='Distance')
    axs[0].set(ylabel='Distance (m)')
    axs[1].plot(time,velocity,label='Speed')
    axs[1].set(ylabel='Speed (m/s)')
    axs[2].plot(time,acceleration,label='Acceleration')
    axs[2].set(xlabel='time',ylabel='Acceleration (m/s/s)')
    
    output()
