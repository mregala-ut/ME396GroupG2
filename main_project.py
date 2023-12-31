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
import glob
import shutil


def FEM_module(velocity):
    '''
    

    Parameters
    ----------
    velocity : numeric
        This function takes an input velocity, and calculates the resultant
        angle of attacks, lift and drag distributions, and total lift and drag.
        These values are appended to theta_values,lift_distributions,
        drag_distributions, lift_areas, drag_areas.

    Returns
    -------
    None.

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
    for i in range(N):
        K_row = K_Total[i]
        equation = sum(K_row[j] * theta[j] for j in range(len(theta)))
        eqn[i] = sp.Eq(equation, 0)
    
    # Solve for unknown_values and reconstruct theta array
    solutions = sp.solve(eqn[1:N], unknown_values)
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

def prePlot(): # convert twist data into plottable points for a wing
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
        
        # function to rotate inputted airfoil for plot
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
    
    return xp,yp,zp,dp,sp
    
def Plots():
    
    # plot index finder
    fnames = []
    fnames.append(glob.glob('NACA0012_main_strain*.gif'))
    fnames.append(glob.glob('NACA0012_main_displacement*.gif'))
    fnames.append(glob.glob('NACA0012_main_data*.png'))
    # fnames.append(glob.glob('NACA0012_displacement*.gif'))
    
    # create new plots with new index number
    plot_index = []
    for names in fnames: # for each plot
        if len(names)==0:
            names = [-1]
        else:
            for i in range(len(names)): # for each instance of a plot
                if names[i][-5:-4].isdigit():
                    names[i] = int(names[i][-5:-4])
                else:
                    names[i] = -1
        plot_index.append(max(names)+1)
    
    # displacement gif update function
    def update_dp(i,zp,plot):
        # colormap
        norm1 = Normalize(vmin=0,vmax=dp[-1].max())
        cmap1 = plt.get_cmap('jet')
        colors1 = cmap1(norm1(dp[i]))
        plot[0].remove()
        plot[0] = ax1.plot_surface(xp[i],yp[i],zp[i],facecolors=colors1,label = 'wing')
        
        # lift (offset from origin/axes for visuals)
        lift.set_data(xp[i].T[0], 0.25*np.ones(len(xp[i].T[0])))
        if all(lift_distributions[i]==0):
            lift.set_3d_properties(lift_distributions[i]+.25)
        else:
            lift.set_3d_properties((lift_distributions[i]/lift_distributions[-1].max())**5+.25)
        
        # drag (offset from origin/axes for visuals)
        drag_temp = drag_distributions[i]
        if all(drag_distributions[i]==0):
            drag_temp = drag_distributions[i]+1.125
        else:
            drag_temp = (drag_distributions[i]/drag_distributions[-1].max())**5+1.125
        drag.set_data(xp[i].T[0], drag_temp)
        drag.set_3d_properties(0*np.ones(len(xp[i].T[0])))
        
        ax1.axis('equal')
        ax1.set_title(f"Displacement \ntime: {time[i]:.1f} seconds, distance: {distance[i]:.1f}m, velocity: {velocity[i]:.1f}m/s")
        ax1.set_xlabel('span')
        ax1.set_ylabel('chord')
        ax1.set_zlabel('height')        
    
    points = len(time)
    GIF_duration = 10
    GIF_interval = max(100,GIF_duration*1000//points)

    
    print('Plotting Figure 1....')    
    fig1 = plt.figure() # figure
    ax1 = fig1.add_subplot(projection='3d') # 3d plot
    ax1.view_init(elev=10, azim=-15, roll=0) # set viewpoint
    
    norm1 = Normalize(vmin=0,vmax=dp[-1].max()) # set color scheme
    cmap1 = plt.get_cmap('jet')
    mappable = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
    # mappable.set_array([])  # This line is needed to make the colorbar work
    colorbar = plt.colorbar(mappable, label='displacement',ax=ax1)
    
    plot1 = [ax1.plot_surface(xp[0],yp[0],zp[0],label='wing')] # 3d plot
    lift, = ax1.plot(xp[0].T[0],0.25*np.ones(len(xp[0].T[0])),lift_distributions[0],color = 'green',label = 'lift') # lift line
    drag, = ax1.plot(xp[0].T[0],drag_distributions[0],0*np.ones(len(xp[0].T[0])),color = 'red',label = 'lift') # drag line
    anim = FuncAnimation(fig1, update_dp, fargs=(zp,plot1), frames=np.arange(0,len(theta_values)), interval=GIF_interval,init_func=lambda: None)
    anim.save(f'NACA0012_main_displacement{plot_index[0]}.gif', dpi=dpi, writer='pillow')
    
    # strain gif update function
    def update_sp(i,zp,plot):

        # colormap
        norm1 = Normalize(vmin=0,vmax=sp[-1].max())
        cmap1 = plt.get_cmap('jet')
        colors1 = cmap1(norm1(sp[i]))
        plot[0].remove()
        plot[0] = ax1.plot_surface(xp[i],yp[i],zp[i],facecolors=colors1,label = 'wing')
        
        # lift (offset from origin/axes for visuals)
        lift.set_data(xp[i].T[0], 0.25*np.ones(len(xp[i].T[0])))
        if all(lift_distributions[i]==0):
            lift.set_3d_properties(lift_distributions[i]+.25)
        else:
            lift.set_3d_properties((lift_distributions[i]/lift_distributions[-1].max())**5+.25)
        
        # drag (offset from origin/axes for visuals)
        drag_temp = drag_distributions[i]
        if all(drag_distributions[i]==0):
            drag_temp = drag_distributions[i]+1.125
        else:
            drag_temp = (drag_distributions[i]/drag_distributions[-1].max())**5+1.125
        drag.set_data(xp[i].T[0], drag_temp)
        drag.set_3d_properties(0*np.ones(len(xp[i].T[0])))
        
        ax1.axis('equal')
        ax1.set_title(f"Strain \ntime: {time[i]:.1f} seconds, distance: {distance[i]:.1f}m, velocity: {velocity[i]:.1f}m/s")
        ax1.set_xlabel('span')
        ax1.set_ylabel('chord')        
    
    print('Plotting Figure 2....')    
    fig1 = plt.figure() # figure
    ax1 = fig1.add_subplot(projection='3d') # 3d plot
    ax1.view_init(elev=10, azim=-15, roll=0) # viewpoint
    
    norm1 = Normalize(vmin=0,vmax=sp[-1].max()) # colormap
    cmap1 = plt.get_cmap('jet')
    mappable = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
    # mappable.set_array([])  # This line is needed to make the colorbar work
    colorbar = plt.colorbar(mappable, label='strain',ax=ax1)
    
    plot1 = [ax1.plot_surface(xp[0],yp[0],zp[0],label='wing')] # 3d plot
    lift, = ax1.plot(xp[0].T[0],0.25*np.ones(len(xp[0].T[0])),lift_distributions[0],color = 'green',label = 'lift') # lift
    drag, = ax1.plot(xp[0].T[0],drag_distributions[0],0*np.ones(len(xp[0].T[0])),color = 'red',label = 'lift') # drag
    anim = FuncAnimation(fig1, update_sp, fargs=(zp,plot1), frames=np.arange(0,len(theta_values)), interval=GIF_interval,init_func=lambda: None)
    anim.save(f'NACA0012_main_strain{plot_index[1]}.gif', dpi=dpi, writer='pillow')
    
    print('Plotting Figure 3....') # plot remaining data
    fig2,axs = plt.subplots(3,2)
    fig2.suptitle('Vehicle Motion')
    axs[0,0].plot(time,distance,label='Distance')
    axs[0,0].set(xlabel='time',ylabel='Distance (m)')
    axs[0,0].grid()
    axs[1,0].plot(time,velocity,label='Speed')
    axs[1,0].set(xlabel='time',ylabel='Speed (m/s)')
    axs[1,0].grid()
    axs[2,0].plot(time,acceleration,label='Acceleration')
    axs[2,0].set(xlabel='time',ylabel='Acceleration (m/s/s)')
    axs[2,0].grid()
    axs[0,1].plot(time,lift_areas,label='Lift')
    axs[0,1].set(xlabel='time',ylabel='Lift (N)')
    axs[0,1].grid()
    axs[1,1].plot(time,drag_areas,label='Drag')
    axs[1,1].set(xlabel='time',ylabel='Drag (N)')
    axs[1,1].grid()
    axs[2,1].plot(time,optimal_power,label='Optimal Power')
    axs[2,1].set(xlabel='time',ylabel='Optimal Power (Hp)')
    axs[2,1].grid()
    fig2.set_size_inches((15, 9))
    fig2.savefig(f'NACA0012_main_data{plot_index[2]}.png', dpi=dpi)

    
if __name__ == '__main__':
    
    # Open input gui
    exec(open("Sim_Inputs.py").read())
    print('Inputs Entered')
    
    # Make a copy of input file with a new index
    ParamText = glob.glob('Input_Parameters*.txt')
    file_index = []
    if len(ParamText)==0:
        ParamText = [-1]
    else:
        for i in range(len(ParamText)): # for each instance of a plot
            if ParamText[i][-5:-4].isdigit():
                ParamText[i] = int(ParamText[i][-5:-4])
            else:
                ParamText[i] = -1
    file_index.append(max(ParamText)+1)
    shutil.copy('Input_Parameters.txt', f'Input_Parameters{file_index[0]}.txt')

    # Read input gui output file
    param = []
    for line in open('Input_Parameters.txt'):
        param.append(eval(line))
    print('Inputs Read')
    
    # Set Parameters
    # User Defined
    alpha_0_deg = param[0] #10
    velocity_initial = 0
    clalpha_defined = 2 * np.pi
    cd_0 = param[1] #0.008
    chord = param[2] #0.30  # m
    span = param[3] #1.00  # m

    m = param[4] #798 # mass kg
    d_w = param[5] #3.6 # wheelbasse m
    r = param[6] #0.23 # tire radius m
    p_e = 0 # engine brake horsepower Hp
    p_max = param[7] #850 # maximum brake horsepower Hp
    t_e = param[8] #0.98 # transmission efficiency
    s_f = param[9] #1.7 # static friction coefficient between tire and asphalt
    g = 9.81 # gravity m / s^2
    c_dc = param[10] #0.85 # car drag coefficient
    A_c = param[11] #1.30 # frontal area of car m^2
    c_rr = param[12] #0.012 # rolling resistance coefficient
    N = param[13] #20 # number of elements
    time_step = param[14] #0.5 s 
    max_distance = param[15] #2000 m
    
    # Parameters & Calculations
    air_density = 1.293  # kg/m^3
    AR = span / chord  # Aspect Ratio
    e = 0.20
    clalpha = clalpha_defined * AR / (2 + AR)
    e = 0.20
    e_oswald = 0.7
    elastic_offset = chord * e  # m
    JG = 1000 * (chord * 3/4 + e) * span / 2 / np.deg2rad(1)
    alpha_0_rad = np.deg2rad(10)

    airfoil = 'NACA0012.txt'
    airfoil_center = (.45,0) # elastic axis
    dpi = 120

    # Initiate data capture lists
    theta_values = [alpha_0_rad*np.ones(N+1)]
    q_inf_values = [0]
    lift_distributions = [0*np.ones(N+1)]
    drag_distributions = [0*np.ones(N+1)]
    lift_areas = [0]
    drag_areas = [0]
    optimal_power = [0]     
    print('Parameters Configured')
    
    time = [0]
    velocity = [velocity_initial]
    distance = [0]
    acceleration = [0]
    print(f'Limit: Max Distance = {max_distance:.1f}m')
    while distance[-1] <= max_distance: # iteration loop
        time.append(time[-1] + time_step)
        FEM_module(velocity[-1])
        acceleration.append(ForcingFunction(velocity[-1],p_e))
        velocity.append(velocity[-1] + acceleration[-1] * time_step)
        distance.append(distance[-1] + velocity[-1] * time_step)
        print(f'Evaluated at time: {time[-1]:.1f}s, distance: {distance[-1]:.1f}m, velocity: {velocity[-1]:.1f}m/s, acceleration: {acceleration[-1]:.1f}m/s/s')
    
    # Print key data points
    print(f'Max Downforce: {max(lift_areas):.1f}N at {time[lift_areas.index(max(lift_areas))]:.1f}s')
    print(f'Max Downforce per unit weight: {max(lift_areas)/m:.1f}N/kg at {time[lift_areas.index(max(lift_areas))]:.1f}s')
    print(f'Max Drag: {max(drag_areas):.1f}N at {time[drag_areas.index(max(drag_areas))]:.1f}s')
    print(f'Max Drag per unit weight: {max(drag_areas)/m:.1f}N/kg at {time[drag_areas.index(max(drag_areas))]:.1f}s')
    print(f'Max acceleration: {max(acceleration):.1f}m/s/s at {time[acceleration.index(max(acceleration))]:.1f}s')
    if p_max in optimal_power:
        print(f'Max bHp Reached at {time[optimal_power.index(p_max)]:.1f}s')
    else:
        print('Max bHp not reached')
    print(f'Reached max distance {max_distance}m at {time[-1]}s')
    
    
    # Plot
    print('Calculations complete. Preparing data for figures.')
    xp,yp,zp,dp,sp = prePlot() 
    print('Ready to Plot')
    Plots()
    print('Plots Completed and Saved')
    print('Program Complete')
