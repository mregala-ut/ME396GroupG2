# Car Performance Modeling with an Elastic Rear Wing
By: Aaron Park, Matthew Regala, Ashton Siegel

This projects intended goal is to evaluate and represent the performance of a race car operating with an elastic rear wing.

To achieve this, the code has four main modules:

1) Input GUI developed in QT Creator. This GUI allows modification of the characteristics of the rear wing, the car, and simulation parameters:
[INSERT TABLE OR GUI EXAMPLE]
  
2) A finite element method (FEM) module to evaluate angular deflections due to moments created by a lift, offset from an elastic axis. This module uses a Bubnov–Galerkin approach to solve for deflections. The differential equation for the problem is: <br>  
$GJ\theta'' + q_{\infty }cec_{l\alpha }\theta = 0$ <br>  
This assumes rigid attachment to endplates at an initial angle of attack, $\theta_0$. <br>
Using compactly supported linear basis functions, the FEM module develops stiffness matrices for the structural and aerodynamic components. The summation of these stiffnesses evolves the form: <br>  
$[K]_{total} \theta = 0$ <br>  
Using sympy, the unknown $\theta$ values are solved for. From this, lift and drag distributions as well as the total lift and drag can be calculated and returned.

3) The performance module uses the car parameters, lift, drag, and velocities to determine the maximum engine power that can be utilized without tire slippage. After doing so, the net forces acting on the car can be determined and instantaneous accelerations returned. This acceleration is used to update the velocity and position of the vehicle.

After finding the updated velocity, the program loops through time steps until the desired time or position is met.


4) Output Figure Module: With all of the kinematic and force data calculated for each time step, this module creates visualizations. This includes graphs of distance, speed, acceleration, lift, drag, and optimal power over time. {br}
   Additionally, using an input NACA0012 surface, force distributions and deflections/strains of the wing are visualized in a 3-dimensional space in an animation. {br}
   <img src= "https://github.com/mregala-ut/ME396GroupG2/blob/main/NACA0012_main_displacement1.gif" width=40% height=40% align="center">
   <img src= "https://github.com/mregala-ut/ME396GroupG2/blob/main/NACA0012_main_displacement1.gif" width=40% height=40% align="center">
   

A high level view of the architecture of the integration between the modules can be seen below: {br}  

![Project Interface Chart](https://github.com/mregala-ut/ME396GroupG2/blob/main/Project_InterfaceChart.png?raw=true)
