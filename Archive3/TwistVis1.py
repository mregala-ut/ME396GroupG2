# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:12:37 2023

@author: Aaron
"""

import numpy as np
import matplotlib.pyplot as plt

# inputs - vals, span, chord airfoil_center, init_angle, delta_angle
vals = []
for line in open('NACA0012.txt'):
    n = line.strip().split(' ')  # intake and split by csv
    # add x,y to list as tuple. crop \n from y
    vals.append((float(n[0]), float(n[-1][:-1])))
span = 1.01 # m
chord = 0.5
airfoil_center = (0.25,0)

init_angle = -10
x = np.linspace(-span/2/chord,span/2/chord)
delta_angle = np.cos(x*np.pi/span*chord)*10 # nose up = positive
del x



# calculation and output
x = np.linspace(-span/2/chord,span/2/chord,len(delta_angle)) # point coordinates
y = [num[0] for num in vals]
z = [num[1] for num in vals]
slope = np.zeros((len(delta_angle),1)) # slope determines strain
for i in range(len(slope)-2):
    slope[i+1] = (delta_angle[i+2]-delta_angle[i])/(x[i+2]-x[i])
slope[0] = (delta_angle[1]-delta_angle[0])/(x[1]-x[0])
slope[-1] = (delta_angle[-1]-delta_angle[-2])/(x[-1]-x[-2])


def angle_transform(x,y,init_angle,deflect_angle,airfoil_center,slope):
    from numpy import cos, sin
    def transform(x,y,angle):
        theta = np.deg2rad(angle)*-1
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        a = np.array([x,y])
        a_ = np.dot(rot,a)
        return a_[0],a_[1]
    
    x0 = x - airfoil_center[0] # coordinate shift
    y0 = y - airfoil_center[1]
    r = np.sqrt(x0**2+y0**2) # radius from rotation axis
    
    x1,y1 = transform(x0,y0,init_angle)
    x2,y2 = transform(x1,y1,deflect_angle)
    
    xout = x2 + airfoil_center[0]
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
        y_,z_,d_,s_ = angle_transform(y[j],z[j],init_angle,delta_angle[i],airfoil_center,slope[i])
        
        xp[i][j] = x[i]
        yp[i][j] = y_
        zp[i][j] = z_
        dp[i][j] = np.abs(d_)
        sp[i][j] = np.abs(s_)


# colormap
from matplotlib.colors import Normalize
norm1 = Normalize(vmin=dp.min(),vmax=dp.max())
cmap1 = plt.get_cmap('jet')
colors1 = cmap1(norm1(dp))
norm2 = Normalize(vmin=sp.min(),vmax=sp.max())
cmap2 = plt.get_cmap('jet')
colors2 = cmap2(norm2(sp))


# plot - displacement color map
from matplotlib import cm
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.plot_surface(xp,yp,zp,facecolors=colors1)
ax1.axis('equal')
mappable = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
mappable.set_array([])  # This line is needed to make the colorbar work
colorbar = plt.colorbar(mappable, label='displacement')
ax1.set_title("Wing")
ax1.set_xlabel('span')
ax1.set_ylabel('chord')
ax1.set_zlabel('height')


# plot - strain color map
fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.plot_surface(xp,yp,zp,facecolors=colors2)
ax2.axis('equal')
mappable = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
mappable.set_array([])  # This line is needed to make the colorbar work
colorbar = plt.colorbar(mappable, label='strain')
ax2.set_title("Wing")
ax2.set_xlabel('span')
ax2.set_ylabel('chord')
ax2.set_zlabel('height')
