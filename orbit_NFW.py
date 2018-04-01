# orbital motion in a dark matter halo
# Modified from M. Zingale for a simple sun-earth problem

from __future__ import print_function

import math
import numpy as np
from potentials import *
import sys

# global parameters
#GM = 4.0*math.pi**2  #(assuming M = 1 solar mass)
G=6.6743 * 1.e-8
Msun=1.98e33
GM=G*Msun
year= np.pi*1e7 # year in seconds

class OrbitHistory(object):
    """ a simple container to store the integrated history of an
        orbit """
    
    def __init__(self):
        self.t = None
        self.x = None
        self.y = None
        self.u = None
        self.v = None
        self.r= None

    def finalR(self):
        """ the radius at the final integration time """
        N = len(self.t)
        return math.sqrt(self.x[N-1]**2 + self.y[N-1]**2)
    

    def displacement(self):
        """ distance between the starting and ending point """
        N = len(self.t)
        return math.sqrt( (self.x[0] - self.x[N-1])**2 +
                          (self.y[0] - self.y[N-1])**2 )


class Orbit(object):
    """ hold the initial conditions of a planet/comet/etc. orbiting
        the Sun and integrate """
    
    def __init__(self, a, e,Msat,Mdisk,cc,g_f,z,R_e_boost=1):
        """ a = semi-major axis (in AU),
            e = eccentricity """

        self.x0 = 0.0          # start at x = 0 by definition
        self.y0 = a*(1.0 - e)  # start at perihelion

        self.a = a
        self.e = e
        self.Msat=Msat
        self.Mdisk=Mdisk
        self.cc=cc
        self.g_f=g_f
        self.R_e_boost=R_e_boost
        self.z=z

        # perihelion velocity (see C&O Eq. 2.33 for ex)
        #self.u0 = -math.sqrt( (GM/a)* (1.0 + e) / (1.0 - e) )
        #self.u0= (G*MenclosedDM(Mvir,10,a)/a)**.5 *2	
        #Mgas=g_f/(1.-g_f)*Mencloseddisk(a,Msat,Mdisk)
        Mgas=Menclosedgas(a,Msat,Mdisk,z,g_f,R_e_boost)
        Mtot=Mencloseddisk(a,Msat,Mdisk,z,R_e_boost)+MenclosedDM(Msat,cc,a,z)+Mgas
        self.u0= (G*Mtot/a)**.5 	
        self.v0 = 0.0

    def kepler_period(self):
        """ return the period of the orbit in yr """
        return math.sqrt(self.a**3)


    def circular_velocity(self):
        """ return the circular velocity (in AU/yr) corresponding to
            the initial radius -- assuming a circle """
        return math.sqrt(GM/self.a)


    def escape_velocity(self):
        """ return the escape velocity (in AU/yr) corresponding to
            the initial radius -- assuming a circle """
        return math.sqrt(2.0*GM/self.a)
        
    def int_RK4(self, dt, tmax):
        """ integrate the equations of motion using 4th order R-K
            method.  """

        # initial conditions
        t = 0.0
        x = self.x0
        y = self.y0
        u = self.u0
        v = self.v0
        r = (self.x0**2 + self.y0**2)**.5
        # store the history for plotting
        tpoints = [t]
        xpoints = [x]
        ypoints = [y]
        upoints = [u]
        vpoints = [v]
        rpoints=  [r]	

        while (t < tmax):

            # make sure that the next step doesn't take us past where
            # we want to be, because of roundoff
            if t+dt > tmax:
                dt = tmax-t            
            
            # get the RHS at several points
            xdot1, ydot1, udot1, vdot1 = self.rhs([x,y], [u,v],t)

            xdot2, ydot2, udot2, vdot2 = \
                self.rhs([x+0.5*dt*xdot1,y+0.5*dt*ydot1], 
                         [u+0.5*dt*udot1,v+0.5*dt*vdot1],t)

            xdot3, ydot3, udot3, vdot3 = \
                self.rhs([x+0.5*dt*xdot2,y+0.5*dt*ydot2], 
                         [u+0.5*dt*udot2,v+0.5*dt*vdot2],t)

            xdot4, ydot4, udot4, vdot4 = \
                self.rhs([x+dt*xdot3,y+dt*ydot3], 
                         [u+dt*udot3,v+dt*vdot3],t)
            

            # advance
            unew = u + (dt/6.0)*(udot1 + 2.0*udot2 + 2.0*udot3 + udot4)
            vnew = v + (dt/6.0)*(vdot1 + 2.0*vdot2 + 2.0*vdot3 + vdot4)
            
            xnew = x + (dt/6.0)*(xdot1 + 2.0*xdot2 + 2.0*xdot3 + xdot4)
            ynew = y + (dt/6.0)*(ydot1 + 2.0*ydot2 + 2.0*ydot3 + ydot4)


            rnew=(xnew**2+ynew**2)**.5
            t += dt

            # store
            tpoints.append(t)
            xpoints.append(xnew)
            ypoints.append(ynew)
            upoints.append(unew)
            vpoints.append(vnew)
            rpoints.append(rnew)

            # set for the next step
            x = xnew; y = ynew; u = unew; v = vnew; r=rnew

        # return a orbitHistory object with the trajectory
        H = OrbitHistory()
        H.t = np.array(tpoints)
        H.x = np.array(xpoints)
        H.y = np.array(ypoints)
        H.u = np.array(upoints)
        H.v = np.array(vpoints)
        H.r = np.array(rpoints)
        return H


    def rhs(self, X, V,t):
        """ RHS of the equations of motion.  X is the input coordinate
            vector and V is the input velocity vector """ 
        # current radius
        r = math.sqrt(X[0]**2 + X[1]**2)

	# position
        xdot = V[0]
        ydot = V[1]
        M=MenclosedDM(self.Msat,self.cc,r,self.z)+Mencloseddisk(r,self.Msat,self.Mdisk,self.z,self.R_e_boost)
        udot = -G*M*X[0]/r**3
        vdot = -G*M*X[1]/r**3
        return xdot, ydot, udot, vdot




