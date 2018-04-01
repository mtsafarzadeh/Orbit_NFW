# compare different ODE methods on the circular orbit problem
#
# M. Zingale (2013-02-19)

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import pylab as plt
from orbit_NFW import *
from potentials import *
import sys


fig,ax=plt.subplots(1,1)
# mark the Sun
#ax.scatter([0], [0], s=250, marker=(5,1), color="k")
#ax.scatter([0], [0], s=200, marker=(5,1), color="k")
# circular orbit
AU=8*3e10*60 #8 light-minutes
year= np.pi*1e7 # year in seconds
pc = 3.e18
kpc= 1e3 * pc
dt=1e6*year
M_abs_sun=4.83
nbins=5
time_averaged=np.zeros(nbins)
radial_bins=np.logspace(19,23,nbins)/kpc
radial_bins=np.linspace(0.001,0.01,nbins)
radial_bins=[0.05]
Msat=1e9
redshift=1.
g_f=0.96
Mdisk=estimate_Mstar(Msat,redshift)
cc=find_cc(Msat,redshift)
color_list=['b','r']
pp=0
for ii in radial_bins:
	o=Orbit(ii*kpc,0,Msat,Mdisk,cc,g_f,redshift)
	#hist_RK4 = o.int_RK4(dt, 5e10*year) # dt, tmax
	hist_RK4 = o.int_RK4(1e5*year, 1e8*year) # dt, tmax
	ax.plot(hist_RK4.x/kpc, hist_RK4.y/kpc, color=color_list[pp],label=r'$\rm r_{init}=%.0f\,kpc$'%ii)
	time_averaged[pp]=np.sum(hist_RK4.r*hist_RK4.t)/np.sum(hist_RK4.t)
	circ=plt.Circle((0,0), radius=ii, color=color_list[pp], fill=False,lw=1,ls='dashed')
	#circ=plt.Circle((0,0), radius=time_averaged[pp]/kpc, color='k', fill=False,lw=1,ls=dashed)
	ax.add_patch(circ)
	print(ii,time_averaged[pp]/kpc)
	mean_distance=np.mean(hist_RK4.r)
	#circ=plt.Circle((0,0), radius=mean_distance/kpc, color='r', fill=False,lw=1)
	#ax.add_patch(circ)
	pp+=1

leg = plt.legend(frameon=False,loc=4)
ltext = leg.get_texts()
plt.setp(ltext, fontsize=24)

max_distance=0.06
ax.set_xlim(-max_distance,max_distance)
ax.set_ylim(-max_distance,max_distance)

ax.set_xlabel(r'$\rm kpc$',fontsize=24)
ax.set_ylabel(r'$\rm kpc$',fontsize=24)
for tick in ax.xaxis.get_major_ticks():
	tick.label.set_fontsize(16)
for tick in ax.yaxis.get_major_ticks():
	tick.label.set_fontsize(16)

#ax.text(0.1,0.9,r'$\rm M_{sat}=%.0e \,M_{\odot},\,$'%Msat+r'$\rm f_{gas}=%.2f$'%g_f,transform=ax.transAxes, fontsize=24)
#ax = plt.gca()
#ax.set_aspect("equal", "datalim")
#plt.savefig('orbit_gf_%.2f'%g_f+'_Mvir_%.1e.pdf'%Msat)
plt.tight_layout()
plt.savefig('tesing.pdf')
