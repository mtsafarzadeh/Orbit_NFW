import numpy as np
import scipy.integrate as integrate
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import sys
from scipy.interpolate import interp1d

behroozi=np.loadtxt('behroozi_z2.dat')
M_h=behroozi[:,1]
M_star=behroozi[:,0]

f = interp1d(M_h,M_star, kind='cubic')

Msun=1.9e33
pc = 3.e18
kpc= 1e3 * pc
Mpc=1e6*pc
km=1e5 #cm
rhocrit=9.2e-30 #gr/cm^3
OmegaLambda=0.73
OmegaM=0.27
G=6.67e-8 #cgs units
H0= 70 * km/Mpc
mp=1.67e-24
KB=1.38e-16 # CGS units
mu=0.6
def H(z):
	return (H0**2 *(OmegaM*(1+z)**3+OmegaLambda))**.5
def R200(Msat,z):
	rhocrit=3*H(z)**2/(8*np.pi*G)
	return  (3./(4.*3.1415)*Msat*1.98E33/(200*rhocrit))**(1./3.)
#Sigma0= 20* Msun/pc**2

def DiskProfile(r,Msat,Mdisk,z,R_e_boost=1):#,Sigma0):
	#Sigma0*=(Msun/pc**2)
	R_e=0.015*R200(Msat,z)*R_e_boost
	ScaleRadius=R_e/1.67
	#Sigma0=Msat*Msun/(2*np.pi*ScaleRadius**2) *0.01 # M_*=0.01 M_H
	Sigma0=Mdisk*Msun/(2*np.pi*ScaleRadius**2)  # M_*=0.01 M_H
	#Sigma0=10**f(np.log10(Msat))*Msun/(2*np.pi*ScaleRadius**2) # AM results Behroozi+13
	return Sigma0 * np.exp(-r/ScaleRadius)

def Gas_DiskProfile(r,Msat,Mdisk,z,f_gas,R_e_boost=1):
        R_e=0.015*R200(Msat,z)*R_e_boost
        ScaleRadius=R_e/1.67 *1.7#*2.6 #Kravtsov +13
        Sigma0=Mdisk*Msun/(2*np.pi*ScaleRadius**2)  # M_*=0.01 M_H
        return Sigma0 * np.exp(-r/ScaleRadius) * f_gas/(1-f_gas)

def g(cc):
	return (np.log(cc+1)-cc/(1+cc))**-1

def MenclosedDM(M,cc,r,z):
	return M*Msun*g(cc) *(np.log(1+cc * r/R200(M,z))-cc *r/R200(M,z)/(1+cc*r/R200(M,z)))

def find_rt(r_t,Msat,cc_sat,Mhost,cc_host,D,z):
	return (MenclosedDM(Msat,cc_sat,r_t*kpc,z)/(3*MenclosedDM(Mhost,cc_host,(D-r_t)*kpc,z)))**(1./3) *(D-r_t)-r_t

def find_rt2(r_t,Msat,cc_sat,Mhost,cc_host,D,z):
	return (MenclosedDM(Msat,cc_sat,r_t*kpc,z)/(3*MenclosedDM(Mhost,cc_host,(D)*kpc,z)))**(1./3) *(D)-r_t

def Mencloseddisk(r,Msat,Mdisk,z,R_e_boost=1):
	#disk_mass=integrate.quad(lambda x: 2*np.pi * x * DiskProfile(x,Msat,Sigma0),0, r)[0]
	disk_mass=integrate.quad(lambda x: 2*np.pi * x * DiskProfile(x,Msat,Mdisk,z,R_e_boost),0, r)[0]
	return disk_mass

def Menclosedgas(r,Msat,Mdisk,z,f_gas,R_e_boost=1):
        disk_mass=integrate.quad(lambda x: 2*np.pi * x * Gas_DiskProfile(x,Msat,Mdisk,z,f_gas,R_e_boost),0, r)[0]
        return disk_mass

def surface_brightness(rr,Msat,Mdisk,z,R_e_boost=1):
	M_abs_sun=4.83
	S_r=np.zeros(len(rr))
	dr=np.diff(rr)[0]
	i=0
	for r in rr:
		I_r=integrate.quad(lambda x: 2*np.pi * x * DiskProfile(x,Msat,Mdisk,z,R_e_boost),r, r+dr)[0]
		if r==0:A0=0
		A1= np.pi* (r+dr)**2
		A=A1-A0
		I_r/=A
		S_r[i]=M_abs_sun+21.572-2.5*np.log10(I_r/(Msun/pc**2))
		i+=1
		A0=A1
	return S_r
def SB(rr,Msat,Mdisk,z,R_e_boost=1):
	M_abs_sun=4.83
	R_e=0.015*R200(Msat,z)*R_e_boost
	ScaleRadius=R_e/1.67
	Sigma0=Mdisk*Msun/(2*np.pi*ScaleRadius**2)  # M_*=0.01 M_H
	SD_profile=Sigma0 * np.exp(-rr/ScaleRadius)
	SB_profile=M_abs_sun+21.572-2.5*np.log10(SD_profile/(Msun/pc**2))
	return SB_profile

def find_cc(Msat,redshift):
	a=0.52+(0.905-0.52)*np.exp(-0.617* redshift**1.21)
	b=-0.101+0.026 *redshift
	xx=a+b*np.log10(Msat/(1e12/.7))
	return 10**xx

def ff(x,alpha,gamma,delta):
	return -np.log10(10**(alpha*x)+1) + delta * (np.log10(1+np.exp(x)))**gamma /(1+np.exp(10**-x))
def estimate_Mstar(Mhalo,z):
	a=1./(1+z)
	nu=np.exp(-4* a**2)
	log_epsilon=-1.77-0.006*(a-1)-0.119*(a-1)
	logM1=11.512+(-1.793*(a-1)+(-0.25)*z)*nu
	alpha=-1.412+(0.73*(a-1))*nu
	delta=3.508+(2.608*(a-1)-0.043*z)*nu
	gamma=0.316+(1.319*(a-1)+0.279*z)*nu
	M1=10**logM1
	epsilon=10**log_epsilon
	#print(logM1+log_epsilon,logM1,alpha,delta,gamma)
	logMstar=np.log10(epsilon*M1) + ff(np.log10(Mhalo/M1),alpha,gamma,delta)-ff(0,alpha,gamma,delta)
	return 10**logMstar

def estimate_f_gas_MainSequence(Mstar,redshift):
	A=9.0*(1+redshift/0.017)**0.03
	B=1.1*(1+redshift)**-0.97
	return 1./(np.exp((np.log10(Mstar)-A)/B) + 1) 

def estimate_f_gas(Mstar,redshift):
	A=9.04*(1+redshift/1.76)**0.24
	B=0.53*(1+redshift)**-0.91
	return 1./(np.exp((np.log10(Mstar)-A)/B) + 1) 
def V_c(Mvir,z):
	return (G*Mvir*Msun/R200(Mvir,z))**.5

def t_dyn(Mvir,z):
	R_gas=0.015*R200(Mvir,z)
	return R_gas/V_c(Mvir,z)
def M_enc_tot(r,Msat,z,f_gas):
	Mdisk=estimate_Mstar(Msat,z)
	Mgas_enclosed=Menclosedgas(r,Msat,Mdisk,z,f_gas)
	Mdisk_enclosed=Mencloseddisk(r,Msat,Mdisk,z)
	cc=find_cc(Msat,z)
	M_dm_enclosed=MenclosedDM(Msat,cc,r,z)
	return M_dm_enclosed+Mdisk_enclosed+Mgas_enclosed
def epicycle(r,Msat,z,f_gas):
	V_c_r=(G*M_enc_tot(r,Msat,z,f_gas)/r)**.5
	dr=0.01*kpc
	rprime=r+dr #dr=0.01 kpc
	V_c_r_plus_dr=(G*M_enc_tot(rprime,Msat,z,f_gas)/rprime)**.5
	dVcdr=(V_c_r_plus_dr-V_c_r)/dr
	return (2*V_c_r/r*(V_c_r/r+dVcdr))**.5
def Toomre_Q(r,Msat,z,f_gas):
	Mdisk=estimate_Mstar(Msat,z)
	Sigma_Gas=Gas_DiskProfile(r,Msat,Mdisk,z,f_gas)
	sigma=10*1e5 # 10 km/s
	return sigma* epicycle(r,Msat,z,f_gas)/(np.pi*G*Sigma_Gas)

def Tvir(Mhalo,redshift):
        return G*Mhalo*Msun/R200(Mhalo,redshift) * mu * mp/(2*KB)

def v_escp(Mvir,z):
	Vc=V_c(Mvir,z)
	cc=find_cc(Mvir,z)
	return (2*Vc**2 * cc/g(cc))**.5 #Madau et al+01

	
