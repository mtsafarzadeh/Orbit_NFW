import numpy as np

mu=0.6
n=1e-4 # /cm^3 relevant for CGM of MW
Mvir=1e12
cc=10.
G=6.6743 * 1.e-8
Msun=1.98e33
pc=3.08e18 # cm
kpc=1e3*pc
mp=1.67e-24
KB=1.38e-16
Msun=1.98e33
Delta=178
rhocrit=9.2e-30 #gr/cm^3
Coeff=7969

def R200(Mvir):
        return  (3./(4.*3.1415)*Mvir*1.98E33/(180*rhocrit))**(1./3.)

def Tvir(Mvir):
        return G*Mvir*Msun/R200(Mvir) * mu * mp/(2*KB)
def NFW_g(M,cc,r):
        A_phi= -1*G*M*Msun/(np.log(1+cc)-cc/(1+cc))
        beta=cc/R200(M)
        return A_phi * (beta/(r*(1+beta*r))-np.log(1+beta*r)/r**2)

def F(cc):
        return np.log(1+cc)-cc/(1+cc)
def V_c(Mvir):
        return (G*Mvir*Msun/R200(Mvir))**.5
def Vesc2(Mvir,cc,r):
        x=r/R200(Mvir)
        return 2*V_c(Mvir)**2 *(F(cc*x)+cc*x/(1+cc*x))/(x*F(cc))
def rho_gas(Mvir,cc,r):
        v_esc02=2*V_c(Mvir)**2 *cc/F(cc)
        #print("v_esc02",v_esc02)
        dmmy=(v_esc02-Vesc2(Mvir,cc,r)) /V_c(Mvir)**2
        #print("dmmy",dmmy)
        return Coeff*rhocrit* np.exp(-dmmy)/mu

def pressure_gas(Mvir,cc,r):
        return rho_gas(Mvir,cc,r)*Tvir(Mvir)*KB/(mp*mu)

def dPdr(Mvir,cc,r):
        return np.diff(pressure_gas(Mvir,cc,r))/np.diff(r)

def rhonfw(r,rvir,cc):
        x=r/rvir
        deltac=178/3*cc**3/F(cc)
        return rhocrit*deltac/(cc*x*(1+cc*x)**2)
def rho_isentropic(Mvir,cc,r):
        eps=1
        x=r/R200(Mvir)
        dmmy1=np.log(1+cc*x)/(cc*x)-np.log(1+cc)/cc
        dmmy2=4./5 * cc/(np.log(1+cc)-cc/(1+cc)) *eps
        coeff=1+dmmy2*dmmy1
        #Rho_vir=rhonfw(r,R200(Mvir),cc)
        Rho_vir= Mvir*Msun/(4*np.pi/3* R200(Mvir)**3)
        return 0.1118421052631579*Rho_vir*coeff**1.5 
def temp_isentropic(Mvir,cc,r):
        gamma=1.6666
        gamma=5./3
        S_vir=Tvir(Mvir)/(rho_isentropic(Mvir,cc,R200(Mvir))/mu/mp)**(2./3)
        A_vir=KB*S_vir/(mu*mp)**gamma
        return A_vir* mu*mp*rho_isentropic(Mvir,cc,r)**(gamma-1)/KB

def dPdr_isentropic(Mvir,cc,r):
        pressure=rho_isentropic(Mvir,cc,r)*temp_isentropic(Mvir,cc,r)*KB/(mp*mu)
        return np.diff(pressure)/np.diff(r)

def entropy(Mvir,cc,r):
        return temp_isentropic(Mvir,cc,r)/(rho_isentropic(Mvir,cc,r)/mu/mp)**(2./3)

def Halo_gas_mass(Mvir):
        return quad(lambda x: 4*np.pi* x**2 *rho_isentropic(Mvir, cc,x), 1, R200(Mvir))[0]/(Msun*Mvir)

def density_isentropic(Mvir,cc,r):
        return rho_isentropic(Mvir,cc,r)/mu/mp


