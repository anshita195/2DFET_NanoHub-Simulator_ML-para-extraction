#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:39:57 2021
Ref.: DOI: 10.1109/TED.2011.2159221
@author: ningyang, jingguo
"""

from __future__ import division
import numpy as np
from scipy.optimize import fsolve
import Rappture
import matplotlib.pyplot as pl
import sys

def fermi(x,fermi_order,fermi_flag):
    if fermi_order == 1/2:
        if fermi_flag == 1:
            exp_fac = np.exp(-0.17 * np.power((x + 1), 2))
            nu = np.power(x, 4) + 50 + 33.6 * x * (1 - 0.68 * exp_fac)
            zeta = 3.0 * np.sqrt(np.pi) / (4 * np.power(nu,0.375))
            return np.exp(x) / (1 + zeta * np.exp(x))
        elif fermi_flag == 0:
            return np.exp(x)

    elif fermi_order == 0:
        if fermi_flag == 1:
            return np.log(1 + np.exp(x))
        elif fermi_flag == 0:
            return np.exp(x)

    elif fermi_order == -1/2:
        if fermi_flag == 1:
            exp_fac = np.exp(-0.17 * np.power((x + 1), 2))
            nu = np.power(x, 4) + 50 + 33.6 * x * (1 - 0.68 * exp_fac)
            zeta = 3 * np.sqrt(np.pi) / (4 * np.power(nu, 0.375))
            nu_prime = 4 * np.power(x, 3) + 33.6 - 22.848 * exp_fac * (1 - 0.34 * (x + np.power(x, 2)))
            zeta_prime = -(9 * np.sqrt(np.pi) / 32) * np.power(nu, -11/8) * nu_prime
            return (np.exp(-x) - zeta_prime) / np.power((np.exp(-x) + zeta), 2)
        elif fermi_flag == 0:
            return np.exp(x)
        
hbar=1.055e-34
qe=1.6e-19  # elementary electron charge
epso0=8.854e-12 # vacuum dielectric constan
m0=9.11e-31
T=300
kT=0.0259*(T/300)

def fx(Utop, *data):  # function for solving Poisson potential
    UL,Ef1,Ef2,U0,N2D=data  # parameter
    y=Utop-UL-U0*(N2D/2)*(fermi((Ef1-Utop)/kT,0,1)+fermi((Ef2-Utop)/kT,0,1)) 
    return y   # solve y=0

def cal_Utop(Vg,Vd,alphag,alphad,Cins,N2D):
    UL=-alphag*Vg-alphad*Vd  # Laplace potential
    Ef1,Ef2=0,-Vd
    U0=qe*alphag/Cins  # charging energy, Ctot=Cins/alphag
    data=(UL,Ef1,Ef2,U0,N2D) # parameter for fx=0  
    Utop=fsolve(fx,0.0,args=data) # Poisson part of potential
    '''
    EPS=1e-9
    Utop=0
    tmp=0.5
    while (abs(Utop-tmp)>EPS): # Newton-Raphson
        Utop=tmp
        tmp=Utop-(Utop-UL-U0*(N2D/2)*(fermi((Ef1-Utop)/kT,0,1)+fermi((Ef2-Utop)/kT,0,1)))/(1+U0*(N2D/2/kT)*(1/(np.exp(Utop/kT-Ef1/kT)+1)+1/(np.exp(Utop/kT-Ef2/kT)+1)))
    Utop=tmp
    '''
    return Utop
    
def get_I_Q(Utop,Vd,N2D,ml):
    Ef1,Ef2=0,-Vd
    Q0=(N2D/2)*(fermi((Ef1-Utop)/kT,0,1)+fermi((Ef2-Utop)/kT,0,1)) 
    I0=np.sqrt(2*kT*qe/(np.pi*ml*m0))*qe*N2D/2*(fermi((Ef1-Utop)/kT,1/2,1)
                                                  -fermi((Ef2-Utop)/kT,1/2,1)) 
    vel=I0/(qe*Q0)
    return I0,Q0,vel

def sim_1bias(Vg,Vd,alphag,alphad,Cins,N2D,ml,ballistic,Lg_nm,mfp_nm):
    Utop=cal_Utop(Vg,Vd,alphag,alphad,Cins,N2D)
    if ballistic:
        I0,Q0,vel=get_I_Q(Utop,Vd,N2D,ml)
    else:   # scattering
        _,Q0,_=get_I_Q(Utop,Vd,N2D,ml)
        vel=cal_sca_vel(Vd,Utop,ml, Lg_nm, mfp_nm)
        I0=qe*Q0*vel
    return I0,Q0,vel,Utop

def cal_sca_vel(Vd,Utop,ml,Lg_nm,mfp_nm):
    Ef1=0
    eta, v0=(Ef1-Utop)/kT, np.sqrt(2*kT*qe/(np.pi*ml*m0))
    mu_bal=v0*(Lg_nm*1e-9)/(2*kT)*fermi(eta,-1/2,1)/fermi(eta,0,1)
    D_sca=(mfp_nm*1e-9)*v0/2
    mu_sca=D_sca/kT
    mu=1/(1/mu_bal+1/mu_sca)
    vsat_bal=v0*(fermi((Ef1-Utop)/kT,1/2,1)/fermi((Ef1-Utop)/kT,0,1))
    vsat_sca=mu_sca/(Lg_nm*1e-9)*Vd
    vsat=vsat_bal*vsat_sca/(vsat_bal+vsat_sca)
    
    infs=1e-10  # avoid singularity as denominator
    Vd_sat=vsat*Lg_nm*1e-9/mu+infs
    beta=1.4
    
    Fs=(Vd/Vd_sat)/(1+(Vd/Vd_sat)**beta)**(1/beta)
    vel=vsat*Fs
    return vel

class TwoDFET(object):  # Quantum Monte Carlo simulation by quantum jump
    def __init__(self, Material,e_or_h,channel_direction, tins, epsor, Vfb, alphag, alphad,transport_model, Lg_nm, mfp_nm, T): # intialize the quantum process circuit class 
        Ngate=2
        self.Vfb=Vfb   
        meff={'MoS2':{'e': [0.5788,0.5664],'h':[0.6659,0.6524]},
              'MoSe2':{'e':[0.6059,0.5933],'h':[0.7114,0.6967]},
              'MoTe2':{'e':[0.6164,0.6033],'h':[0.7586,0.7406]},
              'WS2':{'e':[0.3466,0.3382],'h':[0.4619,0.4501]},
              'BP':{'e':[0.17,1.20],'h':[0.16,6.49]}}
        if channel_direction=='x':
            ml=meff[Material][e_or_h][0]
            mt=meff[Material][e_or_h][1]
        else:
            ml=meff[Material][e_or_h][1]
            mt=meff[Material][e_or_h][0]
        if Material=='BP':
            n_valley=1
        else:
            n_valley=2
        n_spin=2
        self.ml=ml
        self.N2D=n_valley*n_spin*m0*np.sqrt(ml*mt)*kT*qe/(2*np.pi*hbar**2)
        self.tins=tins*1e-9  # in m, insulator thickness
        self.epsor=epsor   # dielectric constant
        self.Cins=Ngate*self.epsor*epso0/self.tins  # gate capacitance
        self.alphag=alphag   # gate control parameter
        self.alphad=alphad  # drain control parameter  
        if (transport_model == 'ballistic'):
            self.ballistic=True # True for ballistic transport, False for scattering
        else:
            self.ballistic=False
        self.Lg_nm=Lg_nm
        self.mfp=mfp_nm
    
    def simulate(self,Vg0,Vg1,N_vg,Vd0,Vd1,N_vd):
        Vgv=np.linspace(Vg0,Vg1,N_vg)
        Vdv=np.linspace(Vd0,Vd1,N_vd)
        Id,Q=np.zeros((N_vg,N_vd)),np.zeros((N_vg,N_vd))
        vel,Utop=np.zeros((N_vg,N_vd)),np.zeros((N_vg,N_vd))

        for iig,Vg in enumerate(Vgv):
            for iid,Vd in enumerate(Vdv):
                Id[iig,iid],Q[iig,iid],vel[iig,iid],Utop[iig,iid]=sim_1bias(Vg-self.Vfb,
                    Vd,self.alphag,self.alphad,self.Cins,self.N2D,self.ml,self.ballistic,
                    self.Lg_nm, self.mfp)
        return Vgv,Vdv,Id,Q,vel,Utop
def cal_Cg(Q,dVg):
    grad=np.gradient(qe*Q,dVg)
    return grad[0]

def plot_2d(xdata,ydata,xlabel='',ylabel=''):
    fig=pl.figure()
    pl.plot(xdata,ydata)
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    return fig
def viz(Vgv,Vdv,Id,Q,vel,Utop,Cg):
    fig=[]
    fig.append(plot_2d(Vdv,Id.T,'$|V_d|$ [V]','$|I_D|$ [$\mu$A/$\mu$m]'))
    '''
    fig.append(pl.figure())
    pl.semilogy(Vgv,Id)
    pl.xlabel('$|V_g|$ [V]')
    pl.ylabel('$|I_D|$ [$\mu$A/$\mu$m]')
    '''
    fig.append(plot_2d(Vgv,Id,'$|V_g|$ [V]','$|I_D|$ [$\mu$A/$\mu$m]'))
    fig.append(plot_2d(Vdv,vel.T,'$|V_d|$ [V]','$v$ [m/s]'))
    fig.append(plot_2d(Vgv,Utop,'$|V_g|$ [V]','$|U_{top}|$ [eV]'))
    fig.append(plot_2d(Vgv,Q,'$|V_g|$ [V]','$density$ [$m^{-2}$]'))
    fig.append(plot_2d(Vgv,Cg,'$|V_g|$ [V]','$C_g$ [F/$m^2$]'))
    return fig

#method to get the parameters from rappture 
def get_label(driver, where, name, units="V"):
    v = driver.get("{}.({}).current".format(where, name))
    
    if (v != None):
        print "{} = {}\n".format(name, v)
    else:
        print "Failed to retrieve {}{} from driver.xml\n".format(where,name)
    
    v2 = Rappture.Units.convert(v, to=units, units="off")
    return (v, v2)  # unit,unitless

def get_rappture_input(driver1):  # get input from rappture
    
    
    Material,_ = get_label(driver1, "input.(device)", "Material") 
    e_or_h_type,_ = get_label(driver1, "input.(device)", "CMOS") 
    if (e_or_h_type=='n-type (electron) channel'):
        e_or_h = 'e'
    else:
        e_or_h = 'h'
    channel_direction,_ = get_label(driver1, "input.(device)", "channel") 
    _,tins = get_label(driver1, "input.(device)", "gateinsulator") 
    _,epsor = get_label(driver1, "input.(device)", "epsr") 
    
    transport_model,_ = get_label(driver1, "input.(models)", "transport")
    _,Lg_nm = get_label(driver1, "input.(models)", "Lg")
    _,mfp_nm = get_label(driver1, "input.(models)", "scatteringmfp")
    _,Vfb = get_label(driver1, "input.(models)", "Ef","V")
    _,alphag = get_label(driver1, "input.(models)", "alphag")
    _,alphad = get_label(driver1, "input.(models)", "alphad")
    
    _,T = get_label(driver1, "input.(voltage)", "temperature","K")
    #Vgv
    _,gVI = get_label(driver1, "input.(voltage).(gV)", "gVI","V")
    _,gVF = get_label(driver1, "input.(voltage).(gV)", "gVF","V")
    _,gNV = get_label(driver1, "input.(voltage).(gV)", "gNV")
    #Vdv
    _,dVI = get_label(driver1, "input.(voltage).(dV)", "dVI","V")
    _,dVF = get_label(driver1, "input.(voltage).(dV)", "dVF","V")
    _,dNV = get_label(driver1, "input.(voltage).(dV)", "dNV")
    gNV=int(gNV)
    dNV=int(dNV)
    return Material, e_or_h, channel_direction, tins, epsor, Vfb,alphag,alphad,transport_model, Lg_nm, mfp_nm,T, gVI, gVF, gNV, dVI ,dVF, dNV
 
   
def fig_plot(fig,name,driver,xlabel,xunit,ylabel,yunit,graN):
    ### Fig of ID-VD drain

    ax = fig.gca()
    lines = ax.lines
    xdata =  []
    ydata = []
    for i in range(len(lines)):
        xdata.append(lines[i].get_xdata())
        ydata.append(lines[i].get_ydata())

    Item = name 
    # Label the output graph with a title, x-axis label,
    # y-axis label, and y-axis units
    curN=1
    driver.put('output.curve(f'+str(graN)+str(curN)+').xaxis.label',xlabel,append=0)
    driver.put('output.curve(f'+str(graN)+str(curN)+').xaxis.units',xunit,append=0)  
    driver.put('output.curve(f'+str(graN)+str(curN)+').yaxis.label',ylabel,append=0)
    driver.put('output.curve(f'+str(graN)+str(curN)+').yaxis.units',yunit,append=0)    
    
    for jj in range(len(xdata)):
        driver.put('output.curve(f'+str(graN)+str(jj+1)+').about.group',Item,append=0)
        for ii in range(len(xdata[jj])):
            line = "%g %g\n" % (xdata[jj][ii], ydata[jj][ii])
            driver.put('output.curve(f'+str(graN)+str(jj+1)+').component.xy', line, append=1)

def output_to_rappture(driver1,fig,output): # output figures and output log to rappture
    
    # image output
    name = ['I_D-V_D','I_D-V_g','velocity-V_d','U_top-V_g','Density-V_g','C_g-V_g']
    xunit = 'V'
    yunit = ['uA/um','uA/um','m/s','eV','m^-2','F/m^2']
    xlabel = ['Vd','Vg','Vd','Vg','Vg','Vg']
    ylabel = ['Id','Id','velocity','Utop','density','Cg']
    for i in range(6):
        fig_plot(fig[i],name[i],driver1,xlabel[i],xunit,ylabel[i],yunit[i],i+1)
    '''
    # log file output
    output
    driver1.put('output.log()',output,append=0)      
    driver1.put('output.log().about.label','Data',append=0)
    '''
    Rappture.result(driver1) # output the results         

if __name__ == '__main__':
    driver1 = Rappture.library(sys.argv[1])
    Material, e_or_h, channel_direction, tins, epsor, Vfb, alphag, alphad, transport_model, Lg_nm, mfp_nm, T, gVI, gVF, gNV, dVI ,dVF, dNV = get_rappture_input(driver1)
    kT=0.0259*(T/300)
    ### simulation below
    fet=TwoDFET(Material,e_or_h,channel_direction, tins, epsor, Vfb, alphag, alphad,transport_model, Lg_nm, mfp_nm, T)
    
    Vgv,Vdv,Id,Q,vel,Utop=fet.simulate(gVI, gVF, gNV, dVI ,dVF, dNV)
    
    Cg=cal_Cg(Q,Vgv[1]-Vgv[0])
    
    fig=viz(Vgv,Vdv,Id,Q,vel,Utop,Cg)
    
    output=Material+e_or_h+channel_direction+str(tins)+str(epsor)+str(Vfb)+str(alphag)+str(alphad)+transport_model+str(Lg_nm)+str(mfp_nm)+str(T)+str(gVI)+str(gVF)+str(gNV)+str(dVI)+str(dVF)+str(dNV) 
    output_to_rappture(driver1,fig,output)
    
    sys.exit()      #exit
