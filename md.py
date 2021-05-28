import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d.axes3d as p3
from scipy.constants import k
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import time
from itertools import product, combinations

writervideo=animation.FFMpegWriter(fps=60)
#set other parameters of atoms
epsilon=0.0103 #in eV
sigma=3.43 #in A
m=39.948 #amu
#-----init---------------------------------------------------------------------------------

class Dynamics:
    def init (self):

        #dimension of the simulation
        self.dim=3

        #read number of atoms from the first line of the xyz file
        self.n_atom=1
        
        #set velocity and position-array to a zero-array with dimension of number of atoms x dimension (v_x, v_y and v_z for each atom)
        self.velocity=np.zeros((self.n_atom, self.dim))
        self.position=np.zeros((self.n_atom, self.dim))
        self.accel=np.zeros((self.n_atom, self.dim))
        
        #just some parameters needed
        self.mass=m #amu
        self.T=T #K
        self.box_len=20 #in m
        self.dt=0.01 #in s
        self.steps=500
        self.filename='filename'
    


#-----init posi and velocity---------------------------------------------------------------------------------


    #randomize each component of velocity through maxwell-boltzmann-distribution
    #maxwell-boltzmann-distribution is a normal distribution of gas-particle velocities
    #where the centre is at loc=0 and the (scale) std_dev is at sigma=sqrt(kT/m)
    #this function sets the component of each velocity to a random value of that normal distribution
    #luckily, numpy has a function that randomizes values considering the chosen distribution
    def init_velocity (self):
        std_dev=np.sqrt(k*self.T/self.mass)
        self.velocity=np.random.normal(loc=0, scale=std_dev, size=(self.n_atom, self.dim))*1e10 #in ~m/s *10^-13 deswegen *10^10 für ausgabe in angstrom
        


    def init_posi_from_file (self):
        #read number of atoms from the first line of the xyz file
        with open(self.filename,'r') as f:
                self.n_atom=int(f.readline()) 
        self.x=np.loadtxt(self.filename, skiprows=1, usecols=(0)) 
        self.y=np.loadtxt(self.filename, skiprows=1, usecols=(1))
        self.z=np.loadtxt(self.filename, skiprows=1, usecols=(2))
        self.position_sw=np.array([self.x, self.y, self.z])
        self.position=np.transpose(self.position_sw)
    

    def init_posi_rnd (self):
        self.position=np.random.random_sample((self.n_atom,self.dim))*0.9*(self.box_len)
        self.x=self.position[:,0]
        self.y=self.position[:,1]
        self.z=self.position[:,2]


    def move_atoms(self):
        for i in range(self.n_atom):
            for j in range(i+1, self.n_atom):
                r=self.position[i]-self.position[j]
                r_mag=np.linalg.norm(r)
                r_norm=r/r_mag
                if (r_mag <0.9*sigma):
                    self.position[i]=self.position[i]+(0.4*sigma)*r_norm
                    self.position[j]=self.position[j]-(0.4*sigma)*r_norm
            self.check_boundary()        


#-----energies---------------------------------------------------------------------------------

    def pe_pair(self, particle_1, particle_2):
        r = self.position[particle_1] - self.position[particle_2]
        r_mag = np.linalg.norm(r)
        
        return 4*epsilon*((sigma/r_mag)**12 - (sigma/r_mag)**6)
    

    def pe(self):
        total_pe = 0.0
        for i in range(self.n_atom):
            for j in range(i+1, self.n_atom):
                total_pe += self.pe_pair(i,j)
        return total_pe
    


    def ekin(self):
        ekin_tot=0
        for i in range(self.n_atom):
            v_mag=np.linalg.norm(self.velocity[i])
            ekin_tot=ekin_tot + 0.5*m*(v_mag**2)
        return ekin_tot


#-----force---------------------------------------------------------------------------------

    #the force is the derivative of the energy (potential)
    #here we use the LJ-12-6-potential, so we have to form the derivative after r
    def lj_interaction (self, particle_1, particle_2):
        r=self.position[particle_1] - self.position[particle_2] #distance between 2 particles in nm
        #you can only calculate force in a direction by multiplying the force with the unit vector in that direction
        r_magnitude=np.linalg.norm(r) #magnitude of the vector
        r_norm=r/r_magnitude #vector in the direction with length of 1
        if (r_magnitude <= 2*sigma):
        #force in a direction = force*(r_i / |r|)
            f_magnitude=(48*epsilon*(sigma**12)/(r_magnitude**13)-24*epsilon*(sigma**6)/(r_magnitude**7))
            f_vector = f_magnitude*r_norm
        else:
            f_vector=[0,0,0]
        return f_vector #return the force vector with the forces acting in each direction as components



    #determine all forces acting on a particle
    def lj_choose (self, p1):
        force=np.zeros(shape=3)
        for p2 in range(self.n_atom): 
            if(p1 == p2): #so the program doesnt calculate the force of the particle with itself
                continue
            force = force + self.lj_interaction(p1, p2) #add forces of all on p1 interacting particles
        return force #in eV/A
    

#-----integrator---------------------------------------------------------------------------------

    def velocity_verlet (self):
        energy=open('Energy.txt', 'w')
        trj=open('trj.xyz', 'w')
        #trj.writelines(str(self.n_atom)+'\n'+'md-sim'+'\n')
#-----velocity verlet---------------------------------------------------------------------------------
        for i in range(self.steps):
            self.position = self.position + 0.5*self.accel*(self.dt**2) + self.velocity*self.dt 
            self.check_boundary()
            self.x=self.position[:,0]
            self.y=self.position[:,1]
            self.z=self.position[:,2]
            trj.writelines(str(self.n_atom)+'\n'+'Frame' + ' ' + str(i)+'\n')
            for j in range(self.n_atom):
                xp=self.x[j]
                yp=self.y[j]
                zp=self.z[j]
                trj.writelines(' '+'Ar'+' '+str('{0:.5f}'.format(xp))+' '+str('{0:.5f}'.format(yp))+' '+str('{0:.5f}'.format(zp))+'\n')
            
            #trj.writelines('\n'+str('{0:.5f}'.format(self.box_len))+' ' + '0.00000' +' '+ '0.00000'+'\n'+ '0.00000' +' '+ str('{0:.5f}'.format(self.box_len))+' ' + '0.00000'+'\n' + '0.00000'+' '+ '0.00000' + ' '+ str('{0:.5f}'.format(self.box_len))+'\n' )
            #trj.writelines('\n') 

            forces=np.array([self.lj_choose(p) for p in range(self.n_atom)])
            accel=(forces/self.mass) #in eV/A*amu
            self.velocity = self.velocity + 0.5*(self.accel+accel)*self.dt
            self.accel=accel
#-----for energy plot---------------------------------------------------------------------------------
            pe_tot=self.pe()
            ekin_tot=self.ekin()
            e_tot=pe_tot+ekin_tot
            energy.writelines(str(e_tot)+'\t'+str(pe_tot)+'\t'+str(ekin_tot)+'\n')
        energy.close()
        trj.close()


    def check_boundary(self):
        for n in range(self.n_atom):
            for c in range(0,2):
                if (self.position[n,c]>self.box_len):
                    self.position[n,c]=self.position[n,c]-self.box_len
                if (self.position[n,c]<=0):
                    self.position[n,c]=self.position[n,c]+self.box_len





#choose the temperature
temperature=input("Choose the simulation temperature in Kelvin:")
T=float(temperature)
#-----start program-------------------------------------------------------------------------------------------------
dyn=Dynamics()
dyn.init()
i=False
while i is False:
    inp_method=input("Do you have a file in xzy format which you would like to simulate? Type Y or N:")
    if inp_method=='Y' or inp_method=='y':
        filename=input('Please enter your exact filename:')
        dyn.filename=filename
        dyn.init_posi_from_file()
        dyn.init_velocity()
        i=True
    if inp_method=='N' or inp_method=='n':
        n_of_atom=input('Please enter your desired number of atoms:')
        dyn.n_atom=int(n_of_atom)
        dyn.init_posi_rnd()
        dyn.init_velocity()
        dyn.move_atoms()
        #dyn.minimize()
        i=True
    else:
        print('False input, please write either Y or N!')



dyn.velocity_verlet()



