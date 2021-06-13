
#-------------------------------------------------------------------#
#                                                                   #
#               MOLECULAR DYNAMICS SIMULATION                       #
#               USED POTENTIAL: LENNARD-JONES-12-6                  #
#               POSSIBLE ATOMS TO SIMULATE: Ar                      #
#               ONLY GAS PHASE                                      #
#       02.06.2021                          AUTHOR: @AZAD KIRSAN    #
#                                                                   #
#-------------------------------------------------------------------#






#----------------------------#
#                            #
#           START            #
#                            #
#----------------------------#

import numpy as np
import math
from scipy.constants import k
import time
#from itertools import product, combinations
import sys
#-----------------------------------------------#
#                                               #
#           INITIALIZATION OF PARAMETERS        #
#                                               #
#-----------------------------------------------#

class Simulation:
    def init (self):
        self.dim=3
        self.n_atom=1
        self.box_len=20 #in A
        self.dt=0.005 #in s
        self.steps=2000
        self.filename='filename'
        self.bc='HW'
        self.epsilon=0.0103 #in eV #
        self.sigma=3.40 #in A      # LJ parameters
        self.m=39.948 #amu


#-------------------------------------------------------#
#                                                       #
#           INITIALIZATION OF POSITION AND VELOCITY     #
#                                                       #
#-------------------------------------------------------#

    def init_velocity (self):
       # std_dev=np.sqrt(k*T/self.m)
       # self.velocity=np.random.normal(loc=0, scale=std_dev, size=(self.n_atom, self.dim))*1e10
        
        P=2*(np.random.rand(self.n_atom, self.dim)-0.5)
        self.velocity=P* (k*T/(self.m*1.602e-19))**0.5 #in eV/amu

    def init_posi_from_file (self):
        with open(self.filename,'r') as f:
                self.n_atom=int(f.readline()) 
        self.x=np.loadtxt(self.filename, skiprows=2, usecols=(1)) 
        self.y=np.loadtxt(self.filename, skiprows=2, usecols=(2))
        self.z=np.loadtxt(self.filename, skiprows=2, usecols=(3))
        self.position_sw=np.array([self.x, self.y, self.z])
        self.position=np.transpose(self.position_sw)
    

    def init_posi_rnd (self):
        self.position=np.random.random_sample((self.n_atom,self.dim))*0.9*(self.box_len)+0.05*self.box_len
        self.x=self.position[:,0]
        self.y=self.position[:,1]
        self.z=self.position[:,2]

    
    def optimize_geo(self):
        for i in range(600):
            pe_start=self.pe()
            particle = np.random.randint(0,self.n_atom,size=None, dtype=int)
            
            r=(np.random.random_sample(size=3)-0.5)*2
            r_magnitude=np.linalg.norm(r)
            r_norm=r/r_magnitude
            dr=0.05*self.sigma*r_norm
            
            self.position[particle] += dr
            pe_end=self.pe()
            if(pe_start<pe_end):
                self.position[particle] -= dr



#-------------------------------------------#
#                                           #
#           CALCULATION OF ENERGIES         #
#                                           #
#-------------------------------------------#

    def pe_interaction(self, particle_1, particle_2): #get energy contribution from p1 to p2
        r = self.get_min_dist(particle_1, particle_2)
        r_magnitude = np.linalg.norm(r)
        
        return 4*self.epsilon*((self.sigma/r_magnitude)**12 - (self.sigma/r_magnitude)**6)
    

    def pe(self):
        pe_total = 0.0
        for i in range(self.n_atom):
            for j in range(i+1, self.n_atom):
                pe_total += self.pe_interaction(i,j)
        return pe_total
    


    def ekin(self):
        ekin_tot=0.0
        for i in range(self.n_atom):
            v_mag=np.linalg.norm(self.velocity[i])
            ekin_tot +=  0.5*self.m*(v_mag**2)
        return ekin_tot


#-------------------------------------------#
#                                           #
#           CALCULATION OF FORCES           #
#                                           #
#-------------------------------------------#

    #calculate the minimum distance between particle 1 and the nearest neighbor of particle 2 (can be an image)

    def get_min_dist(self, p1, p2):
        r_real=self.position[p1]-self.position[p2]
        if self.bc == "hw" or self.bc =="HW":
            return r_real
        else:
            r_x=r_real[0]
            r_y=r_real[1]    
            r_z=r_real[2]


            if(r_x > self.box_len*0.5):
                r_real += np.array([-self.box_len,0,0])
            elif(r_x <= -self.box_len*0.5):
                r_real += np.array([self.box_len,0,0])
            else:
                pass

            if(r_y > self.box_len*0.5):
                r_real += np.array([0,-self.box_len,0])
            elif(r_y <= -self.box_len*0.5):
                r_real += np.array([0,self.box_len,0])
            else:
                pass

            if(r_z > self.box_len*0.5):
                r_real += np.array([0,0,-self.box_len])
            elif(r_z <= -self.box_len*0.5):
                r_real += np.array([0,0,self.box_len])
            else:
                pass
            
            return r_real




    #the force is the derivative of the energy (potential)
    #here we use the LJ-12-6-potential, so we have to form the derivative after r
    #you can only calculate force in a direction by multiplying the force with the unit vector in that direction
    
    def lj_interaction (self, particle_1, particle_2):
        r=self.get_min_dist(particle_1, particle_2) 
        r_magnitude=np.linalg.norm(r) 
        r_norm=r/r_magnitude 
        f_magnitude= 48*self.epsilon*(self.sigma**12/r_magnitude**13) - 24*self.epsilon*(self.sigma**6/r_magnitude**7)
        f_vector = f_magnitude*r_norm
        return f_vector #return the force vector with the forces acting in each direction as components

    

    #determine all forces acting on a particle
    def lj_force (self, p1):
        force=np.zeros(shape=3)
        for p2 in range(self.n_atom): 
            if(p1 == p2): #so the program doesnt calculate the force of the particle with itself
                continue
            force = force + self.lj_interaction(p1, p2) #add forces of all on p1 interacting particles
        return force #in eV/A
    

#-----------------------------------------------#
#                                               #
#           INTEGRATOR AND BOUNDARIES           #
#                                               #
#-----------------------------------------------#

    def velocity_verlet (self):
        accel_0=np.zeros((self.n_atom, self.dim))
        energy=open('Energy.txt', 'w')
        trj=open('trj.xyz', 'w')
        

        for i in range(self.steps):
           
           # pe_tot=self.pe()
           # ekin_tot=self.ekin()
           # e_tot=pe_tot+ekin_tot
           # energy.writelines(str(e_tot)+'\t'+str(pe_tot)+'\t'+str(ekin_tot)+'\n')
        
            self.position = self.position + 0.5*accel_0*(self.dt**2) +self.velocity*self.dt 
            
            
            pe_tot=self.pe()
            
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
           
            
            forces=np.array([self.lj_force(p) for p in range(self.n_atom)])
            
            accel_1=(forces/self.m) #in eV/A*amu
            
            self.velocity = self.velocity + 0.5*(accel_0+accel_1)*self.dt
            
            ekin_tot=self.ekin()
            accel_0=accel_1
            
            e_tot=pe_tot+ekin_tot

            energy.writelines(str(e_tot)+'\t'+str(pe_tot)+'\t'+str(ekin_tot)+'\n')
        energy.close()
        trj.close()


    def check_boundary(self):
        if self.bc == "pbc" or self.bc =="PBC":
            for n in range(self.n_atom):
                for c in range(0,3):
                    if (self.position[n,c]>self.box_len):
                        self.position[n,c]=self.position[n,c]-self.box_len
                    if (self.position[n,c]<=0):
                        self.position[n,c]=self.position[n,c]+self.box_len
        else:
            for n in range(self.n_atom):
                for c in range(0,3):
                    if (abs(self.position[n,c]>=self.box_len) or abs(self.position[n,c]<=0)):
                        self.velocity[n,c] = -self.velocity[n,c]




#-------------------------------#
#                               #
#           THE PROGRAM         #
#                               #
#-------------------------------#

temperature=input("Choose the simulation temperature in Kelvin: ")
T=float(temperature)
j=False
while j is False:
    boundary=input("Do you want Periodic Boundary Conditions or Hard Walls? Please Type either PBC or HW: ")
    if boundary == 'hw' or boundary == 'HW' or boundary == 'pbc' or boundary == 'PBC' :
        j=True
    else:
        print("Wrong input, please write either PBC or HW!")



dyn=Simulation()
dyn.init()
dyn.bc=boundary
i=False
while i is False:
    inp_method=input("Do you have a file in xyz format which you would like to simulate? Type Y or N: ")
    if inp_method=='Y' or inp_method=='y':
        filename=input('Please enter your exact filename: ')
        try:
            f=open(filename)
        except IOError:
            sys.exit("THE FILE", "", filename, "","DOES NOT EXIST! PLEASE START OVER.")
        finally:
            f.close()
        dyn.filename=filename
        dyn.init_posi_from_file()
        dyn.init_velocity()
        i=True
    elif inp_method=='N' or inp_method=='n':
        n_of_atom=input('Please enter your desired number of atoms: ')
        dyn.n_atom=int(n_of_atom)
        dyn.init_posi_rnd()
        t1=time.time()
        dyn.optimize_geo()
        t2=time.time()
        print("Optimize time=", t2-t1)
        dyn.init_velocity()
        i=True
    else:
        print('False input, please write either Y or N!')
        
t_start=time.time()
dyn.velocity_verlet()
t_end=time.time()
print("MD time=",t_end-t_start)


#-------------------------------#
#                               #
#           END                 #
#                               #
#-------------------------------#
