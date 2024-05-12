'''
    ----------------------------------------------------------------------------------------------
    
    Imports
   
    ----------------------------------------------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import scipy as sc
import scipy.constants as sc
from scipy.ndimage import convolve
from scipy.ndimage import generate_binary_structure
import Ising_Model_Functions
from Ising_Model_Functions import *



'''
    ----------------------------------------------------------------------------------------------
    
    Generate desired Lattice
   
    ----------------------------------------------------------------------------------------------
'''

while True:
    dimensionality = int(input("Enter the dimensionality of the lattice (1 for 1D, 2 for 2D, 3 for 3D): "))
    lattice = generate_lattice(dimensionality)
    print("Lattice Array:")

    if dimensionality == 1 and lattice.size < 3:
        print("Error: 1D lattice must have at least 3 objects.")
        continue  # Restart the loop to prompt for dimensionality again
    elif dimensionality == 2 and (lattice.shape[0] < 3 or lattice.shape[1] < 3):
        print("Error: 2D lattice must have at least dimensions of 3x3.")
        continue  # Restart the loop to prompt for dimensionality again
    elif dimensionality == 3 and (lattice.shape[0] < 3 or lattice.shape[1] < 3 or lattice.shape[2] < 3):
        print("Error: 3D lattice must have at least dimensions of 3x3x3.")
        continue  # Restart the loop to prompt for dimensionality again

    if dimensionality == 1: 
        visualize_1d_lattice(lattice)
        n_x = lattice.shape[0]  # Assign the size of the lattice along the first dimension to n_x
        V = n_x
    elif dimensionality == 2:
        visualize_2d_lattice(lattice)
        n_x, n_y = lattice.shape  # Assign the size of the lattice along the first and second dimensions to n_x and n_y
        V = n_x * n_y
    elif dimensionality == 3:
        visualize_3d_lattice(lattice)
        n_x, n_y, n_z = lattice.shape  # Assign the size of the lattice along the three dimensions to n_x, n_y, and n_z
        V = n_x * n_y * n_z

    break  # Exit the loop if the lattice meets the minimum size requirements


'''
    ----------------------------------------------------------------------------------------------
    
    Select boundary conditions 
   
    ----------------------------------------------------------------------------------------------
'''

while True: 
    boundary = input("Enter desired boundary condition: \n a. Periodic \n b. Reflective \n c. Open \n ").strip().lower()
    if boundary in ["a", "a.", "periodic"]:
        print("You have selected 'Periodic' boundary conditions.")
        boundaries = 'wrap'
        break
    elif boundary in ["b", "b.", "reflective"]:
        print("You have selected 'Reflective' boundary conditions.")
        boundaries =  'reflect'
        break
    elif boundary in ["c", "c.", "open"]:
        print("You have selected 'Open' boundary conditions.")
        boundaries = 'constant'
        break
    else:
        print("Invalid boundary conditions. Please choose again.")


'''
    ----------------------------------------------------------------------------------------------
    
    Initialise system
   
    ----------------------------------------------------------------------------------------------
'''

J = (1 * (10**-21)) # Exchange interaction must be of magnitude similar to k to produce reasonable energies


while True:
    T = input("Enter the desired temperature (K) for the lattice's environment:")
    try: 
        T = int(T)
        break
    except:
        print ("Invalid input, value must be an integer.")

while True:
    iterations = input("Enter the desired number of iterations of the Metropolis algorithm:")
    try: 
        iterations = int(iterations)
        break
    except:
        print ("Invalid input, value must be an integer.")
        
while True:
    h = input("Enter the desired external magnetic field value, h:")
    try: 
        h = int(h)
        break
    except:
        print ("Invalid input, value must be an integer.")        
        
        
init_energy = lattice_energy(lattice,boundaries,J,h)


'''
    ----------------------------------------------------------------------------------------------
    
    Apply Metropolis Algorithm to system 
   
    ----------------------------------------------------------------------------------------------
'''

if dimensionality == 1 and boundaries == 'wrap':
    spins, energies, lattice_final =  metropolis_1D_periodic(lattice,boundaries,iterations,T,J,h)  
elif dimensionality == 1   and boundaries ==  'reflect' :
    spins, energies, lattice_final =  metropolis_1D_reflective(lattice,boundaries,iterations,T,J,h)  
elif dimensionality == 1 and boundaries == 'constant':
    spins, energies, lattice_final =  metropolis_1D_open(lattice,boundaries,iterations,T,J,h) 
elif dimensionality == 2 and boundaries == 'wrap':
     spins, energies, lattice_final =  metropolis_2D_periodic(lattice,boundaries,iterations,T,J,h)  
elif dimensionality == 2  and boundaries ==  'reflect' :
     spins, energies, lattice_final =  metropolis_2D_reflective(lattice,boundaries,iterations,T,J,h)  
elif dimensionality == 2 and boundaries == 'constant':
     spins, energies, lattice_final =  metropolis_2D_open(lattice,boundaries,iterations,T,J,h)    
elif dimensionality == 3 and boundaries == 'wrap':
     spins, energies, lattice_final =  metropolis_3D_periodic(lattice,boundaries,iterations,T,J,h)  
elif dimensionality == 3   and boundaries ==  'reflect' :
     spins, energies, lattice_final =  metropolis_3D_reflective(lattice,boundaries,iterations,T,J,h)  
elif dimensionality == 3 and boundaries == 'constant':
     spins, energies, lattice_final =  metropolis_3D_open(lattice,boundaries,iterations,T,J,h)    
    
 
'''
    ----------------------------------------------------------------------------------------------
    
    Plot spins and energies against number of iterations to test the algorithm 
   
    ----------------------------------------------------------------------------------------------
'''

fig,axes = plt.subplots(1,2,figsize=(12,4))
# Plot for Spins
ax = axes[0]
ax.plot(spins/V)
ax.set_xlabel("Metropolis Iterations")
ax.set_ylabel(r'Average Spin $\bar{m}$')
ax.grid()
# Plot for Energies
ax = axes[1]
ax.plot(energies)
ax.set_xlabel("Metropolis Iterations")
ax.set_ylabel(r'Energy $E$')
ax.grid()
fig.tight_layout()
fig.suptitle(r'Evolution of Average Spin and Energy for T = {} K'.format(T),y = 1.07, size=18)
plt.show()


'''
    ----------------------------------------------------------------------------------------------
    
    Obtain temperature range and generate thermodynamic data
   
    ----------------------------------------------------------------------------------------------
'''

print('The algorithm must be applied across a range of temperatures to calculate thermodynmic variables.')
while True:
    T_min = input("Enter the desired minimum temperature (K):")
    try: 
        T_min = int(T_min)
        break
    except:
        print ("Invalid input, value must be an integer.")
while True:
    T_max = input("Enter the desired maximum temperature (K):")
    try: 
        T_max = int(T_max)
        break
    except:
        print ("Invalid input, value must be an integer.")
        
temp_range = np.arange(T_min,T_max,5)        

if dimensionality == 1: 
    E_totals, C, S_totals, Chi = thermo_calc_1D(T_min, T_max, lattice, boundaries, iterations, J, h)
elif dimensionality == 2:
    E_totals, C, S_totals, Chi = thermo_calc_2D(T_min, T_max, lattice, boundaries, iterations, J, h)
elif dimensionality == 3:
    E_totals, C, S_totals, Chi = thermo_calc_3D(T_min, T_max, lattice, boundaries, iterations, J, h)


'''
    ----------------------------------------------------------------------------------------------
    
    Plot total energy for different Temperatures
   
    ----------------------------------------------------------------------------------------------
'''

# Apply Seaborn style
sns.set_style("whitegrid")

# Create a figure and plot the data
f1 = plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.plot(temp_range, E_totals, label="Total Energy")

# Add title and labels 
plt.title("Total Energy of the Ising Model for varying Temperatures", fontsize=16)
plt.xlabel("Temperature (K)", fontsize=14)
plt.ylabel("Total energy (J)", fontsize=14)

# Add legend
plt.legend()

# Show plot
plt.show()


'''
    ----------------------------------------------------------------------------------------------
    
    Plot Specific Heat Capacity for different Temperatures
   
    ----------------------------------------------------------------------------------------------
'''


# Apply Seaborn style
sns.set_style("whitegrid")

# Create a figure and plot the data
f2 = plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.plot(temp_range, C, label="Specific Heat Capacity")

# Add title and labels 
plt.title("Specific Heat Capacity of the Ising Model for varying Temperatures", fontsize=16)
plt.xlabel("Temperature (K)", fontsize=14)
plt.ylabel("Specific Heat Capacity, C(T)", fontsize=14)

# Add legend
plt.legend()

# Show plot
plt.show()


'''
    ----------------------------------------------------------------------------------------------
    
    Plot Magnetisation for different Temperatures
   
    ----------------------------------------------------------------------------------------------
'''


# Apply Seaborn style
sns.set_style("whitegrid")

# Create a figure and plot the data
f3 = plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.plot(temp_range, S_totals, label="Magnetisation")

# Add title and labels 
plt.title("Magnetisation of the Ising Model for varying Temperatures", fontsize=16)
plt.xlabel("Temperature (K)", fontsize=14)
plt.ylabel("Magnetisation, (m)", fontsize=14)

# Add legend
plt.legend()

# Show plot
plt.show()



'''
    ----------------------------------------------------------------------------------------------
    
    Plot Magnetic Susceptibility for different Temperatures
   
    ----------------------------------------------------------------------------------------------
'''


# Apply Seaborn style
sns.set_style("whitegrid")

# Create a figure and plot the data
f4 = plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.plot(temp_range, Chi, label="Magnetic Susceptibility")

# Add title and labels 
plt.title("Magnetic Susceptibility of the Ising Model for varying Temperatures", fontsize=16)
plt.xlabel("Temperature (K)", fontsize=14)
plt.ylabel(r'Magnetic Susceptibility, ($\chi$)', fontsize=14)

# Add legend
plt.legend()

# Show plot
plt.show()


print('Simulation Complete.') # Confirms program is done running thermo_calc and plotting data
