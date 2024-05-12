'''
    ----------------------------------------------------------------------------------------------
    
    Imports
   
    ----------------------------------------------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import scipy as sc
import scipy.constants as sc
from scipy.ndimage import convolve
from scipy.ndimage import generate_binary_structure


'''
    ----------------------------------------------------------------------------------------------
    
    Functions
   
    ----------------------------------------------------------------------------------------------
'''

def visualize_1d_lattice(lattice):
    """
    Visualize a 1D lattice.

    Parameters:
        lattice (numpy.ndarray): The 1D lattice to visualize.

    Returns:
        None
    """
    plt.plot(lattice, marker='o', linestyle='None', color='blue')
    plt.title("1D Lattice Visualization")
    plt.xlabel("Index")
    plt.ylabel("Spin Value")

    # Set ticks based on lattice size
    x_ticks = np.arange(len(lattice))
    plt.xticks(x_ticks, labels=x_ticks)

    plt.show()
    
    
def visualize_2d_lattice(lattice):
    """
    Visualize a 2D lattice.

    Parameters:
        lattice (numpy.ndarray): The 2D lattice to visualize.

    Returns:
        None
    """
    plt.imshow(lattice, cmap='coolwarm', interpolation='nearest')
    plt.title("2D Lattice Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Spin Value")

    # Set ticks based on lattice size
    x_ticks = np.arange(lattice.shape[1])
    y_ticks = np.arange(lattice.shape[0])
    plt.xticks(x_ticks, labels=x_ticks)
    plt.yticks(y_ticks, labels=y_ticks)

    plt.show()
    
    
def visualize_3d_lattice(lattice):
    """
    Visualize a 3D lattice.

    Parameters:
        lattice (numpy.ndarray): The 3D lattice to visualize.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = lattice.nonzero()
    ax.scatter(x, y, z, c=lattice[x, y, z], cmap='coolwarm')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Lattice Visualization")
    cbar = plt.colorbar(ax.scatter(x, y, z, c=lattice[x, y, z], cmap='coolwarm'))
    cbar.set_label("Spin Value")

    # Set ticks based on lattice size
    max_dim = max(lattice.shape)
    ticks = np.arange(max_dim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    plt.show()
    

def generate_lattice(dimensionality):
    """
    Takes a real number on the interval [1,3] and generates a lattice of randomised spins with dimensions equal to the real number. 
    Where 75% of the spins are +1 and 25% are -1.

    Parameters:
        dimensionality: number of dimensions of desired lattice, 1D,2D or 3D.
        
    Returns:
        Lattice: array of spins randomised as -1 or +1.
       
    """
    lattice = None
    if dimensionality == 1:
        size = None
        while True:
            try:
                size = int(input("Enter the number of spins along the 1D lattice: "))
                break  # Exit the loop if input is successfully converted to an integer
            except ValueError:
                print("Invalid input. Please enter an integer.")
        init_random = np.random.random(size) # Generate 2D array of random values between 0 and 1
        lattice = np.zeros(size)  # Lattice of spins
        lattice[init_random>=0.25] = 1  # 75% of spins in lattice will be positive, but ordered randomly 
        lattice[init_random<0.25] = -1       
    elif dimensionality == 2:
        size_x = None
        size_y = None
        while True:
            try:
                size_x = int(input("Enter the number of spins along the X dimension: "))
                size_y = int(input("Enter the number of spins along the Y dimension: "))
                break  # Exit the loop if inputs are successfully converted to integers
            except ValueError:
                print("Invalid input. Please enter integers.")
        init_random = np.random.random((size_x,size_y)) # Generate 2D array of random values between 0 and 1
        lattice = np.zeros((size_x,size_y))  # Lattice of spins
        lattice[init_random>=0.25] = 1  # 75% of spins in lattice will be positive, but ordered randomly 
        lattice[init_random<0.25] = -1
    elif dimensionality == 3:
        size_x = None
        size_y = None
        size_z = None
        while True:
            try:
                size_x = int(input("Enter the number of spins along the X dimension: "))
                size_y = int(input("Enter the number of spins along the Y dimension: "))
                size_z = int(input("Enter the number of spins along the Z dimension: "))
                break  # Exit the loop if inputs are successfully converted to integers
            except ValueError:
                print("Invalid input. Please enter integers.")
        init_random = np.random.random((size_x,size_y,size_z)) # Generate 2D array of random values between 0 and 1
        lattice = np.zeros((size_x,size_y,size_z))  # Lattice of spins
        lattice[init_random>=0.25] = 1  # 75% of spins in lattice will be positive, but ordered randomly 
        lattice[init_random<0.25] = -1
    else:
        print("Unsupported dimensionality.")
    return lattice


def lattice_energy(system, boundaries, J, h):
    ''' 
    Sums over nearest neighbours of every spin in a lattice to find the energy of the lattice.

    Arguments:
        system: Array of spins with border of objects representing boundary condition.
        boundaries: Boundary conditions of the lattice
        J: Exchange interaction.
        h: External magnetic field value.

    Returns:
        total_energy: Total energy of the lattice divided by J.
    '''
    dim = system.ndim
    
    if dim == 1:
        # For 1D lattice, nearest neighbors are simply the adjacent spins
        kernel = np.array([1, 1])
        if boundaries == 'wrap':
            bounds = 'same'
            energies = -system * np.convolve(system, kernel, mode=bounds)
        elif boundaries == "reflect":
            n = len(system)
            energies = np.zeros_like(system)
            for i in range(n):
                if i == 0:
                    energies[i] = -system[i] * system[i + 1]
                elif i == n - 1:
                    energies[i] = -system[i] * system[i - 1]
                else:
                    energies[i] = -system[i] * (system[i - 1] + system[i + 1])
        else:
            n = len(system)
            energies = np.zeros_like(system)
            for i in range(n):
                if i == 0:
                    energies[i] = -system[i] * system[i + 1]
                elif i == n - 1:
                    energies[i] = -system[i] * system[i - 1]
                else:
                    energies[i] = -system[i] * (system[i - 1] + system[i + 1])
    elif dim == 2:
        # For 2D lattice, use a 3x3 kernel to sum over nearest neighbors
        kernel = generate_binary_structure(2, 1)
        kernel[1, 1] = False
        energies = -system * convolve(system, kernel, mode=boundaries, cval=0)
    elif dim == 3:
        # For 3D lattice, use a 3x3x3 kernel to sum over nearest neighbors
        kernel = generate_binary_structure(3, 1)
        kernel[1, 1, 1] = False
        energies = -system * convolve(system, kernel, mode=boundaries, cval=0)
    else:
        raise ValueError("Unsupported array dimensions: Only 1D, 2D, or 3D arrays are supported.")
    
    total_energy = J * energies.sum() - h * system.sum()
    return total_energy

 
def metropolis_1D_periodic(spin_array,boundaries, times, T, J,h):
    '''Function which takes in a 1D array of spins subjected to periodic boundary conditions, times = number of iterations, temperature and an exchange energy J 
    and performs said iterations of the Metropolis Monte Carlo Algorithm to produce an array of net spins and energies,
    based on the Ising Model.
    Arguments:
        spin_array = 1D array representing lattice of spins
        boundaries = boundary conditions of lattice
        times = desired iterations of algorithm
        T = temperature (K)
        J = exchange interaction
        h = external magnetic field value
    Returns:
       net spins = 1D array of net spin of the latticefor each iteration
       net_energies = 1D array of net energies of the lattice for each iteration
       spin_ array = final lattice after all spin flips 
    '''
    energy = lattice_energy(spin_array,boundaries,J,h)
    k = sc.Boltzmann
    B = 1/(k*T) # Calculate beta value for specified temperature
    spin_array = spin_array.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        n_x = len(spin_array)
        x = np.random.randint(0,n_x)
        spin_i = spin_array[x] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy applying periodic boundary conditions
        E_i = 0
        E_f = 0
        # Left neighbour
        if x==0:
            E_i += -spin_i*spin_array[n_x-1]
            E_f += -spin_f*spin_array[n_x-1]
        if x>0:
            E_i += -spin_i*spin_array[x-1]
            E_f += -spin_f*spin_array[x-1]
        # Right neighbour
        if x<n_x-1:
            E_i += -spin_i*spin_array[x+1]
            E_f += -spin_f*spin_array[x+1]
        if x == n_x-1:
            E_i += -spin_i*spin_array[0]
            E_f += -spin_f*spin_array[0]
        
        # 3 / 4. change state with designated probabilities
        E_i = J*E_i
        E_f = J*E_f
        
        dE = E_f-E_i
        P = np.random.random(1)[0] # Generate random number on interval [0,1)
        if (dE>0)*(P < np.exp(-B*dE)):
            spin_array[x]=spin_f
            energy += dE
        elif dE<=0:
            spin_array[x]=spin_f
            energy += dE
            
        net_spins[t] = spin_array.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy, spin_array


def metropolis_2D_periodic(spin_array,boundaries, times, T, J,h):
    '''Function which takes in a 2D array of spins subjected to periodic boundary conditions, times = number of iterations, temperature and an exchange energy J 
    and performs said iterations of the Metropolis Monte Carlo Algorithm to produce an array of net spins and energies,
    based on the Ising Model.
    Arguments:
        spin_array = lattice of spins
        boundaries = boundary conditions of lattice
        times = desired iterations of algorithm
        T = temperature (K)
        J = exchange interaction
        h = external magnetic field value
    Returns:
       net spins = 1D array of net spin of the latticefor each iteration
       net_energies = 1D array of net energies of the lattice for each iteration
       spin_ array = final lattice after all spin flips 
    '''
    energy = lattice_energy(spin_array,boundaries,J,h)
    k = sc.Boltzmann
    B = 1/(k*T) # Calculate beta value for specified temperature
    spin_array = spin_array.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        n_x, n_y = spin_array.shape
        x = np.random.randint(0,n_x)
        y = np.random.randint(0,n_y)
        spin_i = spin_array[x,y] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy applying periodic boundary conditions
        E_i = 0
        E_f = 0
        # Left element
        if x==0:
            E_i += -spin_i*spin_array[n_x-1,y]
            E_f += -spin_f*spin_array[n_x-1,y]
        if x>0:
            E_i += -spin_i*spin_array[x-1,y]
            E_f += -spin_f*spin_array[x-1,y]
        # Right Element
        if x<n_x-1:
            E_i += -spin_i*spin_array[x+1,y]
            E_f += -spin_f*spin_array[x+1,y]
        if x == n_x-1:
            E_i += -spin_i*spin_array[0,y]
            E_f += -spin_f*spin_array[0,y]
        # Below element
        if y==0:
            E_i += -spin_i*spin_array[x,n_y-1]
            E_f += -spin_f*spin_array[x,n_y-1]
        if y>0:
            E_i += -spin_i*spin_array[x,y-1]
            E_f += -spin_f*spin_array[x,y-1]
        # Above element
        if y<n_y-1:
            E_i += -spin_i*spin_array[x,y+1]
            E_f += -spin_f*spin_array[x,y+1]
        if y==n_y-1:
            E_i += -spin_i*spin_array[x,0]
            E_f += -spin_f*spin_array[x,0]
        
        # 3 / 4. change state with designated probabilities
        E_i = J*E_i
        E_f = J*E_f
        
        dE = E_f-E_i
        P = np.random.random(1)[0] # Generate random number on interval [0,1)
        if (dE>0)*(P < np.exp(-B*dE)):
            spin_array[x,y]=spin_f
            energy += dE
        elif dE<=0:
            spin_array[x,y]=spin_f
            energy += dE
            
        net_spins[t] = spin_array.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy, spin_array


def metropolis_3D_periodic(spin_array,boundaries, times, T, J,h):
    '''Function which takes in a 3D array of spins subjected to periodic boundary conditions, times =  number of iterations, temperature and an exchange energy J 
    and performs said iterations of the Metropolis Monte Carlo Algorithm to produce an array of net spins and energies,
    based on the Ising Model.
    Arguments:
        spin_array = 3D array representing lattice of spins
        boundaries = boundary conditions of lattice
        times = desired iterations of algorithm
        T = temperature (K)
        J = exchange interaction
        h = external magnetic field value
    Returns:
       net spins = 1D array of net spin of the latticefor each iteration
       net_energies = 1D array of net energies of the lattice for each iteration
       spin_ array = final lattice after all spin flips 
    '''
    energy = lattice_energy(spin_array,boundaries,J,h)
    k = sc.Boltzmann
    B = 1/(k*T) # Calculate beta value for specified temperature
    spin_array = spin_array.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        n_x ,n_y, n_z = spin_array.shape
        x = np.random.randint(0,n_x)
        y = np.random.randint(0,n_y)
        z = np.random.randint(0,n_z)
        spin_i = spin_array[x,y,z] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy applying periodic boundary conditions
        E_i = 0
        E_f = 0
        # Left element
        if x==0:
            E_i += -spin_i*spin_array[n_x-1,y,z]
            E_f += -spin_f*spin_array[n_x-1,y,z]
        if x>0:
            E_i += -spin_i*spin_array[x-1,y,z]
            E_f += -spin_f*spin_array[x-1,y,z]
        # Right Element
        if x<n_x-1:
            E_i += -spin_i*spin_array[x+1,y,z]
            E_f += -spin_f*spin_array[x+1,y,z]
        if x == n_x-1:
            E_i += -spin_i*spin_array[0,y,z]
            E_f += -spin_f*spin_array[0,y,z]
        # Below element
        if y==0:
            E_i += -spin_i*spin_array[x,n_y-1,z]
            E_f += -spin_f*spin_array[x,n_y-1,z]
        if y>0:
            E_i += -spin_i*spin_array[x,y-1,z]
            E_f += -spin_f*spin_array[x,y-1,z]
        # Above element
        if y<n_y-1:
            E_i += -spin_i*spin_array[x,y+1,z]
            E_f += -spin_f*spin_array[x,y+1,z]
        if y==n_y-1:
            E_i += -spin_i*spin_array[x,0,z]
            E_f += -spin_f*spin_array[x,0,z]
        # front neighbour
        if z==0:
            E_i += -spin_i*spin_array[x,y,n_z-1]
            E_f += -spin_f*spin_array[x,y,n_z-1]
        if z>0:
            E_i += -spin_i*spin_array[x,y,z-1]
            E_f += -spin_f*spin_array[x,y,z-1] 
        # behind neighbour
        if z<n_z-1:
            E_i += -spin_i*spin_array[x,y,z+1]
            E_f += -spin_f*spin_array[x,y,z+1]
        if z==n_z-1:
            E_i += -spin_i*spin_array[x,y,0]
            E_f += -spin_f*spin_array[x,y,0] 
        
        # 3 / 4. change state with designated probabilities
        E_i = J*E_i
        E_f = J*E_f
        
        dE = E_f-E_i
        P = np.random.random(1)[0] # Generate random number on interval [0,1)
        if (dE>0)*(P < np.exp(-B*dE)):
            spin_array[x,y,z]=spin_f
            energy += dE
        elif dE<=0:
            spin_array[x,y,z]=spin_f
            energy += dE
            
        net_spins[t] = spin_array.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy, spin_array


def metropolis_1D_reflective(spin_array,boundaries, times, T, J,h):
    '''Function which takes in a 1D array of spins subjected to reflective boundary conditions, times = number of iterations, temperature and an exchange energy J 
    and performs said iterations of the Metropolis Monte Carlo Algorithm to produce an array of net spins and energies,
    based on the Ising Model.
    Arguments:
        spin_array = 1D array representing lattice of spins
        boundaries = boundary conditions of lattice
        times = desired iterations of algorithm
        T = temperature (K)
        J = exchange interaction
        h = external magnetic field value
    Returns:
       net spins = 1D array of net spin of the latticefor each iteration
       net_energies = 1D array of net energies of the lattice for each iteration
       spin_ array = final lattice after all spin flips 
    '''
    energy = lattice_energy(spin_array,boundaries,J,h)
    k = sc.Boltzmann
    B = 1/(k*T) # Calculate beta value for specified temperature
    spin_array = spin_array.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        n_x = len(spin_array)
        x = np.random.randint(0,n_x)
        spin_i = spin_array[x] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy applying periodic boundary conditions
        E_i = 0
        E_f = 0
        # Left neighbour
        if x==0:
            E_i += -spin_i*spin_i
            E_f += -spin_f*spin_f
        if x>0:
            E_i += -spin_i*spin_array[x-1]
            E_f += -spin_f*spin_array[x-1]
        # Right neighbour
        if x<n_x-1:
            E_i += -spin_i*spin_array[x+1]
            E_f += -spin_f*spin_array[x+1]
        if x == n_x-1:
            E_i += -spin_i*spin_i
            E_f += -spin_f*spin_f
        
        # 3 / 4. change state with designated probabilities
        E_i = J*E_i
        E_f = J*E_f
        
        dE = E_f-E_i
        P = np.random.random(1)[0] # Generate random number on interval [0,1)
        if (dE>0)*(P < np.exp(-B*dE)):
            spin_array[x]=spin_f
            energy += dE
        elif dE<=0:
            spin_array[x]=spin_f
            energy += dE
            
        net_spins[t] = spin_array.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy, spin_array


def metropolis_2D_reflective(spin_array,boundaries, times, T, J,h):
    '''Function which takes in a 2D array of spins subjected to reflective boundary conditions, times = number of iterations, temperature and an exchange energy J 
    and performs said iterations of the Metropolis Monte Carlo Algorithm to produce an array of net spins and energies,
    based on the Ising Model.
    Arguments:
        spin_array = lattice of spins
        boundaries = boundary conditions of lattice
        times = desired iterations of algorithm
        T = temperature (K)
        J = exchange interaction
        h = external magnetic field value
    Returns:
       net spins = 1D array of net spin of the latticefor each iteration
       net_energies = 1D array of net energies of the lattice for each iteration
       spin_ array = final lattice after all spin flips 
    '''
    energy = lattice_energy(spin_array,boundaries,J,h)
    k = sc.Boltzmann
    B = 1/(k*T) # Calculate beta value for specified temperature
    spin_array = spin_array.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        n_x, n_y = spin_array.shape
        x = np.random.randint(0,n_x)
        y = np.random.randint(0,n_y)
        spin_i = spin_array[x,y] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy applying periodic boundary conditions
        E_i = 0
        E_f = 0
        # Left element
        if x==0:
            E_i += -spin_i*spin_i
            E_f += -spin_f*spin_f
        if x>0:
            E_i += -spin_i*spin_array[x-1,y]
            E_f += -spin_f*spin_array[x-1,y]
        # Right Element
        if x<n_x-1:
            E_i += -spin_i*spin_array[x+1,y]
            E_f += -spin_f*spin_array[x+1,y]
        if x == n_x-1:
            E_i += -spin_i*spin_i
            E_f += -spin_f*spin_f
        # Below element
        if y==0:
            E_i += -spin_i*spin_i
            E_f += -spin_f*spin_f
        if y>0:
            E_i += -spin_i*spin_array[x,y-1]
            E_f += -spin_f*spin_array[x,y-1]
        # Above element
        if y<n_y-1:
            E_i += -spin_i*spin_array[x,y+1]
            E_f += -spin_f*spin_array[x,y+1]
        if y==n_y-1:
            E_i += -spin_i*spin_i
            E_f += -spin_f*spin_f
        
        # 3 / 4. change state with designated probabilities
        E_i = J*E_i
        E_f = J*E_f
        
        dE = E_f-E_i
        P = np.random.random(1)[0] # Generate random number on interval [0,1)
        if (dE>0)*(P < np.exp(-B*dE)):
            spin_array[x,y]=spin_f
            energy += dE
        elif dE<=0:
            spin_array[x,y]=spin_f
            energy += dE
            
        net_spins[t] = spin_array.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy, spin_array


def metropolis_3D_reflective(spin_array,boundaries, times, T, J,h):
    '''Function which takes in a 3D array of spins subjected to preflective boundary conditions, times = number of iterations, temperature and an exchange energy J 
    and performs said iterations of the Metropolis Monte Carlo Algorithm to produce an array of net spins and energies,
    based on the Ising Model.
    Arguments:
        spin_array = 3D array representing lattice of spins
        boundaries = boundary conditions of lattice
        times = desired iterations of algorithm
        T = temperature (K)
        J = exchange interaction
        h = external magnetic field value
    Returns:
       net spins = 1D array of net spin of the latticefor each iteration
       net_energies = 1D array of net energies of the lattice for each iteration
       spin_ array = final lattice after all spin flips 
    '''
    energy = lattice_energy(spin_array,boundaries,J,h)
    k = sc.Boltzmann
    B = 1/(k*T) # Calculate beta value for specified temperature
    spin_array = spin_array.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        n_x ,n_y, n_z = spin_array.shape
        x = np.random.randint(0,n_x)
        y = np.random.randint(0,n_y)
        z = np.random.randint(0,n_z)
        spin_i = spin_array[x,y,z] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy applying periodic boundary conditions
        E_i = 0
        E_f = 0
        # Left element
        if x==0:
            E_i += -spin_i*spin_i
            E_f += -spin_f*spin_f
        if x>0:
            E_i += -spin_i*spin_array[x-1,y,z]
            E_f += -spin_f*spin_array[x-1,y,z]
        # Right Element
        if x<n_x-1:
            E_i += -spin_i*spin_array[x+1,y,z]
            E_f += -spin_f*spin_array[x+1,y,z]
        if x == n_x-1:
            E_i += -spin_i*spin_i
            E_f += -spin_f*spin_f
        # Below element
        if y==0:
            E_i += -spin_i*spin_i
            E_f += -spin_f*spin_f
        if y>0:
            E_i += -spin_i*spin_array[x,y-1,z]
            E_f += -spin_f*spin_array[x,y-1,z]
        # Above element
        if y<n_y-1:
            E_i += -spin_i*spin_array[x,y+1,z]
            E_f += -spin_f*spin_array[x,y+1,z]
        if y==n_y-1:
            E_i += -spin_i*spin_i
            E_f += -spin_f*spin_f
        # front neighbour
        if z==0:
            E_i += -spin_i*spin_i
            E_f += -spin_f*spin_f
        if z>0:
            E_i += -spin_i*spin_array[x,y,z-1]
            E_f += -spin_f*spin_array[x,y,z-1] 
        # behind neighbour
        if z<n_z-1:
            E_i += -spin_i*spin_array[x,y,z+1]
            E_f += -spin_f*spin_array[x,y,z+1]
        if z==n_z-1:
            E_i += -spin_i*spin_i
            E_f += -spin_f*spin_f
        
        # 3 / 4. change state with designated probabilities
        E_i = J*E_i
        E_f = J*E_f
        
        dE = E_f-E_i
        P = np.random.random(1)[0] # Generate random number on interval [0,1)
        if (dE>0)*(P < np.exp(-B*dE)):
            spin_array[x,y,z]=spin_f
            energy += dE
        elif dE<=0:
            spin_array[x,y,z]=spin_f
            energy += dE
            
        net_spins[t] = spin_array.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy, spin_array


def metropolis_1D_open(spin_array, boundaries, times, T, J,h):
    '''Function which takes in a 1D array of spins subjected to open boundary conditions, times = number of iterations, temperature and an exchange energy J 
    and performs said iterations of the Metropolis Monte Carlo Algorithm to produce an array of net spins and energies,
    based on the Ising Model.
    Arguments:
        spin_array = 1D array representing lattice of spins
        boundaries = boundary conditions of lattice
        times = desired iterations of algorithm
        T = temperature (K)
        J = exchange interaction
        h = external magnetic field value
    Returns:
       net spins = 1D array of net spin of the latticefor each iteration
       net_energies = 1D array of net energies of the lattice for each iteration
       spin_ array = final lattice after all spin flips 
    '''
    energy = lattice_energy(spin_array,boundaries,J,h)
    k = sc.Boltzmann
    B = 1/(k*T) # Calculate beta value for specified temperature
    spin_array = spin_array.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        n_x = len(spin_array)
        x = np.random.randint(0,n_x)
        spin_i = spin_array[x] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy applying periodic boundary conditions
        E_i = 0
        E_f = 0
        # Left neighbour
        if x==0:
            E_i += -spin_i*(0)
            E_f += -spin_f*(0)
        if x>0:
            E_i += -spin_i*spin_array[x-1]
            E_f += -spin_f*spin_array[x-1]
        # Right neighbour
        if x<n_x-1:
            E_i += -spin_i*spin_array[x+1]
            E_f += -spin_f*spin_array[x+1]
        if x == n_x-1:
            E_i += -spin_i*(0)
            E_f += -spin_f*(0)
        
        # 3 / 4. change state with designated probabilities
        E_i = J*E_i
        E_f = J*E_f
        
        dE = E_f-E_i
        P = np.random.random(1)[0] # Generate random number on interval [0,1)
        if (dE>0)*(P < np.exp(-B*dE)):
            spin_array[x]=spin_f
            energy += dE
        elif dE<=0:
            spin_array[x]=spin_f
            energy += dE
            
        net_spins[t] = spin_array.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy, spin_array


def metropolis_2D_open(spin_array,boundaries, times, T, J,h):
    '''Function which takes in a 2D array of spins subjected to open boundary conditions, times = number of iterations, temperature and an exchange energy J 
    and performs said iterations of the Metropolis Monte Carlo Algorithm to produce an array of net spins and energies,
    based on the Ising Model.
    Arguments:
        spin_array = lattice of spins
        boundaries = boundary conditions of lattice
        times = desired iterations of algorithm
        T = temperature (K)
        J = exchange interaction
        h = external magnetic field value
    Returns:
       net spins = 1D array of net spin of the latticefor each iteration
       net_energies = 1D array of net energies of the lattice for each iteration
       spin_ array = final lattice after all spin flips 
    '''
    energy = lattice_energy(spin_array,boundaries,J,h)
    k = sc.Boltzmann
    B = 1/(k*T) # Calculate beta value for specified temperature
    spin_array = spin_array.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        n_x, n_y = spin_array.shape
        x = np.random.randint(0,n_x)
        y = np.random.randint(0,n_y)
        spin_i = spin_array[x,y] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy applying periodic boundary conditions
        E_i = 0
        E_f = 0
        # Left element
        if x==0:
            E_i += -spin_i*(0)
            E_f += -spin_f*(0)
        if x>0:
            E_i += -spin_i*spin_array[x-1,y]
            E_f += -spin_f*spin_array[x-1,y]
        # Right Element
        if x<n_x-1:
            E_i += -spin_i*spin_array[x+1,y]
            E_f += -spin_f*spin_array[x+1,y]
        if x == n_x-1:
            E_i += -spin_i*(0)
            E_f += -spin_f*(0)
        # Below element
        if y==0:
            E_i += -spin_i*(0)
            E_f += -spin_f*(0)
        if y>0:
            E_i += -spin_i*spin_array[x,y-1]
            E_f += -spin_f*spin_array[x,y-1]
        # Above element
        if y<n_y-1:
            E_i += -spin_i*spin_array[x,y+1]
            E_f += -spin_f*spin_array[x,y+1]
        if y==n_y-1:
            E_i += -spin_i*(0)
            E_f += -spin_f*(0)
        
        # 3 / 4. change state with designated probabilities
        E_i = J*E_i
        E_f = J*E_f
        
        dE = E_f-E_i
        P = np.random.random(1)[0] # Generate random number on interval [0,1)
        if (dE>0)*(P < np.exp(-B*dE)):
            spin_array[x,y]=spin_f
            energy += dE
        elif dE<=0:
            spin_array[x,y]=spin_f
            energy += dE
            
        net_spins[t] = spin_array.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy, spin_array


def metropolis_3D_open(spin_array,boundaries, times, T, J,h):
    '''Function which takes in a 3D array of spins subjected to open boundary conditions, times = number of iterations, temperature and an exchange energy J 
    and performs said iterations of the Metropolis Monte Carlo Algorithm to produce an array of net spins and energies,
    based on the Ising Model.
    Arguments:
        spin_array = 3D array representing lattice of spins
        boundaries = boundary conditions of lattice
        times = desired iterations of algorithm
        T = temperature (K)
        J = exchange interaction
        h = external magnetic field value
    Returns:
       net spins = 1D array of net spin of the latticefor each iteration
       net_energies = 1D array of net energies of the lattice for each iteration
       spin_ array = final lattice after all spin flips 
    '''
    energy = lattice_energy(spin_array,boundaries,J,h)
    k = sc.Boltzmann
    B = 1/(k*T) # Calculate beta value for specified temperature
    spin_array = spin_array.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0,times-1):
        # 2. pick random point on array and flip spin
        n_x ,n_y, n_z = spin_array.shape
        x = np.random.randint(0,n_x)
        y = np.random.randint(0,n_y)
        z = np.random.randint(0,n_z)
        spin_i = spin_array[x,y,z] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        # compute change in energy applying periodic boundary conditions
        E_i = 0
        E_f = 0
        # Left element
        if x==0:
            E_i += -spin_i*(0)
            E_f += -spin_f*(0)
        if x>0:
            E_i += -spin_i*spin_array[x-1,y,z]
            E_f += -spin_f*spin_array[x-1,y,z]
        # Right Element
        if x<n_x-1:
            E_i += -spin_i*spin_array[x+1,y,z]
            E_f += -spin_f*spin_array[x+1,y,z]
        if x == n_x-1:
            E_i += -spin_i*(0)
            E_f += -spin_f*(0)
        # Below element
        if y==0:
            E_i += -spin_i*(0)
            E_f += -spin_f*(0)
        if y>0:
            E_i += -spin_i*spin_array[x,y-1,z]
            E_f += -spin_f*spin_array[x,y-1,z]
        # Above element
        if y<n_y-1:
            E_i += -spin_i*spin_array[x,y+1,z]
            E_f += -spin_f*spin_array[x,y+1,z]
        if y==n_y-1:
            E_i += -spin_i*(0)
            E_f += -spin_f*(0)
        # front neighbour
        if z==0:
            E_i += -spin_i*(0)
            E_f += -spin_f*(0)
        if z>0:
            E_i += -spin_i*spin_array[x,y,z-1]
            E_f += -spin_f*spin_array[x,y,z-1] 
        # behind neighbour
        if z<n_z-1:
            E_i += -spin_i*spin_array[x,y,z+1]
            E_f += -spin_f*spin_array[x,y,z+1]
        if z==n_z-1:
            E_i += -spin_i*(0)
            E_f += -spin_f*(0)
        
        # 3 / 4. change state with designated probabilities
        E_i = J*E_i
        E_f = J*E_f
        
        dE = E_f-E_i
        P = np.random.random(1)[0] # Generate random number on interval [0,1)
        if (dE>0)*(P < np.exp(-B*dE)):
            spin_array[x,y,z]=spin_f
            energy += dE
        elif dE<=0:
            spin_array[x,y,z]=spin_f
            energy += dE
            
        net_spins[t] = spin_array.sum()
        net_energy[t] = energy
            
    return net_spins, net_energy, spin_array

    
def thermo_calc_1D(min_temp, max_temp, system, boundaries, times, J, h):
    ''' Calculates the totals and standard deviations of energies and spins over a range of temperatures 
    using the Metropolis Monte Carlo Algortithm and finds the associated thermodynamic quantities.
    Arguments:
        min_temp = minimum temperature in range
        max_temp = maximum temperature in range
        system = array representing lattice of spins
        boundaries = boundary conditions of lattice
        times = desired iterations of the metropolis algorithm
        J = exchange interaction    
        h =  external magnetic field value
    Returns:
       E_totals = array of energy means
       C = array of Specific heat capacities
       S_totals = array of spin means
       Chi = array of Magnetic Susceptibilities times number of spins in lattice 
    '''
    k = sc.Boltzmann
    # Create array for desired temperature range
    temps = np.arange(min_temp, max_temp,5)
    n = len(temps)
    # Create blank arrays for totals and variances
    E_totals = np.zeros(n)
    E_vars = np.zeros(n)
    C = np.zeros(n)
    N = len(system)
    
    S_totals = np.zeros(n)
    S_vars = np.zeros(n)
    chi = np.zeros(n)
    
    # Iterating over all temperatures in desired range, use metropolis function to find the spins and energies
    for i in range(n):
        if boundaries == 'wrap':
            spins, energies, lattice_final = metropolis_1D_periodic(system,boundaries,times,temps[i],J,h) # Run Metropolis for varying temp
        elif boundaries == 'reflect':
            spins, energies, lattice_final = metropolis_1D_reflective(system,boundaries,times,temps[i],J,h) # Run Metropolis for varying temp
        elif boundaries == 'constant':
            spins, energies, lattice_final = metropolis_1D_open(system,boundaries,times,temps[i],J,h) # Run Metropolis for varying temp 
        
        E_totals[i] += np.mean(energies[-10000:])/(N) # Calculate total energy of lattice at each temperature
        E_vars[i] += np.var(energies[-10000:]) # Calculate variance of energies at each temperature
        C[i] = E_vars[i]/(N)*k*(temps[i]**2) # Specific Heat Capacity from energy variance
    
        S_totals[i] += np.mean(spins[-10000:])/(N) # Calculate total spin of lattice at each temperature
        S_vars[i] += np.var(spins[-10000:]) # Calculate variance of spins at each temperature
        chi[i] += S_vars[i]/((N)*k*temps[i]) # Calculate Susceptibility from spin variance
        
    return E_totals, C, S_totals, chi


def thermo_calc_2D(min_temp, max_temp, system, boundaries, times, J, h):
    ''' Calculates the totals and standard deviations of energies and spins over a range of temperatures 
    using the Metropolis Monte Carlo Algortithm and finds the associated thermodynamic quantities.
    Arguments:
        min_temp = minimum temperature in range
        max_temp = maximum temperature in range
        system = array representing lattice of spins
        boundaries = boundary conditions of lattice
        times = desired iterations of the metropolis algorithm
        J = exchange interaction    
        h =  external magnetic field value
    Returns:
       E_totals = array of energy means
       C = array of Specific heat capacities
       S_totals = array of spin means
       Chi = array of Magnetic Susceptibilities times number of spins in lattice 
    '''
    k = sc.Boltzmann
    # Create array for desired temperature range
    temps = np.arange(min_temp, max_temp,5)
    n = len(temps)
    # Create blank arrays for totals and variances
    E_totals = np.zeros(n)
    E_vars = np.zeros(n)
    C = np.zeros(n)
    n_x,n_y = system.shape
    N  = (n_x)*(n_y)
    
    S_totals = np.zeros(n)
    S_vars = np.zeros(n)
    chi = np.zeros(n)
    
    # Iterating over all temperatures in desired range, use metropolis function to find the spins and energies
    for i in range(n):
        if boundaries == 'wrap':
            spins, energies, lattice_final = metropolis_2D_periodic(system,boundaries,times,temps[i],J,h) # Run Metropolis for varying temp
        elif boundaries == 'reflect':
            spins, energies, lattice_final = metropolis_2D_reflective(system,boundaries,times,temps[i],J,h) # Run Metropolis for varying temp
        elif boundaries == 'constant':
            spins, energies, lattice_final = metropolis_2D_open(system,boundaries,times,temps[i],J,h) # Run Metropolis for varying temp 
        
        E_totals[i] += np.mean(energies[-10000:])/N # Calculate total energy of lattice at each temperature
        E_vars[i] += np.var(energies[-10000:]) # Calculate variance of energies at each temperature
        C[i] = E_vars[i]/N*k*(temps[i]**2) # Specific Heat Capacity from energy variance
    
        S_totals[i] += np.mean(spins[-10000:])/((n_x)*(n_y)) # Calculate total spin of lattice at each temperature
        S_vars[i] += np.var(spins[-10000:]) # Calculate variance of spins at each temperature
        chi[i] += S_vars[i]/(N*k*temps[i]) # Calculate Susceptibility from spin variance
        
    return E_totals, C, S_totals, chi


def thermo_calc_3D(min_temp, max_temp, system, boundaries, times, J, h):
    ''' Calculates the totals and standard deviations of energies and spins over a range of temperatures 
    using the Metropolis Monte Carlo Algortithm and finds the associated thermodynamic quantities.
    Arguments:
        min_temp = minimum temperature in range
        max_temp = maximum temperature in range
        system = array representing lattice of spins
        boundaries = boundary conditions of lattice
        times = desired iterations of the metropolis algorithm
        J = exchange interaction    
        h =  external magnetic field value
    Returns:
       E_totals = array of energy means
       C = array of Specific heat capacities
       S_totals = array of spin means
       Chi = array of Magnetic Susceptibilities times number of spins in lattice 
    '''
    k = sc.Boltzmann
    # Create array for desired temperature range
    temps = np.arange(min_temp, max_temp,5)
    n = len(temps)
    # Create blank arrays for totals and variances
    E_totals = np.zeros(n)
    E_vars = np.zeros(n)
    C = np.zeros(n)
    n_x,n_y,n_z = system.shape
    N  = (n_x)*(n_y)*(n_z)
    
    S_totals = np.zeros(n)
    S_vars = np.zeros(n)
    chi = np.zeros(n)
    
    # Iterating over all temperatures in desired range, use metropolis function to find the spins and energies
    for i in range(n):
        if boundaries == 'wrap':
            spins, energies, lattice_final = metropolis_3D_periodic(system,boundaries,times,temps[i],J,h) # Run Metropolis for varying temp
        elif boundaries == 'reflect':
            spins, energies, lattice_final = metropolis_3D_reflective(system,boundaries,times,temps[i],J,h) # Run Metropolis for varying temp
        elif boundaries == 'constant':
            spins, energies, lattice_final = metropolis_3D_open(system,boundaries,times,temps[i],J,h) # Run Metropolis for varying temp 
        
        E_totals[i] += np.mean(energies[-10000:])/N # Calculate total energy of lattice at each temperature
        E_vars[i] += np.var(energies[-10000:]) # Calculate variance of energies at each temperature
        C[i] = E_vars[i]/N*k*(temps[i]**2) # Specific Heat Capacity from energy variance
    
        S_totals[i] += np.mean(spins[-10000:])/((n_x)*(n_y)) # Calculate total spin of lattice at each temperature
        S_vars[i] += np.var(spins[-10000:]) # Calculate variance of spins at each temperature
        chi[i] += S_vars[i]/(N*k*temps[i]) # Calculate Susceptibility from spin variance
        
    return E_totals, C, S_totals, chi