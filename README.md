# Metropolis Monte Carlo Simulation for the Ising Model (python)
## Overview
This python program produces a Monte Carlo simulation for the Ising model in 1D, 2D, and 3D lattices, utilising the Metropolis-Hastings algorithm. The program allows the user to determine the dimensions and external conditions of the lattice, as well as the parameters of the algorithm. The Metropolis-Hastings algorithm then generates a range of thermodynamic data from the simulation, from which the user can identify the point at which the lattice undergoes ferromagnetic phase transiton.

## Table of Contents
1. Installation
2. Usage
3. Functions
4. Examples
5. Contributing
6. License

## Installation 
Use the package manager pip to install both the NumPy and SciPy libraries:
```bash
pip install numpy scipy
```
## Usage
The user will first be prompted to input the desired number of dimensions for the lattice they wish to simulate, 1 for 1D, 2 for 2D, or 3 for 3D, any other input will register as invalid and restart the prompt. The generate_lattice function is then called, prompting the user to input the desired number of spins along each dimension of their chosen lattice, note that aonly integer values are accepted and the number of spins along each dimension does not have to be identical. 
```python
while True:
    dimensionality = int(input("Enter the dimensionality of the lattice (1 for 1D, 2 for 2D, 3 for 3D): "))
    lattice = generate_lattice(dimensionality)
 print("Lattice Array:")
```
However, note that there are a minimum number of spins that must be input based on the selected dimensionality of the lattice, otherwise the metropolis algorithm would not be able to function.

```python
 if dimensionality == 1 and lattice.size < 3:
        print("Error: 1D lattice must have at least 3 objects.")
        continue  # Restart the loop to prompt for dimensionality again
    elif dimensionality == 2 and (lattice.shape[0] < 3 or lattice.shape[1] < 3):
        print("Error: 2D lattice must have at least dimensions of 3x3.")
        continue  # Restart the loop to prompt for dimensionality again
    elif dimensionality == 3 and (lattice.shape[0] < 3 or lattice.shape[1] < 3 or lattice.shape[2] < 3):
        print("Error: 3D lattice must have at least dimensions of 3x3x3.")
        continue  # Restart the loop to prompt for dimensionality again
```
Provided all the prior inputs are valid, the program will produce a plot for the lattice, at random, 75% of the spins in the lattice will be allocated as +1 and 25% as -1, as the initial lattice must be in a ferromagnetic state before the algorithm is applied.

```python
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
```

As the algorithm works based on selecting a random spin and then summing over the spin values of its nearest neighbours, the user must select a boundary condition for when a spin at the edge of the lattice is selected. The program offers the user 3 options:

 1. Periodic - the lattice wraps back around to the opposite side, as if it repeats itself along this dimension.
 2. Reflective - the edge of the lattice is reflected, so the nearest neighbour effectively becomes the selected spin itself.
 3. Open - the lattice is assumed to be surrounded by no interacting spins, so the nearest neighbour contribution at the boundary is taken to be zero.

```python
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
```

The user will then be prompted to initialise the lattice's environment and apply the algorithm once for testing purposes. A good example temperature to start with would be 1000K, while the value of the external magnetic field contribution (h) will depend on the specific environment being simulated. However, 0 is an acceptable value for testing the simulation in the absence of an external field. For the number of iterations of the algorithm, a good minimum for 1D lattices would be 10,000, multiplying by a factor of 10 with each added dimension. The program will then produce two plots showing the evolution of the lattice's net spin and net energy over the specified number of iterations. Note that all inputs must be in integer form.

```python
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

```
The final user prompt requests a minimum and maximum temperature, the algorithm will then be applied over this temperature range to produce thermodynamic data, such as the lattice's net energy, magnetisation (net spins), specfic heat capacity and magnetic susceptibility. Again, all inputs must be in integer form.

```python
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
```

## Functions

There are 5 primary types of function the program utilises. The first of which is the generate_lattice() function, which takes in an integer for the dimensionaity and  produces a NumPy array to simulate the lattice of spins described by the Ising model:


```python
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
```

The second type of function is simply used to visualise the NumPy array using Matplotlib. There are 3 variations of this function, each corresponding to the dimensionality of the input array, for example, the function used to visualise a 2D lattice array: 

```python
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
```

The third type of function calculates the energy of of the lattice array based on the configuration of its spins. The nearest neighbour summation of the Ising model is replicated by convolving the lattice array with a boolean array to represent the neighbouring spin contributions. This caclulation is integral to the Metropolis-Hastings algorithm, but for reasons that will become clear, it is preferable to have a separate function for this calculation: 

```python
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
```    

The next type of function makes up the majority of the code, as it requires a unique variation for each combination of lattice dimensionality and boundary conditions. These are the Metropolis-Hastings algorithm functions, all of which calculates the energy of each lattice configuration for the number of iterations specified by the user. For example, we have the Metropolis function for a 3D lattice subjected to periodic boundary conditions: 

```python
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
```  

The final type of function applies the Metropolis algorithm function for each temperature in a range specified by the user, generating 4 distinct 1D NumPy arrays containing thermodynamic data. By plotting this data, the temperature at which the lattice undergoes ferromagnetic phase transition can be idenitified. The process necessitates a version of this function for each possible dimensionality of the lattice. For example, in the case of a 3D lattice,the function takes the form: 

```python
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
```

## Examples




















