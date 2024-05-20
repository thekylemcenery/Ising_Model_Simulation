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

Finally, the user will be prompted to initialise the lattice's environment and apply the algorithm. A good example temperature to start with would be 1000K, while the value of the external magnetic field contribution (h) will depend on the specific environment being simulated. However, 0 is an acceptable value for testing the simulation in the absence of an external field. For the number of iterations of the algorithm, a good minimum for 1D lattices would be 10,000, multiplying by a factor of 10 with each added dimension.

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

## Functions

There are 5 main types of function the program utilises. The first type is the generate_lattice() function, which takes in an integer for the dimensionaity and  produces a NumPy array to simulate the lattice of spins described by the Ising model:


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




## Examples




















