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


