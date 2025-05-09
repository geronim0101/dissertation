import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Pauli matrices
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
I2 = np.eye(2)

def kron_n(*args):
    """Compute the Kronecker product of multiple matrices."""
    result = args[0]
    for mat in args[1:]:
        result = np.kron(result, mat)
    return result

def jordan_wigner(L):
    """Construct the Jordan-Wigner transformation for a system of L spins."""
    dim = 2**L
    a = []  # Fermionic annihilation operators
    adag = []  # Fermionic creation operators

    for j in range(L):
        # Compute the Jordan-Wigner string (product of all sz before site j)
        jw_string = [sz] * j    
        
        # Define annihilation and creation operators
        aj = kron_n(*jw_string, (sx - 1j * sy) / 2, *[I2] * (L - j - 1))
        adag_j = kron_n(*jw_string, (sx + 1j * sy) / 2, *[I2] * (L - j - 1))
        
        a.append(sp.csr_matrix(aj))
        adag.append(sp.csr_matrix(adag_j))
    
    return a, adag

def apply_operator(op, state):
    """Apply a matrix operator to a given quantum state."""
    return op @ state

# Example usage
L = 4  # Number of spins
annihilation_ops, creation_ops = jordan_wigner(L)

# Define an initial state (e.g., |0001⟩ in computational basis)
initial_state = np.zeros(2**L)
initial_state[-1] = 1  # Set last basis state to 1

# Apply the first annihilation operator
new_state = apply_operator(creation_ops[1], initial_state)

# Print results
print("Initial state:")
print(initial_state)
print("\nState after applying a_0:")
print(new_state)
