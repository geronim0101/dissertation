import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Pauli matrices
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])

def initialize_state(L):
    result = [np.array([0,1]) for _ in range(L)]
    return result

def kron_n(*args):
    """Compute the Kronecker product of multiple matrices."""
    result = args[0]
    for mat in args[1:]:
        result = np.kron(result, mat)
    return result

def construct_hamiltonian(L, Jx, Jy, h):
    """Construct the nearest-neighbor Hamiltonian matrix with periodic boundary conditions."""
    dim = 2**L  # Total Hilbert space dimension
    H = sp.csr_matrix((dim, dim), dtype=np.complex128)

    for j in range(L):  # Interaction terms (including PBC)
        j_next = (j + 1) % L  # Wrap-around for periodic boundary condition

        if j == L - 1:  # Special case: periodic interaction (L-1,0)
            sx_j_sx_j1 = kron_n(sx, *[np.eye(2)] * (L - 2), sx)
            sy_j_sy_j1 = kron_n(sy, *[np.eye(2)] * (L - 2), sy)
        else:  # Standard nearest-neighbor interactions
            sx_j_sx_j1 = kron_n(
                *[np.eye(2)] * j, sx, sx, *[np.eye(2)] * (L - j - 2)
            )
            sy_j_sy_j1 = kron_n(
                *[np.eye(2)] * j, sy, sy, *[np.eye(2)] * (L - j - 2)
            )

        H += -Jx[j] * sp.csr_matrix(sx_j_sx_j1)
        H += -Jy[j] * sp.csr_matrix(sy_j_sy_j1)

    for j in range(L):  # On-site field terms
        sz_j = kron_n(*[np.eye(2)] * j, sz, *[np.eye(2)] * (L - j - 1))
        H += -h[j] * sp.csr_matrix(sz_j)
    
    return H.toarray()

if __name__ == "__main__":
    # Example parameters
    L = 2  # Number of spins
    Jx = np.ones(L)  # Interaction strengths J_x (extended to L for PBC)
    Jy = np.ones(L)  # Interaction strengths J_y
    # h = np.random.rand(L)  # Random local fields
    h = [0.5 for i in range(L)]
    print(Jx, Jy, h)

    upup = np.kron(np.array([1,0]),np.array([1,0]))
    updown = np.kron(np.array([1,0]),np.array([0,1]))
    downup = np.kron(np.array([0,1]),np.array([1,0]))
    downdown = np.kron(np.array([0,1]),np.array([0,1]))

    basis = [upup, updown, downup, downdown]

    mult_matrix = np.zeros((L**2,L**2))
    for i in range(L**2):
        for j in range(L**2):
            mult_matrix[i,j] = np.dot(basis[i],basis[j])


    # Construct the Hamiltonian with periodic boundary conditions
    H = construct_hamiltonian(L, Jx, Jy, h)
    for b in basis:
        print(b,np.dot(b,H@b))
    
