import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def initialize_matrices(N, h, J, k):
    # Initialize A and B as N x N zero matrices
    A = np.zeros((N, N))
    B = np.zeros((N, N))

    # Fill A and B with given conditions
    for j in range(N):
        A[j, j] = h  # Diagonal elements of A

        if j < N - 1:
            A[j, j+1] = A[j+1, j] = -J/2  # Symmetric off-diagonal
            B[j, j+1] = -k*J/2
            B[j+1, j] = k*J/2  # Antisymmetric off-diagonal

    # Construct the block matrix H
    H = np.block([[A, B], [-B, -A]])

    return A, B, H

# Parameters
N = 4
J = 1.0
k = 1
h_list = np.arange(0.1,2,0.1)
eigenvalues_list = []

for h in h_list:
    print(h)
    A, B, H = initialize_matrices(N, h, J, k)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = scipy.linalg.eig(H)

    # Extract real parts
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Separate positive and negative eigenvalues with indices
    pos_indices = np.where(eigenvalues > 0)[0]
    neg_indices = np.where(eigenvalues < 0)[0]

    # Sort positive eigenvalues in descending order
    pos_sorted_indices = pos_indices[np.argsort(-eigenvalues[pos_indices])]

    # Sort negative eigenvalues by magnitude (largest first, keeping them negative)
    neg_sorted_indices = neg_indices[np.argsort(-np.abs(eigenvalues[neg_indices]))]

    # Combine sorted eigenvalues in the desired order
    sorted_eigenvalues = np.concatenate((eigenvalues[pos_sorted_indices], eigenvalues[neg_sorted_indices]))

    # Store result
    eigenvalues_list.append(2*sorted_eigenvalues)

    # Create matrices U and V
    # U = eigenvectors[:, pos_sorted_indices]  # Eigenvectors for positive eigenvalues
    # V = eigenvectors[:, neg_sorted_indices]  # Eigenvectors for negative eigenvalues
    # P = np.block([U, V])
    # P_inv = scipy.linalg.inv(P)
    # Print results
    # print(len(eigenvalues))
    # print(eigenvectors)
    # print("Sorted Eigenvalues:", len(sorted_eigenvalues))
    # print("\nMatrix U (Positive Eigenvectors):\n", U)
    # print("\nMatrix V (Negative Eigenvectors):\n", V)  
    # print("Matrix P: \n", np.round(P,2))
    # print("Matrix P_inv: \n", np.round(P_inv,2))
    # print("Product: \n",np.round(P_inv@P,2))
    # print("H_reconstruct: \n", np.round(P@np.diag(sorted_eigenvalues)@P_inv,2))
    # print("H_original: \n", np.round(H,2))

plt.plot(h_list, eigenvalues_list) 

plt.xlabel(r'$h/J$')
plt.ylabel(r'$\varepsilon/J$', rotation = 0)
plt.ylim(0,3)
plt.title('Eigenvalues vs. $h/J$')
plt.legend()
plt.grid(True)
plt.show()