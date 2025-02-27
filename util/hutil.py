import numpy as np
from tqdm import tqdm

def recursive_linear_system(A, w, N, T, tolerance=1e-8, verbose=False):

    A = np.array(A)
    w = np.array(w)

    z = np.zeros(len(w))

    if A.shape[0] < N or A.shape[1] < N:
        A_padded = np.zeros((N, N))
        A_padded[:A.shape[0], :A.shape[1]] = A
        A = A_padded

    if w.shape[0] < N:
        w = np.concatenate([w, np.zeros(N - w.shape[0])])

    if z.shape[0] < N:
        z = np.concatenate([z, np.zeros(N - z.shape[0])])

    I = np.eye(N)
    try:
        z_star = np.linalg.solve(I - A, w)
    except np.linalg.LinAlgError:
        if verbose:
            print("Singular problem, check matrix.")
        z_star = None

    errors = [] 
    for t in tqdm(range(1, T + 1), "Loopting through.."):
        z = A @ z + w

        if z_star is not None:
            # Torodov paper approximation error
            approx_error = np.mean(np.abs(z - z_star)) / np.mean(z_star)
            errors.append(approx_error)
            if verbose:
                print(f"Iteration: {t}\n z = {z}\n Approx. Error: {approx_error}")

            if approx_error < tolerance:
                print(f"Converged at {t} iterations; Approx. error {approx_error} < {tolerance}.")
                break
    return z_star, z, errors

def ckks_bootstrap(ciph, evaluator, rot_keys, conj_key, relin_key, encoder):
    _, ciph_boot = evaluator.bootstrap(
        ciph,
        rot_keys, 
        conj_key,
        relin_key,
        encoder
    )
    ciph = ciph_boot
    return ciph

def compute_inner_product(ciph1, ciph2, Nslots, encoder, \
                    scaling_factor, evaluator, i_rot_keys, b_rot_keys, conj_key, relin_key):
    
    ciph_ip = evaluator.multiply(ciph1, ciph2, relin_key)

    rot = 1
    for k in range(int(np.log2(Nslots))):
        ciph_rot = evaluator.rotate(ciph_ip, rot, i_rot_keys[k])
        ciph_ip = evaluator.add(ciph_ip, ciph_rot)
        rot <<= 1
        
    ciph_ip = evaluator.rescale(ciph_ip, scaling_factor)
    ciph_ip = ckks_bootstrap(ciph_ip, evaluator, b_rot_keys, conj_key, relin_key, encoder)

    return ciph_ip