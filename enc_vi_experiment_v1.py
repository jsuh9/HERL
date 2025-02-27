import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import *
from util.lmdp import GridWorldMDP
from util.hutil import *
from ckks_base.ckks import *

grid_size = 3 # for (gird_size - by - grid_size) grid world
goal_state_loc = (2, 2) # (2, 2) grid is the goal
trap_state_loc = (2, 0) # (2, 0) is trap
lamb = 10.0

max_iterations = 100
tol = 1e-6

mdp = GridWorldMDP(size=grid_size, goal_state=goal_state_loc, trap_state=trap_state_loc, lambda_reg=lamb)
Z_star = mdp.solve_Z_star()
Z_star = mdp.solve_Z_star_vi(max_iter = max_iterations, tolerance = tol)
A, w = mdp.construct_A_w()

print(f"\n\nMatrix A has a shape: {A.shape}")
print(f"Vector w has a shape: {w.shape}")

eigenvalues = np.linalg.eigvals(A)
print("\n\nEigvals of A:", eigenvalues)
max_abs_eig = np.max(np.abs(eigenvalues))
print("Max. abs. eigval:", max_abs_eig)
print("Schur Stable?", max_abs_eig < 1)
print("\n")

A = A.tolist()
w = w.tolist()

degree = 16

Nslots = degree // 2 # number of slots for messaages

T = 20  # Number of iterations
z_star, z_final, errors = recursive_linear_system(A, w, Nslots, T, tolerance=1e-6)

# Initialize parameter set
params_set_meta = []

ciph_modulus_set = [100]  # Bits for ciph_modulus
scaling_factors_set = [40]  # Bits for scaling_factor

# Other fixed experimental parameters
degree = 16  # Polynomial degree
big_modulus = 1 << 1800  # Fixed big modulus

# Generate valid parameter sets
for ciph_modulus_bits in ciph_modulus_set:
    for scaling_factor_bits in scaling_factors_set:
        # Convert bit-lengths to values
        ciph_modulus = 1 << ciph_modulus_bits
        scaling_factor = 1 << scaling_factor_bits
        
        # Check if ciph_modulus is at least twice the scaling_factor
        if ciph_modulus_bits >= 2 * scaling_factor_bits:
            # Create a parameter set
            params_meta = {
                "poly_degree": degree,
                "ciph_modulus": ciph_modulus,
                "big_modulus": big_modulus,
                "scaling_factor": scaling_factor
            }
            params_set_meta.append(params_meta)

            # Print detailed information
            print(f"\nValid Parameters Added:")
            print(f"  Polynomial Degree: {params_meta['poly_degree']}")
            print(f"  Ciphertext Modulus: {params_meta['ciph_modulus']} (2^{ciph_modulus_bits} bits)")
            # print(f"  Big Modulus: {params_meta['big_modulus']} (2^1800 bits)")
            print(f"  Big Modulus: ... (2^1800 bits)")
            print(f"  Scaling Factor: {params_meta['scaling_factor']} (2^{scaling_factor_bits} bits)")
            print("-" * 50)

# Summary of generated parameter sets
print(f"\nGenerated {len(params_set_meta)} valid parameter sets.")

def simulate_encrypted_linear_VI(key_generator, encoder, encryptor, evaluator, relin_key, decryptor, first_iter=True, to_bootstrap=None):

    # key prepd
    rot = 1
    ip_rot_keys = []
    for k in range(int(np.log2(Nslots))):
        ip_rot_keys.append(key_generator.generate_rot_key(rot))
        rot <<= 1
        
    boot_rot_keys = {} # bootstrapping rotation keys
    for i in range(degree // 2):
        boot_rot_keys[i] = key_generator.generate_rot_key(i)

    conj_key = key_generator.generate_conj_key()


    if first_iter == True:
        #-------------------------------- Client Side ----------------------------------------
        zero_vector = [0] * Nslots
        zero_plain = encoder.encode(zero_vector, scaling_factor)
        ciph_sum = encryptor.encrypt(zero_plain)
        ciph_sum = evaluator.multiply(ciph_sum, ciph_sum, relin_key)

        ciph2 = ciph_sum

        message3 = w
        one_vector = [1] * Nslots
        one_plain = encoder.encode(one_vector, scaling_factor)
        ciph_one = encryptor.encrypt(one_plain)

        message3.extend([0] * (Nslots - len(message3)))
        plain3 = encoder.encode(message3, scaling_factor)
        ciph3 = encryptor.encrypt(plain3)

        ciph3 = evaluator.multiply(ciph3, ciph_one, relin_key)

        i_ciphers = []
        client_ciphertexts = []
        for n, row in enumerate(A):
            i_n = [1 if i == n else 0 for i in range(len(A))]  # indicator vector
            i_n.extend([0] * (Nslots - len(i_n)))
            i_n_plain = encoder.encode(i_n, scaling_factor)
            i_n_cipher = encryptor.encrypt(i_n_plain)
            i_ciphers.append(i_n_cipher)

            message1 = row
            message1.extend([0] * (Nslots - len(message1)))
            plain1 = encoder.encode(message1, scaling_factor)
            ciph1 = encryptor.encrypt(plain1)
            client_ciphertexts.append(ciph1)
    

        # send away to the server the following: client_ciphertexts, i_ciphers, ciph2, ciph3, i_rot_keys, b_rot_keys, conj_key, relin_key

        #-------------------------------- Server Side ----------------------------------------

        # receives the following from the client: client_ciphertexts, ciph2, ciph3
        ciph_sum = None  # Initialize sum
        for k, ciph1 in enumerate(client_ciphertexts):

            ciph_inner_prod = compute_inner_product(ciph1, ciph2, Nslots, encoder, \
                                    scaling_factor, evaluator, ip_rot_keys, boot_rot_keys, conj_key, relin_key)

            ciph_inner_prod = evaluator.multiply(ciph_inner_prod, i_ciphers[k], relin_key)

            if ciph_sum is None:
                ciph_sum = ciph_inner_prod
            else:
                ciph_sum = evaluator.add(ciph_sum, ciph_inner_prod)

        ciph_sum = evaluator.add(ciph_sum, ciph3) # adding constant
        ciph_sum = evaluator.rescale(ciph_sum, scaling_factor)

        # return `ciph_sum` back to the client for decryption

        #-------------------------------- Client Side: Decryption (Intermediate; can be commented out) ----------------------------------------
        # ciph_sum_dec = decryptor.decrypt(ciph_sum)
        # ciph_sum_decoded = encoder.decode(ciph_sum_dec)

        # print(ciph_sum_decoded)

    else:

        ciph_sum = to_bootstrap

        # bootstrapping - can be perofrmed by the server
        ciph_boot = ckks_bootstrap(ciph_sum, evaluator, boot_rot_keys, conj_key, relin_key, encoder)

        #-------------------------------- Client Side ----------------------------------------
        ciph2 = ciph_boot # for 2, 3, 4, ... iterations

        message3 = w
        one_vector = [1] * Nslots
        one_plain = encoder.encode(one_vector, scaling_factor)
        ciph_one = encryptor.encrypt(one_plain)

        message3.extend([0] * (Nslots - len(message3)))
        plain3 = encoder.encode(message3, scaling_factor)
        ciph3 = encryptor.encrypt(plain3)

        ciph3 = evaluator.multiply(ciph3, ciph_one, relin_key)

        i_ciphers = []
        client_ciphertexts = []
        for n, row in enumerate(A):
            i_n = [1 if i == n else 0 for i in range(len(A))]  # indicator vector
            i_n.extend([0] * (Nslots - len(i_n)))
            i_n_plain = encoder.encode(i_n, scaling_factor)
            i_n_cipher = encryptor.encrypt(i_n_plain)
            i_ciphers.append(i_n_cipher)

            message1 = row
            message1.extend([0] * (Nslots - len(message1)))
            plain1 = encoder.encode(message1, scaling_factor)
            ciph1 = encryptor.encrypt(plain1)
            client_ciphertexts.append(ciph1)
            
        # send away to the server the following: client_ciphertexts, i_ciphers, ciph2, ciph3, i_rot_keys, b_rot_keys, conj_key, relin_key

        #-------------------------------- Server Side ----------------------------------------

        # receives the following from the client: client_ciphertexts, ciph2, ciph3
        ciph_sum = None  # Initialize sum
        for k, ciph1 in enumerate(client_ciphertexts):  # Loop through received ciphertexts
                
            ciph_inner_prod = compute_inner_product(ciph1, ciph2, Nslots, encoder, \
                                    scaling_factor, evaluator, ip_rot_keys, boot_rot_keys, conj_key, relin_key)

            ciph_inner_prod = evaluator.multiply(ciph_inner_prod, i_ciphers[k], relin_key)

            if ciph_sum is None:
                ciph_sum = ciph_inner_prod
            else:
                ciph_sum = evaluator.add(ciph_sum, ciph_inner_prod)
        ciph_sum = evaluator.add(ciph_sum, ciph3) # adding constant
        ciph_sum = evaluator.rescale(ciph_sum, scaling_factor)

        # return `ciph_sum` back to the client for decryption

    # # -------------------------------- Client Side: Decryption (Intermediate; can be commented out) ----------------------------------------
    ciph_sum_dec = decryptor.decrypt(ciph_sum)
    ciph_sum_decoded = encoder.decode(ciph_sum_dec)

    ciph_sum_decoded_real = [c.real for c in ciph_sum_decoded]

    return ciph_sum, ciph_sum_decoded_real


for experiment_num in range(len(params_set_meta)):

    print(f"Experiment Number: {experiment_num}")

    params = CKKSParameters(
        poly_degree=params_set_meta[experiment_num]['poly_degree'],
        ciph_modulus=params_set_meta[experiment_num]['ciph_modulus'],
        big_modulus=params_set_meta[experiment_num]['big_modulus'],
        scaling_factor=params_set_meta[experiment_num]['scaling_factor']
    )

    # Generate keys and setup
    key_generator = CKKSKeyGenerator(params)
    public_key = key_generator.public_key
    secret_key = key_generator.secret_key
    relin_key = key_generator.relin_key
    encoder = CKKSEncoder(params)
    encryptor = CKKSEncryptor(params, public_key, secret_key)
    decryptor = CKKSDecryptor(params, secret_key)
    evaluator = CKKSEvaluator(params)

    enc_errors_list = []

    enc_errors = []

    ciph_sum, ciph_sum_decoded_real = simulate_encrypted_linear_VI(key_generator, encoder, encryptor, evaluator, relin_key, decryptor, first_iter=True, to_bootstrap=None)

    enc_error = np.mean(np.abs(ciph_sum_decoded_real - z_final)) / np.mean(z_star)  # Approx error
    enc_errors.append(enc_error)

    print(f"Iteration 1, z_encdec = {ciph_sum_decoded_real}, Error = {enc_error}.")


    for t in tqdm(range(1, (T + 1 + 5)), "Iterating over FHE..."):

        ciph_sum, ciph_sum_decoded_real = simulate_encrypted_linear_VI(key_generator, encoder, encryptor, evaluator, relin_key, decryptor, first_iter=False, to_bootstrap=ciph_sum)

        enc_error = np.mean(np.abs(ciph_sum_decoded_real - z_final)) / np.mean(z_star)  # Approx error
        enc_errors.append(enc_error)

        
        print(f"Iteration {t+1}, z_encdec = {ciph_sum_decoded_real}, Error = {enc_error}.")

        # check termination
        if enc_error < 1e-6:
            print(f"Converged at iteration {t} with (encrypted) error {enc_error} < {1e-6}.")
            break

    enc_errors_list.append(enc_errors)

plt.figure(figsize=(8, 6))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'text.latex.preamble': r'\usepackage{amssymb}',
    'axes.grid': True,
    'grid.linestyle': ':',
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 28,
})

plt.plot(errors, 'o--', label=r'$\frac{\mathbb{E}_{x}|Z - Z^*|}{\mathbb{E}_{x}[Z^*]}$')
plt.plot(enc_errors[:20], 'o-', fillstyle='none', label=r'$\frac{\mathbb{E}_{x}|\tilde{Z} - Z^*|}{\mathbb{E}_{x}[Z^*]}$')
plt.xlim([6, 20])
plt.ylim([-0.000001, 0.0001])
plt.legend()
# plt.title(r'$\text{Convergence of Approx Error}$')
plt.xlabel('Iterations')
plt.ylabel('Approximation Errors')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('approx_error_plot.png', dpi=300, bbox_inches='tight')
