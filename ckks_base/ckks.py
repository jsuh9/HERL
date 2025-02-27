import math
from math import sqrt
from util.ciphertext import Ciphertext
from util.crt import CRTContext
import util.matrix_operations
from util.plaintext import Plaintext
from util.polynomial import Polynomial
from util.ntt import FFTContext
from util.public_key import PublicKey
from util.rotation_key import RotationKey
from util.secret_key import SecretKey
from util.random_sample import sample_triangle, sample_uniform, sample_hamming_weight_vector

class CKKSBootstrappingContext:

    """An object that stores information necessary for bootstrapping.

    Attributes:
        poly_degree: Polynomial degree of ring.
        old_modulus: Original modulus of initial ciphertext.
        num_taylor_iterations: Number of iterations to perform for Taylor series
            for exp.
        encoding_mat0: Matrix for slot to coeff.
        encoding_mat1: Matrix for slot to coeff.
        encoding_mat_transpose0: Matrix for coeff to slot.
        encoding_mat_transpose1: Matrix for coeff to slot.
        encoding_mat_conj_transpose0: Matrix for coeff to slot.
        encoding_mat_conj_transpose1: Matrix for coeff to slot.
    """

    def __init__(self, params):
        """Generates private/public key pair for CKKS scheme.

        Args:
            params (CKKSParameters): Parameters including polynomial degree,
                ciphertext modulus, etc.
        """
        self.poly_degree = params.poly_degree
        self.old_modulus = params.ciph_modulus
        self.num_taylor_iterations = params.num_taylor_iterations
        self.generate_encoding_matrices()

    def get_primitive_root(self, index):
        """Returns the ith out of the n roots of unity, where n is 2 * poly_degree.

        Args:
            index (int): Index i to specify.

        Returns:
            The ith out of nth root of unity.
        """
        angle = math.pi * index / self.poly_degree
        return complex(math.cos(angle), math.sin(angle))

    def generate_encoding_matrices(self):
        """Generates encoding matrices for coeff_to_slot and slot_to_coeff operations.
        """
        num_slots = self.poly_degree // 2
        primitive_roots = [0] * num_slots
        power = 1
        for i in range(num_slots):
            primitive_roots[i] = self.get_primitive_root(power)
            power = (power * 5) % (2 * self.poly_degree)

        # Compute matrices for slot to coeff transformation.
        self.encoding_mat0 = [[1] * num_slots for _ in range(num_slots)]
        self.encoding_mat1 = [[1] * num_slots for _ in range(num_slots)]

        for i in range(num_slots):
            for k in range(1, num_slots):
                self.encoding_mat0[i][k] = self.encoding_mat0[i][k - 1] * primitive_roots[i]

        for i in range(num_slots):
            self.encoding_mat1[i][0] = self.encoding_mat0[i][-1] * primitive_roots[i]

        for i in range(num_slots):
            for k in range(1, num_slots):
                self.encoding_mat1[i][k] = self.encoding_mat1[i][k - 1] * primitive_roots[i]

        # Compute matrices for coeff to slot transformation.
        self.encoding_mat_transpose0 = util.matrix_operations.transpose_matrix(self.encoding_mat0)
        self.encoding_mat_conj_transpose0 = util.matrix_operations.conjugate_matrix(
            self.encoding_mat_transpose0)
        self.encoding_mat_transpose1 = util.matrix_operations.transpose_matrix(self.encoding_mat1)
        self.encoding_mat_conj_transpose1 = util.matrix_operations.conjugate_matrix(
            self.encoding_mat_transpose1)



class CKKSDecryptor:

    """An object that can decrypt data using CKKS given a secret key.

    Attributes:
        poly_degree: Degree of polynomial in quotient ring.
        crt_context: CRT context for multiplication.
        secret_key (SecretKey): Secret key used for encryption.
    """

    def __init__(self, params, secret_key):
        """Initializes decryptor for CKKS scheme.

        Args:
            params (Parameters): Parameters including polynomial degree,
                plaintext modulus, and ciphertext modulus.
            secret_key (SecretKey): Secret key used for decryption.
        """
        self.poly_degree = params.poly_degree
        self.crt_context = params.crt_context
        self.secret_key = secret_key

    def decrypt(self, ciphertext, c2=None):
        """Decrypts a ciphertext.

        Decrypts the ciphertext and returns the corresponding plaintext.

        Args:
            ciphertext (Ciphertext): Ciphertext to be decrypted.
            c2 (Polynomial): Optional additional parameter for a ciphertext that
                has not been relinearized.

        Returns:
            The plaintext corresponding to the decrypted ciphertext.
        """
        (c0, c1) = (ciphertext.c0, ciphertext.c1)

        message = c1.multiply(self.secret_key.s, ciphertext.modulus, crt=self.crt_context)
        message = c0.add(message, ciphertext.modulus)
        if c2:
            secret_key_squared = self.secret_key.s.multiply(self.secret_key.s, ciphertext.modulus)
            c2_message = c2.multiply(secret_key_squared, ciphertext.modulus, crt=self.crt_context)
            message = message.add(c2_message, ciphertext.modulus)

        message = message.mod_small(ciphertext.modulus)
        return Plaintext(message, ciphertext.scaling_factor)
        

class CKKSEncoder:
    """An encoder for several complex numbers as specified in the CKKS scheme.

    Attributes:
        degree (int): Degree of polynomial that determines quotient ring.
        fft (FFTContext): FFTContext object to encode/decode.
    """

    def __init__(self, params):
        """Inits CKKSEncoder with the given parameters.

        Args:
            params (Parameters): Parameters including polynomial degree,
                plaintext modulus, and ciphertext modulus.
        """
        self.degree = params.poly_degree
        self.fft = FFTContext(self.degree * 2)

    def encode(self, values, scaling_factor):
        """Encodes complex numbers into a polynomial.

        Encodes an array of complex number into a polynomial.

        Args:
            values (list): List of complex numbers to encode.
            scaling_factor (float): Scaling factor to multiply by.

        Returns:
            A Plaintext object which represents the encoded value.
        """
        num_values = len(values)
        plain_len = num_values << 1

        # Canonical embedding inverse variant.
        to_scale = self.fft.embedding_inv(values)

        # Multiply by scaling factor, and split up real and imaginary parts.
        message = [0] * plain_len
        for i in range(num_values):
            message[i] = int(to_scale[i].real * scaling_factor + 0.5)
            message[i + num_values] = int(to_scale[i].imag * scaling_factor + 0.5)

        return Plaintext(Polynomial(plain_len, message), scaling_factor)


    def decode(self, plain):
        """Decodes a plaintext polynomial.

        Decodes a plaintext polynomial back to a list of integers.

        Args:
            plain (Plaintext): Plaintext to decode.

        Returns:
            A decoded list of integers.
        """
        if not isinstance(plain, Plaintext):
            raise ValueError("Input to decode must be a Plaintext")

        plain_len = len(plain.poly.coeffs)
        num_values = plain_len >> 1

        # Divide by scaling factor, and turn back into a complex number.
        message = [0] * num_values
        for i in range(num_values):
            message[i] = complex(plain.poly.coeffs[i] / plain.scaling_factor,
                                 plain.poly.coeffs[i + num_values] / plain.scaling_factor)

        # Compute canonical embedding variant.
        return self.fft.embedding(message)


class CKKSEncryptor:

    """An object that can encrypt data using CKKS given a public key.

    Attributes:
        poly_degree: Degree of polynomial in quotient ring.
        coeff_modulus: Coefficient modulus in ciphertext space.
        big_modulus: Bootstrapping modulus.
        crt_context: CRT context for multiplication.
        public_key (PublicKey): Public key used for encryption.
        secret_key (SecretKey): Only used for secret key encryption.
    """

    def __init__(self, params, public_key, secret_key=None):
        """Generates private/public key pair for CKKS scheme.

        Args:
            params (Parameters): Parameters including polynomial degree,
                ciphertext modulus, etc.
            public_key (PublicKey): Public key used for encryption.
            secret_key (SecretKey): Optionally passed for secret key encryption.
        """
        self.poly_degree = params.poly_degree
        self.coeff_modulus = params.ciph_modulus
        self.big_modulus = params.big_modulus
        self.crt_context = params.crt_context
        self.public_key = public_key
        self.secret_key = secret_key

    def encrypt_with_secret_key(self, plain):
        """Encrypts a message with secret key encryption.

        Encrypts the message for secret key encryption and returns the corresponding ciphertext.

        Args:
            plain (Plaintext): Plaintext to be encrypted.

        Returns:
            A ciphertext consisting of a pair of polynomials in the ciphertext
            space.
        """
        assert self.secret_key != None, 'Secret key does not exist'

        sk = self.secret_key.s
        random_vec = Polynomial(self.poly_degree, sample_triangle(self.poly_degree))
        error = Polynomial(self.poly_degree, sample_triangle(self.poly_degree))

        c0 = sk.multiply(random_vec, self.coeff_modulus, crt=self.crt_context)
        c0 = error.add(c0, self.coeff_modulus)
        c0 = c0.add(plain.poly, self.coeff_modulus)
        c0 = c0.mod_small(self.coeff_modulus)

        c1 = random_vec.scalar_multiply(-1, self.coeff_modulus)
        c1 = c1.mod_small(self.coeff_modulus)

        return Ciphertext(c0, c1, plain.scaling_factor, self.coeff_modulus)

    def encrypt(self, plain):
        """Encrypts a message.

        Encrypts the message and returns the corresponding ciphertext.

        Args:
            plain (Plaintext): Plaintext to be encrypted.

        Returns:
            A ciphertext consisting of a pair of polynomials in the ciphertext
            space.
        """
        p0 = self.public_key.p0
        p1 = self.public_key.p1
        
        random_vec = Polynomial(self.poly_degree, sample_triangle(self.poly_degree))
        error1 = Polynomial(self.poly_degree, sample_triangle(self.poly_degree))
        error2 = Polynomial(self.poly_degree, sample_triangle(self.poly_degree))

        c0 = p0.multiply(random_vec, self.coeff_modulus, crt=self.crt_context)
        c0 = error1.add(c0, self.coeff_modulus)
        c0 = c0.add(plain.poly, self.coeff_modulus)
        c0 = c0.mod_small(self.coeff_modulus)

        c1 = p1.multiply(random_vec, self.coeff_modulus, crt=self.crt_context)
        c1 = error2.add(c1, self.coeff_modulus)
        c1 = c1.mod_small(self.coeff_modulus)

        return Ciphertext(c0, c1, plain.scaling_factor, self.coeff_modulus)

    def raise_modulus(self, new_modulus):
        """Rescales scheme to have a new modulus.

        Raises ciphertext modulus.

        Args:
            new_modulus (int): New modulus.
        """
        self.coeff_modulus = new_modulus
        


class CKKSEvaluator:

    """An instance of an evaluator for ciphertexts.

    This allows us to add, multiply, and relinearize ciphertexts.

    Attributes:
        degree (int): Polynomial degree of ring.
        big_modulus (int): Modulus q of coefficients of polynomial
            ring R_q.
        scaling_factor (float): Scaling factor to encode new plaintexts with.
        boot_context (CKKSBootstrappingContext): Bootstrapping pre-computations.
        crt_context (CRTContext): CRT functions.
    """

    def __init__(self, params):
        """Inits Evaluator.

        Args:
            params (Parameters): Parameters including polynomial degree, ciphertext modulus,
                and scaling factor.
        """
        self.degree = params.poly_degree
        self.big_modulus = params.big_modulus
        self.ciph_modulus = params.ciph_modulus  # Add this line
        self.scaling_factor = params.scaling_factor
        self.boot_context = CKKSBootstrappingContext(params)
        self.crt_context = params.crt_context

    def add(self, ciph1, ciph2):
        """Adds two ciphertexts.

        Adds two ciphertexts within the context.

        Args:
            ciph1 (Ciphertext): First ciphertext.
            ciph2 (Ciphertext): Second ciphertext.

        Returns:
            A Ciphertext which is the sum of the two ciphertexts.
        """
        assert isinstance(ciph1, Ciphertext)
        assert isinstance(ciph2, Ciphertext)
        assert ciph1.scaling_factor == ciph2.scaling_factor, "Scaling factors are not equal. " \
            + "Ciphertext 1 scaling factor: %d bits, Ciphertext 2 scaling factor: %d bits" \
            % (math.log(ciph1.scaling_factor, 2), math.log(ciph2.scaling_factor, 2))
        assert ciph1.modulus == ciph2.modulus, "Moduli are not equal. " \
            + "Ciphertext 1 modulus: %d bits, Ciphertext 2 modulus: %d bits" \
            % (math.log(ciph1.modulus, 2), math.log(ciph2.modulus, 2))

        modulus = ciph1.modulus

        c0 = ciph1.c0.add(ciph2.c0, modulus)
        c0 = c0.mod_small(modulus)
        c1 = ciph1.c1.add(ciph2.c1, modulus)
        c1 = c1.mod_small(modulus)
        return Ciphertext(c0, c1, ciph1.scaling_factor, modulus)

    def add_plain(self, ciph, plain):
        """Adds a ciphertext with a plaintext.

        Adds a ciphertext with a plaintext polynomial within the context.

        Args:
            ciph (Ciphertext): A ciphertext to add.
            plain (Plaintext): A plaintext to add.

        Returns:
            A Ciphertext which is the sum of the ciphertext and plaintext.
        """
        assert isinstance(ciph, Ciphertext)
        assert isinstance(plain, Plaintext)
        assert ciph.scaling_factor == plain.scaling_factor, "Scaling factors are not equal. " \
            + "Ciphertext scaling factor: %d bits, Plaintext scaling factor: %d bits" \
            % (math.log(ciph.scaling_factor, 2), math.log(plain.scaling_factor, 2))

        c0 = ciph.c0.add(plain.poly, ciph.modulus)
        c0 = c0.mod_small(ciph.modulus)

        return Ciphertext(c0, ciph.c1, ciph.scaling_factor, ciph.modulus)

    def subtract(self, ciph1, ciph2):
        """Subtracts second ciphertext from first ciphertext.

        Computes ciph1 - ciph2.

        Args:
            ciph1 (Ciphertext): First ciphertext.
            ciph2 (Ciphertext): Second ciphertext.

        Returns:
            A Ciphertext which is the difference between the two ciphertexts.
        """
        assert isinstance(ciph1, Ciphertext)
        assert isinstance(ciph2, Ciphertext)
        assert ciph1.scaling_factor == ciph2.scaling_factor, "Scaling factors are not equal. " \
            + "Ciphertext 1 scaling factor: %d bits, Ciphertext 2 scaling factor: %d bits" \
            % (math.log(ciph1.scaling_factor, 2), math.log(ciph2.scaling_factor, 2))
        assert ciph1.modulus == ciph2.modulus, "Moduli are not equal. " \
            + "Ciphertext 1 modulus: %d bits, Ciphertext 2 modulus: %d bits" \
            % (math.log(ciph1.modulus, 2), math.log(ciph2.modulus, 2))

        modulus = ciph1.modulus

        c0 = ciph1.c0.subtract(ciph2.c0, modulus)
        c0 = c0.mod_small(modulus)
        c1 = ciph1.c1.subtract(ciph2.c1, modulus)
        c1 = c1.mod_small(modulus)
        return Ciphertext(c0, c1, ciph1.scaling_factor, modulus)

    def multiply(self, ciph1, ciph2, relin_key):
        """Multiplies two ciphertexts.

        Multiplies two ciphertexts within the context, and relinearizes.

        Args:
            ciph1 (Ciphertext): First ciphertext.
            ciph2 (Ciphertext): Second ciphertext.
            relin_key (PublicKey): Relinearization keys.

        Returns:
            A Ciphertext which is the product of the two ciphertexts.
        """
        assert isinstance(ciph1, Ciphertext)
        assert isinstance(ciph2, Ciphertext)
        assert ciph1.modulus == ciph2.modulus, "Moduli are not equal. " \
            + "Ciphertext 1 modulus: %d bits, Ciphertext 2 modulus: %d bits" \
            % (math.log(ciph1.modulus, 2), math.log(ciph2.modulus, 2))

        modulus = ciph1.modulus

        c0 = ciph1.c0.multiply(ciph2.c0, modulus, crt=self.crt_context)
        c0 = c0.mod_small(modulus)

        c1 = ciph1.c0.multiply(ciph2.c1, modulus, crt=self.crt_context)
        temp = ciph1.c1.multiply(ciph2.c0, modulus, crt=self.crt_context)
        c1 = c1.add(temp, modulus)
        c1 = c1.mod_small(modulus)

        c2 = ciph1.c1.multiply(ciph2.c1, modulus, crt=self.crt_context)
        c2 = c2.mod_small(modulus)

        return self.relinearize(relin_key, c0, c1, c2, ciph1.scaling_factor * ciph2.scaling_factor,
                                modulus)

    def multiply_plain(self, ciph, plain):
        """Multiplies a ciphertext with a plaintext.

        Multiplies a ciphertext with a plaintext polynomial within the context.

        Args:
            ciph (Ciphertext): A ciphertext to multiply.
            plain (Plaintext): A plaintext to multiply.

        Returns:
            A Ciphertext which is the product of the ciphertext and plaintext.
        """
        assert isinstance(ciph, Ciphertext)
        assert isinstance(plain, Plaintext)

        c0 = ciph.c0.multiply(plain.poly, ciph.modulus, crt=self.crt_context)
        c0 = c0.mod_small(ciph.modulus)

        c1 = ciph.c1.multiply(plain.poly, ciph.modulus, crt=self.crt_context)
        c1 = c1.mod_small(ciph.modulus)

        return Ciphertext(c0, c1, ciph.scaling_factor * plain.scaling_factor, ciph.modulus)

    def relinearize(self, relin_key, c0, c1, c2, new_scaling_factor, modulus):
        """Relinearizes a 3-dimensional ciphertext.

        Reduces 3-dimensional ciphertext back down to 2 dimensions.

        Args:
            relin_key (PublicKey): Relinearization keys.
            c0 (Polynomial): First component of ciphertext.
            c1 (Polynomial): Second component of ciphertext.
            c2 (Polynomial): Third component of ciphertext.
            new_scaling_factor (float): New scaling factor for ciphertext.
            modulus (int): Ciphertext modulus.

        Returns:
            A Ciphertext which has only two components.
        """
        new_c0 = relin_key.p0.multiply(c2, modulus * self.big_modulus, crt=self.crt_context)
        new_c0 = new_c0.mod_small(modulus * self.big_modulus)
        new_c0 = new_c0.scalar_integer_divide(self.big_modulus)
        new_c0 = new_c0.add(c0, modulus)
        new_c0 = new_c0.mod_small(modulus)

        new_c1 = relin_key.p1.multiply(c2, modulus * self.big_modulus, crt=self.crt_context)
        new_c1 = new_c1.mod_small(modulus * self.big_modulus)
        new_c1 = new_c1.scalar_integer_divide(self.big_modulus)
        new_c1 = new_c1.add(c1, modulus)
        new_c1 = new_c1.mod_small(modulus)

        return Ciphertext(new_c0, new_c1, new_scaling_factor, modulus)

    def rescale(self, ciph, division_factor):
        """Rescales a ciphertext to a new scaling factor.

        Divides ciphertext by division factor, and updates scaling factor
        and ciphertext. modulus.

        Args:
            ciph (Ciphertext): Ciphertext to modify.
            division_factor (float): Factor to divide by.

        Returns:
            Rescaled ciphertext.
        """
        c0 = ciph.c0.scalar_integer_divide(division_factor)
        c1 = ciph.c1.scalar_integer_divide(division_factor)
        return Ciphertext(c0, c1, ciph.scaling_factor // division_factor,
                          ciph.modulus // division_factor)

    def lower_modulus(self, ciph, division_factor):
        """Rescales a ciphertext to a new scaling factor.

        Divides ciphertext by division factor, and updates scaling factor
        and ciphertext modulus.

        Args:
            ciph (Ciphertext): Ciphertext to modify.
            division_factor (float): Factor to divide by.

        Returns:
            Rescaled ciphertext.
        """
        new_modulus = ciph.modulus // division_factor
        c0 = ciph.c0.mod_small(new_modulus)
        c1 = ciph.c1.mod_small(new_modulus)
        return Ciphertext(c0, c1, ciph.scaling_factor, new_modulus)

    def force_modulus_reduction(self, ciph, target_modulus):
        """Forces reduction of ciphertext modulus to target size."""
        if ciph.modulus <= target_modulus:
            return ciph
            
        reduction_factor = ciph.modulus // target_modulus
        return self.lower_modulus(ciph, reduction_factor)

    def switch_key(self, ciph, key):
        """Outputs ciphertext with switching key.

        Performs KS procedure as described in CKKS paper.

        Args:
            ciph (Ciphertext): Ciphertext to change.
            switching_key (PublicKey): Switching key.

        Returns:
            A Ciphertext which encrypts the same message under a different key.
        """

        c0 = key.p0.multiply(ciph.c1, ciph.modulus * self.big_modulus, crt=self.crt_context)
        c0 = c0.mod_small(ciph.modulus * self.big_modulus)
        c0 = c0.scalar_integer_divide(self.big_modulus)
        c0 = c0.add(ciph.c0, ciph.modulus)
        c0 = c0.mod_small(ciph.modulus)

        c1 = key.p1.multiply(ciph.c1, ciph.modulus * self.big_modulus, crt=self.crt_context)
        c1 = c1.mod_small(ciph.modulus * self.big_modulus)
        c1 = c1.scalar_integer_divide(self.big_modulus)
        c1 = c1.mod_small(ciph.modulus)

        return Ciphertext(c0, c1, ciph.scaling_factor, ciph.modulus)

    def rotate(self, ciph, rotation, rot_key):
        """Rotates a ciphertext by the amount specified in rotation.

        Returns a ciphertext for a plaintext which is rotated by the amount
        in rotation.

        Args:
            ciph (Ciphertext): Ciphertext to rotate.
            rotation (int): Amount to rotate by.
            rot_key (RotationKey): Rotation key corresponding to the rotation.

        Returns:
            A Ciphertext which is the encryption of the rotation of the original
            plaintext.
        """
        rot_ciph0 = ciph.c0.rotate(rotation)
        rot_ciph1 = ciph.c1.rotate(rotation)
        rot_ciph = Ciphertext(rot_ciph0, rot_ciph1, ciph.scaling_factor, ciph.modulus)
        return self.switch_key(rot_ciph, rot_key.key)

    def conjugate(self, ciph, conj_key):
        """Conjugates the ciphertext.

        Returns a ciphertext for a plaintext which is conjugated.

        Args:
            ciph (Ciphertext): Ciphertext to conjugate.
            conj_key (PublicKey): Conjugation key.

        Returns:
            A Ciphertext which is the encryption of the conjugation of the original
            plaintext.
        """

        conj_ciph0 = ciph.c0.conjugate().mod_small(ciph.modulus)
        conj_ciph1 = ciph.c1.conjugate().mod_small(ciph.modulus)
        conj_ciph = Ciphertext(conj_ciph0, conj_ciph1, ciph.scaling_factor, ciph.modulus)
        return self.switch_key(conj_ciph, conj_key)

    def multiply_matrix_naive(self, ciph, matrix, rot_keys, encoder):
        """Multiplies the ciphertext by the given matrix.

        Returns a ciphertext for the matrix multiplication.

        Args:
            ciph (Ciphertext): Ciphertext to multiply.
            matrix (2-D Array): Matrix to multiply.
            rot_keys (dict (RotationKey)): Rotation keys
            encoder (CKKSEncoder): Encoder for CKKS.

        Returns:
            A Ciphertext which is the product of matrix and ciph.
        """
        diag = util.matrix_operations.diagonal(matrix, 0)
        diag = encoder.encode(diag, self.scaling_factor)
        ciph_prod = self.multiply_plain(ciph, diag)

        for j in range(1, len(matrix)):
            diag = util.matrix_operations.diagonal(matrix, j)
            diag = encoder.encode(diag, self.scaling_factor)
            rot = self.rotate(ciph, j, rot_keys[j])
            ciph_temp = self.multiply_plain(rot, diag)
            ciph_prod = self.add(ciph_prod, ciph_temp)

        return ciph_prod

    def multiply_matrix(self, ciph, matrix, rot_keys, encoder):
        """Multiplies the ciphertext by the given matrix quickly.

        Returns a ciphertext for the matrix multiplication using the Baby-Step Giant-Step algorithm
        described in the CKKS paper.

        Args:
            ciph (Ciphertext): Ciphertext to multiply.
            matrix (2-D Array): Matrix to multiply.
            rot_keys (dict (RotationKey)): Rotation keys
            encoder (CKKSEncoder): Encoder for CKKS.

        Returns:
            A Ciphertext which is the product of matrix and ciph.
        """

        # Compute two factors of matrix_len (a power of two), both near its square root.
        matrix_len = len(matrix)
        matrix_len_factor1 = int(sqrt(matrix_len))
        if matrix_len != matrix_len_factor1 * matrix_len_factor1:
            matrix_len_factor1 = int(sqrt(2 * matrix_len))
        matrix_len_factor2 = matrix_len // matrix_len_factor1

        # Compute rotations.
        ciph_rots = [0] * matrix_len_factor1
        ciph_rots[0] = ciph
        for i in range(1, matrix_len_factor1):
            ciph_rots[i] = self.rotate(ciph, i, rot_keys[i])

        # Compute sum.
        outer_sum = None
        for j in range(matrix_len_factor2):
            inner_sum = None
            shift = matrix_len_factor1 * j
            for i in range(matrix_len_factor1):
                diagonal = util.matrix_operations.diagonal(matrix, shift + i)
                diagonal = util.matrix_operations.rotate(diagonal, -shift)
                diagonal_plain = encoder.encode(diagonal, self.scaling_factor)
                dot_prod = self.multiply_plain(ciph_rots[i], diagonal_plain)
                if inner_sum:
                    inner_sum = self.add(inner_sum, dot_prod)
                else:
                    inner_sum = dot_prod

            rotated_sum = self.rotate(inner_sum, shift, rot_keys[shift])
            if outer_sum:
                outer_sum = self.add(outer_sum, rotated_sum)
            else:
                outer_sum = rotated_sum

        outer_sum = self.rescale(outer_sum, self.scaling_factor)
        return outer_sum

    # BOOTSTRAPPING

    def create_constant_plain(self, const):
        """Creates a plaintext containing a constant value.

        Takes a floating-point constant, and turns it into a plaintext.

        Args:
            const (float): Constant to encode.

        Returns:
            Plaintext with constant value.
        """
        plain_vec = [0] * (self.degree)
        plain_vec[0] = int(const * self.scaling_factor)
        return Plaintext(Polynomial(self.degree, plain_vec), self.scaling_factor)

    def create_complex_constant_plain(self, const, encoder):
        """Creates a plaintext containing a constant value.

        Takes any constant, and turns it into a plaintext.

        Args:
            const (float): Constant to encode.
            encoder (CKKSEncoder): Encoder.

        Returns:
            Plaintext with constant value.
        """
        plain_vec = [const] * (self.degree // 2)
        return encoder.encode(plain_vec, self.scaling_factor)

    def coeff_to_slot(self, ciph, rot_keys, conj_key, encoder):
        """Takes a ciphertext coefficients and puts into plaintext slots.

        Takes an encryption of t(x) = t_0 + t_1x + ... and transforms to
        encryptions of (t_0, t_1, ..., t_(n/2)) and (t_(n/2 + 1), ..., t_(n-1))
        before these vectors are encoded.

        Args:
            ciph (Ciphertext): Ciphertext to transform.
            rot_keys (dict (RotationKey)): Rotation keys
            conj_key (PublicKey): Conjugation key.
            encoder (CKKSEncoder): Encoder for CKKS.

        Returns:
            Two Ciphertexts which are transformed.
        """
        # Compute new ciphertexts.
        s1 = self.multiply_matrix(ciph, self.boot_context.encoding_mat_conj_transpose0,
                                  rot_keys, encoder)
        s2 = self.conjugate(ciph, conj_key)
        s2 = self.multiply_matrix(s2, self.boot_context.encoding_mat_transpose0, rot_keys,
                                  encoder)
        ciph0 = self.add(s1, s2)
        constant = self.create_constant_plain(1 / self.degree)
        ciph0 = self.multiply_plain(ciph0, constant)
        ciph0 = self.rescale(ciph0, self.scaling_factor)

        s1 = self.multiply_matrix(ciph, self.boot_context.encoding_mat_conj_transpose1,
                                  rot_keys, encoder)
        s2 = self.conjugate(ciph, conj_key)
        s2 = self.multiply_matrix(s2, self.boot_context.encoding_mat_transpose1, rot_keys,
                                  encoder)
        ciph1 = self.add(s1, s2)
        ciph1 = self.multiply_plain(ciph1, constant)
        ciph1 = self.rescale(ciph1, self.scaling_factor)

        return ciph0, ciph1

    def slot_to_coeff(self, ciph0, ciph1, rot_keys, encoder):
        """Takes plaintext slots and puts into ciphertext coefficients.

        Takes encryptions of (t_0, t_1, ..., t_(n/2)) and (t_(n/2 + 1), ..., t_(n-1))
        before these vectors are encoded and transofmrs to an encryption of
        t(x) = t_0 + t_1x + ...

        Args:
            ciph0 (Ciphertext): First ciphertext to transform.
            ciph1 (Ciphertext): Second ciphertext to transform.
            rot_keys (dict (RotationKey)): Rotation keys.
            encoder (CKKSEncoder): Encoder for CKKS.

        Returns:
            Ciphertext which is transformed.
        """
        s1 = self.multiply_matrix(ciph0, self.boot_context.encoding_mat0, rot_keys,
                                  encoder)
        s2 = self.multiply_matrix(ciph1, self.boot_context.encoding_mat1, rot_keys,
                                  encoder)
        ciph = self.add(s1, s2)

        return ciph

    def exp_taylor(self, ciph, relin_key, encoder):
        """Evaluates the exponential function on the ciphertext.

        Takes an encryption of m and returns an encryption of e^(2 * pi * m).

        Args:
            ciph (Ciphertext): Ciphertext to transform.
            relin_key (PublicKey): Relinearization key.
            encoder (CKKSEncoder): Encoder.

        Returns:
            Ciphertext for exponential.
        """
        ciph2 = self.multiply(ciph, ciph, relin_key)
        ciph2 = self.rescale(ciph2, self.scaling_factor)

        ciph4 = self.multiply(ciph2, ciph2, relin_key)
        ciph4 = self.rescale(ciph4, self.scaling_factor)

        const = self.create_constant_plain(1)
        ciph01 = self.add_plain(ciph, const)

        const = self.create_constant_plain(1)
        ciph01 = self.multiply_plain(ciph01, const)
        ciph01 = self.rescale(ciph01, self.scaling_factor)

        const = self.create_constant_plain(3)
        ciph23 = self.add_plain(ciph, const)

        const = self.create_constant_plain(1 / 6)
        ciph23 = self.multiply_plain(ciph23, const)
        ciph23 = self.rescale(ciph23, self.scaling_factor)

        ciph23 = self.multiply(ciph23, ciph2, relin_key)
        ciph23 = self.rescale(ciph23, self.scaling_factor)
        ciph01 = self.lower_modulus(ciph01, self.scaling_factor)
        ciph23 = self.add(ciph23, ciph01)

        const = self.create_constant_plain(5)
        ciph45 = self.add_plain(ciph, const)

        const = self.create_constant_plain(1 / 120)
        ciph45 = self.multiply_plain(ciph45, const)
        ciph45 = self.rescale(ciph45, self.scaling_factor)

        const = self.create_constant_plain(7)
        ciph = self.add_plain(ciph, const)

        const = self.create_constant_plain(1 / 5040)
        ciph = self.multiply_plain(ciph, const)
        ciph = self.rescale(ciph, self.scaling_factor)

        ciph = self.multiply(ciph, ciph2, relin_key)
        ciph = self.rescale(ciph, self.scaling_factor)

        ciph45 = self.lower_modulus(ciph45, self.scaling_factor)
        ciph = self.add(ciph, ciph45)

        ciph = self.multiply(ciph, ciph4, relin_key)
        ciph = self.rescale(ciph, self.scaling_factor)

        ciph23 = self.lower_modulus(ciph23, self.scaling_factor)
        ciph = self.add(ciph, ciph23)

        return ciph

    def raise_modulus(self, ciph):
        """Raises ciphertext modulus.

        Takes a ciphertext (mod q), and scales it up to mod Q_0. Also increases the scaling factor.

        Args:
            ciph (Ciphertext): Ciphertext to scale up.

        Returns:
            Ciphertext for exponential.
        """
        # Raise scaling factor.
        self.scaling_factor = ciph.modulus
        ciph.scaling_factor = self.scaling_factor

        # Raise ciphertext modulus.
        ciph.modulus = self.big_modulus

    def exp(self, ciph, const, relin_key, encoder):
        """Evaluates the exponential function on the ciphertext.

        Takes an encryption of m and returns an encryption of e^(const * m).

        Args:
            ciph (Ciphertext): Ciphertext to transform.
            const (complex): Constant to multiply ciphertext by.
            relin_key (PublicKey): Relinearization key.
            encoder (CKKSEncoder): Encoder.

        Returns:
            Ciphertext for exponential.
        """
        num_iterations = self.boot_context.num_taylor_iterations
        const_plain = self.create_complex_constant_plain(const / 2**num_iterations, encoder)
        ciph = self.multiply_plain(ciph, const_plain)
        ciph = self.rescale(ciph, self.scaling_factor)
        ciph = self.exp_taylor(ciph, relin_key, encoder)

        for _ in range(num_iterations):
            ciph = self.multiply(ciph, ciph, relin_key)
            ciph = self.rescale(ciph, self.scaling_factor)

        return ciph

    # def bootstrap(self, ciph, rot_keys, conj_key, relin_key, encoder):
    #     """Evaluates the bootstrapping circuit on ciph.

    #     Takes a ciphertext (mod q), that encrypts some value m, and outputs a new
    #     ciphertext (mod Q_0) that also encrypts m, via bootstrapping.

    #     Args:
    #         ciph (Ciphertext): Ciphertext to transform.
    #         rot_keys (dict (RotationKey)): Dictionary of rotation keys, indexed by rotation number
    #         conj_key (PublicKey): Conjugation key.
    #         relin_key (PublicKey): Relinearization key.
    #         encoder (CKKSEncoder): Encoder.

    #     Returns:
    #         Ciphertext for exponential.
    #     """
    #     # Raise modulus.
    #     old_modulus = ciph.modulus
    #     old_scaling_factor = self.scaling_factor

    #     print(f"\n=== TRACKING MODULUS CHANGES ===")
    #     print(f"1. Initial modulus before raise: {int(math.log(ciph.modulus, 2))} bits")
        
    #     # Raise modulus
    #     self.raise_modulus(ciph)
    #     print(f"2. Modulus after raise: {int(math.log(ciph.modulus, 2))} bits")
        
    #     # Coeff to slot
    #     ciph0, ciph1 = self.coeff_to_slot(ciph, rot_keys, conj_key, encoder)
    #     print(f"3. Modulus after coeff_to_slot: {int(math.log(ciph0.modulus, 2))} bits")
        
    #     # Exponentiate
    #     const = self.scaling_factor / old_modulus * 2 * math.pi * 1j
    #     ciph_exp0 = self.exp(ciph0, const, relin_key, encoder)
    #     print(f"4. Modulus after first exp: {int(math.log(ciph_exp0.modulus, 2))} bits")
        
    #     ciph_neg_exp0 = self.conjugate(ciph_exp0, conj_key)
    #     ciph_exp1 = self.exp(ciph1, const, relin_key, encoder)
    #     print(f"5. Modulus after second exp: {int(math.log(ciph_exp1.modulus, 2))} bits")
        
    #     ciph_neg_exp1 = self.conjugate(ciph_exp1, conj_key)

    #     # Compute sine
    #     ciph_sin0 = self.subtract(ciph_exp0, ciph_neg_exp0)
    #     ciph_sin1 = self.subtract(ciph_exp1, ciph_neg_exp1)
    #     print(f"6. Modulus after sine computation: {int(math.log(ciph_sin0.modulus, 2))} bits")

    #     # Scale answer
    #     plain_const = self.create_complex_constant_plain(
    #         old_modulus / self.scaling_factor * 0.25 / math.pi / 1j, encoder)
    #     ciph0 = self.multiply_plain(ciph_sin0, plain_const)
    #     ciph1 = self.multiply_plain(ciph_sin1, plain_const)
    #     print(f"7. Modulus after plain multiplication: {int(math.log(ciph0.modulus, 2))} bits")
        
    #     ciph0 = self.rescale(ciph0, self.scaling_factor)
    #     ciph1 = self.rescale(ciph1, self.scaling_factor)
    #     print(f"8. Modulus after rescaling: {int(math.log(ciph0.modulus, 2))} bits")

    #     # Slot to coeff
    #     old_ciph = ciph
    #     ciph = self.slot_to_coeff(ciph0, ciph1, rot_keys, encoder)
    #     print(f"9. Modulus after slot_to_coeff: {int(math.log(ciph.modulus, 2))} bits")

    #     # Reset scaling factor
    #     self.scaling_factor = old_scaling_factor
    #     ciph.scaling_factor = self.scaling_factor
    #     print(f"10. Final modulus before return: {int(math.log(ciph.modulus, 2))} bits")
    #     print("===============================\n")

    #     return old_ciph, ciph


    def bootstrap(self, ciph, rot_keys, conj_key, relin_key, encoder):
        """Evaluates the bootstrapping circuit on ciph."""
        old_modulus = ciph.modulus
        old_scaling_factor = self.scaling_factor
        
        # print(f"\n=== TRACKING MODULUS CHANGES ===")
        # print(f"1. Initial modulus before raise: {int(math.log(ciph.modulus, 2))} bits")
        
        # Raise modulus
        self.raise_modulus(ciph)
        # print(f"2. Modulus after raise: {int(math.log(ciph.modulus, 2))} bits")
        
        # Coeff to slot
        ciph0, ciph1 = self.coeff_to_slot(ciph, rot_keys, conj_key, encoder)
        # print(f"3. Modulus after coeff_to_slot: {int(math.log(ciph0.modulus, 2))} bits")
        
        # Exponentiate
        const = self.scaling_factor / old_modulus * 2 * math.pi * 1j
        ciph_exp0 = self.exp(ciph0, const, relin_key, encoder)
        ciph_exp1 = self.exp(ciph1, const, relin_key, encoder)
        # print(f"4. Modulus after exp: {int(math.log(ciph_exp0.modulus, 2))} bits")
        
        # Compute sine
        ciph_neg_exp0 = self.conjugate(ciph_exp0, conj_key)
        ciph_neg_exp1 = self.conjugate(ciph_exp1, conj_key)
        ciph_sin0 = self.subtract(ciph_exp0, ciph_neg_exp0)
        ciph_sin1 = self.subtract(ciph_exp1, ciph_neg_exp1)
        # print(f"5. Modulus after sine: {int(math.log(ciph_sin0.modulus, 2))} bits")
        
        # Scale answer
        plain_const = self.create_complex_constant_plain(
            old_modulus / self.scaling_factor * 0.25 / math.pi / 1j, encoder)
        ciph0 = self.multiply_plain(ciph_sin0, plain_const)
        ciph1 = self.multiply_plain(ciph_sin1, plain_const)
        
        ciph0 = self.rescale(ciph0, self.scaling_factor)
        ciph1 = self.rescale(ciph1, self.scaling_factor)
        # print(f"6. Modulus after rescaling: {int(math.log(ciph0.modulus, 2))} bits")
        
        # Slot to coeff
        old_ciph = ciph
        ciph = self.slot_to_coeff(ciph0, ciph1, rot_keys, encoder)
        # print(f"7. Modulus after slot_to_coeff: {int(math.log(ciph.modulus, 2))} bits")
        
        # Final exact modulus reduction - using original parameter modulus
        target_modulus = self.ciph_modulus  # Use the parameter modulus instead of input modulus
        final_reduction = ciph.modulus // target_modulus
        if final_reduction > 1:
            ciph = self.lower_modulus(ciph, final_reduction)
        # print(f"8. Final modulus after reduction: {int(math.log(ciph.modulus, 2))} bits")
        
        # Reset scaling factor
        self.scaling_factor = old_scaling_factor
        ciph.scaling_factor = self.scaling_factor
        
        # print("===============================\n")
        
        return old_ciph, ciph
    
class CKKSKeyGenerator:

    """An instance to generate a public/secret key pair and relinearization keys.

    The secret key s is generated randomly, and the public key is the
    pair (-as + e, a). The relinearization keys are generated, as
    specified in the CKKS paper.

    Attributes:
        params (Parameters): Parameters including polynomial degree, plaintext,
            and ciphertext modulus.
        secret_key (Polynomial): secret key randomly generated from R_q.
        public_key (tuple of Polynomials): public key generated from
            secret key.
        relin_key (tuple of Polynomials): relinearization key generated
            from secret key.
    """

    def __init__(self, params):
        """Generates secret/public key pair for CKKS scheme.

        Args:
            params (Parameters): Parameters including polynomial degree,
                plaintext, and ciphertext modulus.
        """
        self.params = params
        self.generate_secret_key(params)
        self.generate_public_key(params)
        self.generate_relin_key(params)

    def generate_secret_key(self, params):
        """Generates a secret key for CKKS scheme.

        Args:
            params (Parameters): Parameters including polynomial degree,
                plaintext, and ciphertext modulus.
        """
        key = sample_hamming_weight_vector(params.poly_degree, params.hamming_weight)
        self.secret_key = SecretKey(Polynomial(params.poly_degree, key))

    def generate_public_key(self, params):
        """Generates a public key for CKKS scheme.

        Args:
            params (Parameters): Parameters including polynomial degree,
                plaintext, and ciphertext modulus.
        """
        mod = self.params.big_modulus

        pk_coeff = Polynomial(params.poly_degree, sample_uniform(0, mod, params.poly_degree))
        pk_error = Polynomial(params.poly_degree, sample_triangle(params.poly_degree))
        p0 = pk_coeff.multiply(self.secret_key.s, mod)
        p0 = p0.scalar_multiply(-1, mod)
        p0 = p0.add(pk_error, mod)
        p1 = pk_coeff
        self.public_key = PublicKey(p0, p1)

    def generate_switching_key(self, new_key):
        """Generates a switching key for CKKS scheme.

        Generates a switching key as described in KSGen in the CKKS paper.

        Args:
            new_key (Polynomial): New key to generate switching key.

        Returns:
            A switching key.
        """
        mod = self.params.big_modulus
        mod_squared = mod ** 2

        swk_coeff = Polynomial(self.params.poly_degree, sample_uniform(0, mod_squared, self.params.poly_degree))
        swk_error = Polynomial(self.params.poly_degree, sample_triangle(self.params.poly_degree))

        sw0 = swk_coeff.multiply(self.secret_key.s, mod_squared)
        sw0 = sw0.scalar_multiply(-1, mod_squared)
        sw0 = sw0.add(swk_error, mod_squared)
        temp = new_key.scalar_multiply(mod, mod_squared)
        sw0 = sw0.add(temp, mod_squared)
        sw1 = swk_coeff
        return PublicKey(sw0, sw1)

    def generate_relin_key(self, params):
        """Generates a relinearization key for CKKS scheme.

        Args:
            params (Parameters): Parameters including polynomial degree,
                plaintext, and ciphertext modulus.
        """
        sk_squared = self.secret_key.s.multiply(self.secret_key.s, self.params.big_modulus)
        self.relin_key = self.generate_switching_key(sk_squared)

    def generate_rot_key(self, rotation):
        """Generates a rotation key for CKKS scheme.

        Args:
            rotation (int): Amount ciphertext is to be rotated by.

        Returns:
            A rotation key.
        """

        # Generate K_5^r(s).
        new_key = self.secret_key.s.rotate(rotation)
        rk = self.generate_switching_key(new_key)
        return RotationKey(rotation, rk)

    def generate_conj_key(self):
        """Generates a conjugation key for CKKS scheme.

        Returns:
            A conjugation key.
        """

        # Generate K_{-1}(s).
        new_key = self.secret_key.s.conjugate()
        return self.generate_switching_key(new_key)
    
class CKKSParameters:

    """An instance of parameters for the CKKS scheme.

    Attributes:
        poly_degree (int): Degree d of polynomial that determines the
            quotient ring R.
        ciph_modulus (int): Coefficient modulus of ciphertexts.
        big_modulus (int): Large modulus used for bootstrapping.
        scaling_factor (float): Scaling factor to multiply by.
        hamming_weight (int): Hamming weight parameter for sampling secret key.
        taylor_iterations (int): Number of iterations to perform for Taylor series in
            bootstrapping.
        prime_size (int): Minimum number of bits in primes for RNS representation.
        crt_context (CRTContext): Context to manage RNS representation.
    """

    def __init__(self, poly_degree, ciph_modulus, big_modulus, scaling_factor, taylor_iterations=7,
                 prime_size=59):
        """Inits Parameters with the given parameters.

        Args:
            poly_degree (int): Degree d of polynomial of ring R.
            ciph_modulus (int): Coefficient modulus of ciphertexts.
            big_modulus (int): Large modulus used for bootstrapping.
            scaling_factor (float): Scaling factor to multiply by.
            taylor_iterations (int): Number of iterations to perform for Taylor series in
                bootstrapping.
            prime_size (int): Minimum number of bits in primes for RNS representation. Can set to 
                None if using the RNS representation if undesirable.
        """
        self.poly_degree = poly_degree
        self.ciph_modulus = ciph_modulus
        self.big_modulus = big_modulus
        self.scaling_factor = scaling_factor
        self.num_taylor_iterations = taylor_iterations
        self.hamming_weight = poly_degree // 4
        self.crt_context = None

        if prime_size:
            num_primes = 1 + int((1 + math.log(poly_degree, 2) + 4 * math.log(big_modulus, 2) \
             / prime_size))
            self.crt_context = CRTContext(num_primes, prime_size, poly_degree)

    def print_parameters(self):
        """Prints parameters.
        """
        print("Encryption parameters")
        print("\t Polynomial degree: %d" %(self.poly_degree))
        print("\t Ciphertext modulus size: %d bits" % (int(math.log(self.ciph_modulus, 2))))
        print("\t Big ciphertext modulus size: %d bits" % (int(math.log(self.big_modulus, 2))))
        print("\t Scaling factor size: %d bits" % (int(math.log(self.scaling_factor, 2))))
        print("\t Number of Taylor iterations: %d" % (self.num_taylor_iterations))
        if self.crt_context:
            rns = "Yes"
        else:
            rns = "No"
        print("\t RNS: %s" % (rns))
