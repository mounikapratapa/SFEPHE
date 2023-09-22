from gmpy2 import kronecker, is_prime, mpz
import gmpy2
import numpy as np
from sympy.ntheory import factorint
import random
import galois
from random import randint
from functools import reduce
import time
start_time = time.time()
a = 1938
b = 31
k = 512
size = 3072
"""a, b are used to generate the sequence and k is the size of the sequence, 
which in-turn denotes the size of the function. 
Here 512 indicates it is a 9 bit function"""

f = [0] * (256) + [1] * 256
""" f() is the function to implement. We used an example domain that has 
exactly 50% zeros and 50% ones. The convention we'll use is that: 
f(i)=0 denotes a quadratic residue (1), and f(i)= 1 denotes a non-residue (-1)"""

sequence = [a * i + b for i in range(k)]
bl = 3072
B = bl * 20
primorial = gmpy2.primorial(B) // 2

def gcd(a, b):
    """returns the greatest common divisor of a and b"""    
    
    while b:
        return gcd(b, a % b)
    return a

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

def XGCD(a, b):
    """Extended GCD:
    Returns (gcd, x, y), where gcd is greatest common divisor of a and b.
    The numbers x,y are such that gcd = ax+by."""
    prevx, x = 1, 0
    prevy, y = 0, 1
    while b:
        q, r = divmod(a, b)
        x, prevx = prevx - q * x, x
        y, prevy = prevy - q * y, y
        a, b = b, r
    return a, prevx, prevy

def crt(a,b,m=None,n=None):
    """
    This function is similar to the crt algorithm for sagemath's
    miscellaneous arithmetic functions library. We adopted it for python3.
    
    It returns a solution to Chinese Remainder problem, where:
    - a, b are two residues / two lists- one of residues and
      one of moduli.
    - m, n are two moduli. In case a, b are lists the values of m, n are none.
    - if m, n are given, the function returns a solution x to the
    simultaneous congruences "x = a (mod m)" and "x =  b (mod n)", if one exists.
    - According to Chinese remainder theorem, a solution to the
    simultaneous congruences exists if and only if
    "a =  b (mod(gcd(m,n))". 
    - The solution x is  well-defined in "(mod lcm(m,n))".
    - If a, b are lists, then m=n=None and the crt function returns solution to:
        "x =  a_i (mod b_i)", if it exists.    
    """
    if isinstance(a, list):
        return CRT_list(a, b)
    if isinstance(a, (int)): 
        a = int(a) # otherwise q, r = divmod(b-a, g) doesnt work
    g, alpha, beta = XGCD(m, n)
    q, r = divmod(b-a, g)
    if r != 0:
        raise ValueError("No solution to crt problem since gcd(%s,%s) does not divide %s-%s"
                         % (m, n, a, b))
    return (a + (q*alpha*m)) % lcm(m, n)

CRT = crt

def CRT_list(v, moduli):
    """ Given a list v of elements and a list of corresponding
    moduli, find a single element that reduces to each element of
    v modulo the corresponding moduli."""
    if not isinstance(v,list) or not isinstance(moduli,list):
        raise ValueError("Arguments to CRT_list should be lists")
    if len(v) != len(moduli):
        raise ValueError("Arguments to CRT_list should be lists of the same length")
    if len(v) == 0:
        return int(0)
    if len(v) == 1:
        return moduli[0].parent()(v[0])
    x = v[0]
    m = moduli[0]
    for i in range(1,len(v)):
        x = CRT(x,v[i],m,moduli[i])
        m = lcm(m,moduli[i])
    return x%m


def map_fn_to_symbol(x):
    """Converts the elements from function f's codomain to Legendre symbols"""
    if x == 0:
        return 1
    elif x == 1:
        return -1


def find_first_nonzero(a_row):
    """Returns the index of the first non-zero element in  row a"""
    for i in range(len(a_row)):
        if a_row[i] != 0:
            return i


def find_quadratic_residues(number_list):
    """given a list of prime factors, returns the list of ALL quadratic residues
    and quadratic non residues with respect to each prime factor ai to form as a
    look up table"""
    results = []

    for number in number_list:
        residues = []
        non_residues = []

        for x in range(1, number):
            if kronecker(x, number) == 1:
                residues.append(x)
            else:
                non_residues.append(x)

        results.append((number, residues, non_residues))

    return results


def find_random_element_with_symbol(s, residues, non_residues):
    """from a given list of residues and non residues, returns a random choice"""
    
    if s == 1:
        return random.choice(residues)
    else:
        return random.choice(non_residues)


def compute_factor_base(seq):
    """Determines the set of unique prime factors seen across a sequence
    Ignores prime factors of even power"""

    fbase = []
    for element in seq:
        factors = factorint(element)
        for factor, power in factors.items():
            if power % 2 == 1 and factor % 2 == 1 and factor not in fbase:
                fbase.append(factor)
    return fbase


def compute_power(a, b):
    power = 0
    while gcd(a, b) > 1:
        b = int(b // a)
        power += 1
    return power


def compute_linear_system(factor_base):
    """Returns a matrix A of the sequence expressed as a linear system in
    GF(2). Rows represent each element of the sequence. Columns represent
    the presence or absence of each prime factor in that sequence number.
   The last column represents the intented function output at that
   sequence number s, i.e. f(s)"""
    
    system_of_congruences = [] # Build the linear system.

    
    for i in range(len(sequence)):
        """ For each element i of seq, creates a binary vector corresponding to
         whether j is an odd factor of i"""
        sequence_value = sequence[i]
        

        for prime_factor in factor_base:
            """For prime factor of the factor base, checks whether it is present
             in the given sequence number"""
            
            if compute_power(prime_factor, sequence_value) % 2 == 1:
                system_of_congruences.append(1)
            else:
                system_of_congruences.append(0)
                
    
        system_of_congruences.append(f[i])
    #print(system_of_congruences)
    GF = galois.GF(2)
    
    return GF(np.reshape(system_of_congruences, (k, len(factor_base) + 1)))

def compute_prime(bi, factor_base):
    """Returns a squre-free prime number satisfying the crt between
    specified in bi and factor_base of required size"""
    
    p = crt(bi, factor_base)
    x = reduce(lambda a, b: a * b, factor_base)
    p = p + x
    while p.bit_length() < size:
        kn = randint(1, random.getrandbits(size))
        p = p + kn * x
    return p, x


def run():
    factor_base = compute_factor_base(sequence)

    if 2 in factor_base:
        print("ERROR: Factor base contains 2, which may lead to inconsistent results. Please select a sequence containing only odd integers")
        return

    A = compute_linear_system(factor_base)
    A_reduced = A.row_reduce()

    print("System to matrix space--- %s seconds ---" % (time.time() - start_time))
    print("")

    for row in A_reduced:
        if np.all(row[0:-1] == 0) and row[-1] != 0:
            print("ERROR: System is inconsistent. Pick a different a, b")

    prime_factor_symbols = [1] * len(factor_base)

    for row in A_reduced:
        prime_factor_symbols[find_first_nonzero(row)] = map_fn_to_symbol(int(row[-1]))

    print("")
    print("After symbol change:", prime_factor_symbols)
    print("")
    print("ai =", factor_base)
    res_list = find_quadratic_residues(factor_base)
    qr = []
    nr = []
    for num, residues, nonresidues in res_list:
        qr.append(residues)
        nr.append(nonresidues)

    bitime = time.time()
    bi = []

    for i in range(len(factor_base)):
        bi.append(find_random_element_with_symbol(prime_factor_symbols[i], qr[i], nr[i]))

    print("bi =", bi)
    print("Finding b_i--- %s seconds ---" % (time.time() - bitime))

    #crttime = time.time()

    p, x = compute_prime(bi, factor_base)

    findingptime = time.time()
    while  not (is_prime(mpz(p)) and p % 4 == 1):
        print("Could not find prime, trying again")
        bi = []

        for i in range(len(factor_base)):
            bi.append(find_random_element_with_symbol(prime_factor_symbols[i], qr[i], nr[i]))

        p, x = compute_prime(bi, factor_base)

    print("The final prime is =", p)
    print("Total Time --- %s seconds ---" % (time.time() - findingptime))
    for i in range(len(sequence)):
       #print(kronecker(sequence[i], p))
        if map_fn_to_symbol(f[i]) != kronecker(sequence[i], p):
            print("ERROR: symbols of sequence not implement function f")
            return
    print("SUCCESS: Symbols of sequence modulo p implement function f")



run()
