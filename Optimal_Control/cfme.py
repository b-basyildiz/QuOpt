__version__ = 'v0.1.0'
__author__ = 'George Grattan'

import numpy as np
from numpy.polynomial.legendre import leggauss

import torch
from torch import tensor, sqrt, conj
from torch.linalg import matrix_exp

def generate_cfme_unitary(A, t0:torch.float64, T:torch.float64, h:float, n:int, m:int):
    operator_arr = cfme_with_final_time(A, t0, T, h, n,m)
    operator_tensor = torch.stack(operator_arr)

    unitary = generate_unitary_from_operator_stack(operator_tensor)

    return unitary


def generate_unitary_from_operator_stack(operators):
    """
    Convert 4th rank tensor of stacked quantum operators to a unitary.

    According to equation (73-74) in the paper, the operators must be applied
    in REVERSE order: exp(D_m) ... exp(D_2) exp(D_1) state

    Args:
        operators: Tensor of shape (num_timesteps, m, dim, dim)

    Returns:
        unitary which is product of all operations
    """
    num_timesteps, m, N, M = operators.shape
    assert N == M
    result = torch.eye(N,dtype=torch.complex128)

    for i in range(num_timesteps):
        # Apply operators in REVERSE order (from m-1 down to 0)
        for j in reversed(range(m)):
            result = operators[i, j, :, :] @ result
    return result

def apply_operator_stack(state, operators):
    """
    Apply stacked quantum operators to a state.

    According to equation (73-74) in the paper, the operators must be applied
    in REVERSE order: exp(D_m) ... exp(D_2) exp(D_1) state

    Args:
        operators: Tensor of shape (num_timesteps, m, dim, dim)
        state: Tensor of shape (dim,) or (dim, 1)

    Returns:
        Final state after all operations
    """

    num_timesteps = operators.shape[0]
    m = operators.shape[1]
    result = state.clone()

    for i in range(num_timesteps):
        # Apply operators in REVERSE order (from m-1 down to 0)
        for j in reversed(range(m)):
            result = operators[i, j, :, :] @ result
        result /=torch.norm(result)
    return result



def cfme(A, t0:torch.float64, h:torch.float64, N:int, n:int, m:int):
     cfme_operator_arr = []
     for i in range(N):
        curr_t0 = t0+h*i
        cfme_operator_arr.append(cfme_timestep(A=A, t0=curr_t0, h=h, n=n,m=m))


     return cfme_operator_arr

def cfme_with_final_time(A, t0:torch.float64, T:torch.float64, h:float, n:int, m:int):
    cfme_operator_arr = []
    N = int(T//h) #number of magnus steps

    for i in range(N):
        curr_t0 = t0+h*i
        cfme_operator_arr.append(cfme_timestep(A=A, t0=curr_t0, h=h, n=n,m=m))

    #handle the remainder
    h_remainder = h*(T/h -N)
    curr_t0 = t0+h*N
    cfme_operator_arr.append(cfme_timestep(A=A,t0=curr_t0, h=h_remainder, n=n, m=m ))

    return cfme_operator_arr


def cfme_timestep(A, t0:torch.float64, h:torch.float64, n:int, m:int):
     """
     Implements
     """

     if n == 4:
          return fourth_order_cfme(A, t0, h, n, m)

     elif n == 6:
          return sixth_order_cfme(A, t0, h, n, m)
     else:
          raise ValueError(f"order, n, must be eithger 4 or 6")

def fourth_order_cfme(A, t0:torch.float64,h:torch.float64, n:int, m:int)->tensor:
    s = int(n/2)
    r = int(n-2)

    b1 = commutator_free_magnus_operators(A, t0, h, s, r, i=1)
    b2 = commutator_free_magnus_operators(A, t0, h, s, r, i=2)


    if m == 2: #eq 37
        x11 = 1/2
        x12 = 1/6

        #construct the operators and ensure they are unitary
        op1 = make_unitary( matrix_exp(x11 * b1 + x12*b2))
        op2 = make_unitary( matrix_exp(x11 * b1 - x12*b2))

        return torch.stack((op1, op2))

    elif m==3: #eq. 38
        x11 = 0
        x21 = 1
        x12 = 1/12

        #construct the operators
        op1 = make_unitary( matrix_exp(x11*b1 + x12*b2))
        op2 = make_unitary( matrix_exp(x21*b1))
        op3 = make_unitary( matrix_exp(x11*b1 - x12*b2))

        #Note: Since x11 = 0 we can optimize the following expression but I kept it in case we decided to change a choice of x11
        return torch.stack((op1, op2, op3))
    else:
        raise NotImplementedError("m >3 is not currently implemented for 4th order")


def sixth_order_cfme(A, t0:torch.float64,h:torch.float64,n:int,m:int):
    s = int(n/2)
    r = int(n-2)
    b1 = commutator_free_magnus_operators(A, t0, h, s, r, i=1)
    b2 = commutator_free_magnus_operators(A, t0, h, s, r, i=2)
    b3 = commutator_free_magnus_operators(A, t0, h, s, r, i=3)

    if m == 5:
        x11 = 0.2
        x12 = 0.08734395950888931101
        x13 =  0.03734395950888931101

        x21 = 0.34815492558797391479
        x22 = 0.053438272547684150
        x23 =  0.00584269157837031012

        x31 = 1-2*(x11 + x21)
        x32 = 0
        x33 = (1/12) - 2*(x13 + x23)

        op1 = make_unitary( matrix_exp(x11*b1 + x12*b2 + x13*b3))
        op2 = make_unitary( matrix_exp(x21*b1 + x22*b2 + x23*b3))

        op3 = make_unitary( matrix_exp(x31*b1 + x32*b2 + x33*b3))

        op4 = make_unitary( matrix_exp(x21*b1 - x22*b2 + x23*b3))
        op5 = make_unitary( matrix_exp(x11*b1 - x12*b2 + x13*b3))

        return torch.stack((op1, op2, op3, op4, op5))

    elif m == 6:

        #TODO: Store these coefficients as a matrix and then use torch to generate the operators with matrix operations instead of doing it manually like below
        x11 = 0.208
        x12 = 0.09023186422416794596
        x13 = 0.03823186422416794596

        x21 = 0.312
        x22 = 0.04467385661651479788
        x23 = 0.00439421553992544024

        x31 = 1/2 - (x11 + x21)
        x32 = 0.01407960659498524468
        x33 = 1/24 - (x13 + x23)

        op1 = make_unitary( matrix_exp(x11*b1 + x12*b2 + x13*b3))
        op2 = make_unitary( matrix_exp(x21*b1 + x22*b2 + x23*b3))
        op3 = make_unitary( matrix_exp(x31*b1 + x32*b2 + x33*b3))

        op4 = make_unitary( matrix_exp(x31*b1 - x32*b2 + x33*b3))
        op5 = make_unitary( matrix_exp(x21*b1 - x22*b2 + x23*b3))
        op6 = make_unitary( matrix_exp(x11*b1 - x12*b2 + x13*b3))


        return torch.stack((op1, op2, op3, op4, op5, op6))


def commutator_free_magnus_operators(A, t:torch.float64, h:torch.float64,  s:int, r:int, i:int)->torch.tensor:
    """
Constructs the commutator free operators, b_i, from equation (25), These generate a Lie algebra which is equivalent to the commutators in the normal magnus expansion ...
\n
    A (func: float -> tensor):  A(t) is the time dependent tensor operator dictating the system evolution\n
    t (float):                 current starting time\n
    h (float):                  timestep\n
    s (int):                    quadrature degree, if we want a magnus approximant of order n, s=n/2\n
    r (int):                    This is the number of magnus terms we account for. , if we want an approximant of order n, r = 2s-2 = n-2\n
    i (int):                    index for which operator we'd like to generate\n
    """

    #create the tensor to hold the data for output
    operator_shape = A(0).shape
    b_i = torch.zeros(operator_shape, dtype=torch.complex128)

    nodes, weights = leggauss(r) #Note this gets called twice, once here, once in legendre_gauss_quad_matrix

    # Transform nodes from [-1,1] to [0,1]
    nodes_01 = (nodes+1)/2

    R_s = special_inversion_matrix(s)
    Q_G_s_r = legendre_gauss_quad_matrix(s,r)
    RxQ = R_s @ Q_G_s_r

    for j in range(r):
        A_j = A(t + nodes_01[j]*h) #A_i = A(t_0 + c_i*h) just above eq. 20
        b_i += RxQ[i-1,j] * A_j #i-1 because the paper starts with index 1 while python starts with zero

    #multiply by h and -i and return
    return h*b_i

def special_inversion_matrix(s:int):
    """
    Matrix R, defined in eq (24), hardcoded for simplicity

    s (int):    quadrature degree/number of terms
    """
    if s == 2:
        R = ((1,0),(0,12))

    if s == 3:
        R = ((9/4, 0 , 15), (0, 12, 0),(-15, 0, 180))
    return tensor(R, dtype=torch.complex128)

def legendre_gauss_quad_matrix(s:int, r:int) ->torch.tensor:
    """
given the quadrature degree, s, and the magnus order, r, generate the matrix defined in Eq (22) in the main
reference.

This implementation assumes an integral of [-h/2, h/2], which when doing a change of interval to a gaussian
quadrature introduces a factor of 1/2

NOTE: I could NOT replicate the matrices from eq 22. using eq. 21, so I reverse engineered the construction of
eq. 22 using the legendre-gauss weights and nodes.


    s (int):    quadrature degree or in this case, how many weights and nodes of the legendre-gauss we need
                if we want an approximant of order n, s=n/2

    r (int):    This is the number of magnus terms we account for. , if we want an approximant of order n, r = 2s-2 = n-2
                Furthermore
    """
    nodes, weights = leggauss(r) #nodes on [-1,1]

    #transform nodes to [0,1]

    nodes_01 = (nodes+1)/2

    mat = []
    for i in range(s):
        row = []
        for j, weight in enumerate(weights):

            node = nodes_01[j]  # Use transformed node
            # This gives (c_j - 1/2)^i with proper weight scaling
            entry = weight * (node - 0.5)**i * (1/2)

            #entry = weight * (node - 1/2)**(i) #This what eq. 21 presents
            #entry =  weight * node**i * (1/2)**(i+1) #This is what I used to reconstruct eq.22
            row.append(entry)
        mat.append(row)
    return tensor(mat, dtype=torch.complex128)

#Helper Functions

def dag(op:torch.tensor):
     """
     Implements the conjugate transpose for a torch tensor

     op (torch.tensor): operator to conjugate transpose
     """

     return conj(op.T)

def make_unitary(op:torch.tensor):
    f"""
    Ensures a given operator, U, is unitary with the following formula

    op (torch.tensor):

    """
    #op_dag = dag(op)

    #return op @ torch.inverse(sqrt(op_dag @ op))

    #U, _, Vh = torch.linalg.svd(op)
    #return U @ Vh

    return op
