import pennylane as qml
import pennylane.numpy as qnp
import numpy as np

from typing import Tuple

def local_result(
    n_qubits: int,
    x: np.ndarray,
    circuit: qml.QNode,
    params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the function value from the output and gradient w.r.t the parameters

    Args:
        n_qubits: number of qubits
        x: guided_state of x
        circuit: quantum circuit
        params: list of parameters
    Return:
        f_x, grad
         
    """

    # Compute fucntion value of the circuit
    f_x = circuit(params)
    f_x = np.sum(f_x)/np.sqrt(n_qubits)
    
    #Compute the gradient w.r.t parameters via parameter-shift rules
    grad = qml.gradients.param_shift(circuit)(x, params)
    grad = np.sum(np.array(grad), axis=0)/np.sqrt(n_qubits)
    
    return f_x, grad

def kernel_entry(
    x: np.ndarray, 
    x_prime: np.ndarray,
    n_qubits: int, 
    circuit: qml.QNode, 
    params: np.ndarray):
    """
    Calucate the kernel entry of x and x_prime

    Args: 
        x: guided_state w.r.t x
        x_prime: guided_state w.r.t x_prime
        n_qubits: number of qubits
        circuit: parameterized circuit
        params: training parameters
    Returns:
        the kernel entry of x, x_prime
    """
    # Compute the gradient of the function with input of x and x_prime
    _, grad_x = local_result(n_qubits, x, circuit, params)
    _, grad_x_prime = local_result(n_qubits, x_prime, circuit, params)

    # The kernel entry as the dot product between the two gradient vectors
    result = np.dot(grad_x, grad_x_prime)
    
    return result 

    



