import pennylane as qml
import pennylane.numpy as np

def ALA(m, n, L, observables, dev):
    """
    Create alternating layered ansatz.

    Args: 
        m: number of qubits in one block
        n: number of qubits 
        L: depth of circuit
        observales: list of observables
        dev: device
    Return:
        qml.QNode(circuit)
    """
    # we assume the number of qubits in one block could device the total number of qubits 
    assert n%m==0
    num_blocks = n//m
    wires = np.arange(n)
    gate_set = [qml.RX, qml.RY, qml.RZ]
    def circuit(guided_state_vector, weights):
        qml.QubitStateVector(guided_state_vector, wires = range(n))

        for layer in range(L):
            if layer%2 == 0:
                block_start_qubits = np.arange(0, n, m)
                for block_id in range(num_blocks):
                    qml.broadcast(np.random.choice(gate_set, 1)[0], wires=wires[block_start_qubits[block_id]: block_start_qubits[block_id]+m], \
                        pattern='single', parameters=weights[layer*L+block_start_qubits[block_id]:layer*L+block_start_qubits[block_id]+m])
                    qml.broadcast(qml.CNOT, wires=wires[block_start_qubits[block_id]: block_start_qubits[block_id]+m], \
                        pattern='double')
                    qml.broadcast(qml.CNOT, wires=wires[block_start_qubits[block_id]: block_start_qubits[block_id]+m], \
                        pattern='double_odd')
            else:
                block_start_qubits = [0]
                block_start_qubits.extend(np.arange(m//2, n-m//2, m))
                block_start_qubits.append(n-m//2)
                for block_id in range(num_blocks+1):
                    if block_id == 0 or block_id == num_blocks:
                        qml.broadcast(np.random.choice(gate_set, 1)[0], wires=wires[block_start_qubits[block_id]: block_start_qubits[block_id]+m//2], \
                        pattern='single', parameters=weights[layer*L+block_start_qubits[block_id]:layer*L+block_start_qubits[block_id]+m//2])
                        qml.broadcast(qml.CNOT, wires=wires[block_start_qubits[block_id]: block_start_qubits[block_id]+m//2], \
                            pattern='double')
                        qml.broadcast(qml.CNOT, wires=wires[block_start_qubits[block_id]: block_start_qubits[block_id]+m//2], \
                        pattern='double_odd')
                    else:
                        qml.broadcast(np.random.choice(gate_set, 1)[0], wires=wires[block_start_qubits[block_id]: block_start_qubits[block_id]+m], \
                        pattern='single', parameters=weights[layer*L+block_start_qubits[block_id]:layer*L+block_start_qubits[block_id]+m])
                        qml.broadcast(qml.CNOT, wires=wires[block_start_qubits[block_id]: block_start_qubits[block_id]+m], \
                            pattern='double')
                        qml.broadcast(qml.CNOT, wires=wires[block_start_qubits[block_id]: block_start_qubits[block_id]+m], \
                            pattern='double_odd')
            
        return [qml.expval(o) for o in observables]
    return qml.QNode(circuit, device=dev)
                    
            
