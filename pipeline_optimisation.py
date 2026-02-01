from qiskit import QuantumCircuit, transpile, qasm2
from rmsynth import Circuit, Optimizer
import pyzx as zx
import numpy as np
import pygridsynth as ps
from scipy.linalg import logm
from qiskit.quantum_info import Operator, Pauli
from qiskit.quantum_info.operators import Pauli


def build_pauli_gadget_circuit(decomposition):
    """
    Transforme une décomposition de Pauli en circuit via Pauli Gadgets.
    On suppose U = exp(-i * sum(coeff * P))
    """
    qc = QuantumCircuit(2)
    
    for label, coeff in decomposition.items():
        # On extrait l'angle (la norme du coefficient complexe)
        # Note: En pratique, le lien entre U et H nécessite un logarithme de matrice
        theta = np.real(coeff) 
        
        if label == "II":
            # La phase globale (I \otimes I)
            qc.global_phase += theta
            continue
            
        # 1. Changement de base pour ramener chaque Pauli vers la base Z
        # On applique sur chaque qubit selon le caractère X, Y ou Z
        for i, pauli in enumerate(label):
            if pauli == 'X':
                qc.h(i)
            elif pauli == 'Y':
                qc.sdg(i)
                qc.h(i)
        
        # 2. Intricateur (Chaîne de CNOT pour calculer la parité)
        qc.cx(0, 1)
        
        # 3. Rotation de phase
        qc.rz(2 * theta, 1)
        
        # 4. Inverse de l'intricateur
        qc.cx(0, 1)
        
        # 5. Inverse du changement de base
        for i, pauli in enumerate(label):
            if pauli == 'X':
                qc.h(i)
            elif pauli == 'Y':
                qc.h(i)
                qc.s(i)
        
    return qc


def decompose_to_pauli(U):
    paulis = ['I', 'X', 'Y', 'Z']
    decomposition = {}
    
    # 1. Calculer l'Hamiltonien H tel que U = exp(-iH)
    # H = i * log(U)
    # U est un Operator qiskit, on prend .data
    U_matrix = U.data if hasattr(U, 'data') else U
    H_matrix = 1j * logm(U_matrix)
    
    # On s'assure que H est Hermitien (correction d'erreurs numériques)
    H_matrix = (H_matrix + H_matrix.conj().T) / 2
    
    for p1 in paulis:
        for p2 in paulis:
            label = p1 + p2
            # Création de l'opérateur de Pauli P = P_i \otimes P_j
            P = Operator(Pauli(label)).data
            
            # Projection de H sur la base de Pauli
            # h_k = 1/4 * Tr(P @ H)
            coeff = (1/4) * np.trace(np.dot(P, H_matrix))
            
            # On ne garde que la partie réelle (H est hermitien)
            real_coeff = np.real(coeff)
            
            if abs(real_coeff) > 1e-10:
                decomposition[label] = real_coeff
                
    return decomposition


def get_rz_approx_circuit(theta: str, epsilon: str="0.1") -> QuantumCircuit:
    """
    Generates a quantum circuit approximating the rotation Rz(theta) using the GridSynth library.
    Epsilon determines the approximation precision.
    The smaller epsilon is, the more precise the approximation, but the circuit will be longer.
    Epsilon is between 0 and 1.
    Returns a Qiskit QuantumCircuit.
    The output consists of Clifford+T gates.
    """    
    # 1. Retrieve gates
    gates_list = ps.gridsynth_gates(theta, epsilon=epsilon)
    
    # 2. Convert to string
    gate_string = "".join([str(g) for g in gates_list])
    
    # 3. Reconstruct the circuit
    qc = QuantumCircuit(1)
    for char in reversed(gate_string):  # Reverse for temporal order
        if char == 'H':
            qc.h(0)
        elif char == 'T':
            qc.t(0)
        elif char == 't':
            qc.tdg(0)
        elif char == 'S':
            qc.s(0)
        elif char == 's':
            qc.sdg(0)
        elif char in ('Z', 'z'):
            qc.z(0)
        elif char in ('X', 'x'):
            qc.x(0)
        elif char in ('Y', 'y'):
            qc.y(0)
        else:
            print(f"Unknown gate : {char}")
    return qc


def apply_global_pyzx_reduction(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Transforme le circuit Qiskit en graphique PyZX, applique une réduction 
    massive du T-count (full_reduce + teleport_reduce), et retourne 
    un nouveau circuit Qiskit.
    """
    # 1. Conversion Qiskit -> PyZX via QASM
    qasm_str = qasm2.dumps(circuit)
    zx_circ = zx.Circuit.from_qasm(qasm_str)
    
    # 2. Conversion en Graphe ZX
    g = zx_circ.to_graph()
    
    # 3. Réduction Globale
    # full_reduce() applique toutes les règles de simplification du ZX-calculus
    zx.full_reduce(g)
    
    # teleport_reduce() est souvent plus efficace pour extraire un circuit 
    # propre après la simplification, car il limite la profondeur.
    zx.simplify.teleport_reduce(g)
    
    # 4. Extraction du circuit simplifié
    # On force l'extraction vers des portes de base (CNOT, H, T)
    optimized_zx_circ = zx.extract.extract_circuit(g.copy()).to_basic_gates()
    
    # 5. Conversion PyZX -> Qiskit
    # On repasse par le QASM pour reconstruire l'objet QuantumCircuit
    new_qc = qasm2.loads(optimized_zx_circ.to_qasm())
    
    # Petit check pour le log
    print(f"PyZX: T-count initial estimé: {zx_circ.tcount()}")
    print(f"PyZX: T-count final après réduction: {optimized_zx_circ.tcount()}")
    
    return new_qc


def decompose_rz_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Décompose les portes Rz du circuit en portes Clifford+T.
    Retourne un circuit composé de Clifford+T.
    """
    new_qc = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit.data:
        gate = instruction.operation
        qubits = instruction.qubits
        if gate.name == 'rz':
            angle = gate.params[0]
            approx_qc = get_rz_approx_circuit(str(angle))
            new_qc.compose(approx_qc, qubits, inplace=True)
        else:
            new_qc.compose(gate, qubits, inplace=True)
    return new_qc


def optimize_qasm_with_rmsynth(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Prend un circuit en format QASM, l'optimise avec rmsynth,
    et retourne le résultat en format QASM.
    NOTE : L'input est composé de portes H + Cx + Rz + S + sdg
    """
    # 1. Transformer les Rz en portes de clifford + T (Synthèse de grille)
    qc = decompose_rz_circuit(circuit)

    # 2. Appliquer une réduction avec PyZX
    qc = apply_global_pyzx_reduction(qc)

    # 3. Optimisation par BLOCS (pour préserver les H)
    # Stratégie : On accumule les portes (CX, T, S, Z) dans un bloc rmsynth
    # Quand on voit une porte H (ou autre non supportée), on optimise le bloc courant,
    # On l'ajoute au circuit final, puis on ajoute la porte H.
    n = qc.num_qubits
    qc_output = QuantumCircuit(n, qc.num_clbits)
    
    # Circuit rmsynth temporaire pour le bloc courant
    rm_circ = Circuit(n_qubits=n)
    block_is_empty = True
    
    optimizer = Optimizer()

    def flush_rmsynth_block(rm_c, target_qc):
        """
        Optimise le bloc rmsynth courant et l'ajoute au target_qc.
        """
        nonlocal block_is_empty
        if block_is_empty:
            return
            
        # Optimisation du bloc
        optimized_rm, report = optimizer.optimize(rm_c)
        print(f"[Block] T-count: {report.before_t} -> {report.after_t}") 
        
        # Reconstruction Qiskit du bloc optimisé
        for g in optimized_rm.ops:
            if g.kind == "cnot":
                target_qc.cx(g.ctrl, g.tgt)
            elif g.kind == "phase":
                k = g.k % 8
                if k == 1: target_qc.t(g.q)
                elif k == 2: target_qc.s(g.q)
                elif k == 3: 
                    target_qc.s(g.q)
                    target_qc.t(g.q)
                elif k == 4: target_qc.z(g.q)
                elif k == 5:
                    target_qc.z(g.q)
                    target_qc.t(g.q)
                elif k == 6: target_qc.sdg(g.q)
                elif k == 7: target_qc.tdg(g.q)
        
        # Reset pour le prochain bloc
        # Note: on ne peut pas 'vider' rm_c facilement, on en recrée un ou on le vide
        # rmsynth.Circuit n'a pas de méthode clear(), donc il faut le recréer dans la boucle principale
        pass

    for instruction in qc.data:
        gate = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        q_idx = [qc.find_bit(q).index for q in instruction.qubits]
        
        # Est-ce une porte optimisable par rmsynth ?
        is_supported = False
        if gate.name == 'cx':
            rm_circ.add_cnot(q_idx[0], q_idx[1])
            is_supported = True
        elif gate.name == 't':
            rm_circ.add_phase(q_idx[0], 1)
            is_supported = True
        elif gate.name == 'tdg':
            rm_circ.add_phase(q_idx[0], 7)
            is_supported = True
        elif gate.name == 's':
            rm_circ.add_phase(q_idx[0], 2)
            is_supported = True
        elif gate.name == 'sdg':
            rm_circ.add_phase(q_idx[0], 6)
            is_supported = True
        elif gate.name == 'z':
            rm_circ.add_phase(q_idx[0], 4)
            is_supported = True
            
        if is_supported:
            block_is_empty = False
        else:
            # Porte NON supportée (ex: H, mesure, barrier)
            # 1. On "flush" (optimise) ce qu'on a accumulé avant
            flush_rmsynth_block(rm_circ, qc_output)
            
            # 2. On reset le bloc rmsynth
            rm_circ = Circuit(n_qubits=n)
            block_is_empty = True
            
            # 3. On ajoute la porte non supportée telle quelle au circuit final
            qc_output.append(gate, qubits, clbits)

    # 4. Flush final (si le circuit finit par un bloc optimisable)
    flush_rmsynth_block(rm_circ, qc_output)
            
    # 6. Retourner en format QASM via qasm2
    return qc_output



if __name__ == "__main__":
    m2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0.97, -0.22],
        [0, 0, 0.22, 0.97]
    ])

    # --- Processing Loop ---
    U = Operator(m2)

    # Transforme en circuit via Pauli
    paulis = decompose_to_pauli(U)
    circuit = build_pauli_gadget_circuit(paulis)    # Consitué de : H + Cx + Rz + S + sdg

    # Optimise le circuit
    optimized_circuit = optimize_qasm_with_rmsynth(circuit)

    # Garder uniquement H, T, Tdg, S, Sdg, CX
    basis = ['h', 't', 'tdg', 's', 'sdg', 'cx']
    qc_transpiled = transpile(optimized_circuit, basis_gates=basis, optimization_level=1)

    # Mettre circuit en format QASM
    filename = f"submission_task2.qasm"
    QASM = qasm2.dumps(qc_transpiled)
    with open(filename, "w") as f:
        f.write(QASM)
        
    print(f"✅ Fichier '{filename}' généré avec succès !")
