# Superquantum Challenge - iQuHACK 2026

**Team:** QuebHackers  
**Challenge:** Minimizing fault-tolerant cost (clifford+T gates) for quantum circuits.

## Overview

In fault-tolerant quantum computing, non-Clifford gates (specifically T-gates) are computationally expensive. This challenge involves compiling various unitary matrices and quantum operations into efficient sequences of **Clifford + T** gates: $\{H, T, T^\dagger, S, S^\dagger, CNOT\}$.

Our approach leverages a sophisticated multi-stage optimization pipeline combining Pauli decompositions, ZX-calculus reduction, and Reed-Muller synthesis.

---

## Our Optimization Pipeline

We developed a robust pipeline to transform arbitrary unitaries into optimized circuits:

```
    A[Input Unity U] --> B[Pauli Decomposition]
    B --> C[Pauli Gadget Circuit]
    C --> D[GridSynth Approximation]
    D --> E[PyZX Global Reduction]
    E --> F[RMSynth Optimization]
    F --> G[Final Transpilation]
```

1.  **Pauli Decomposition**: Decompose unitary $U = e^{-iH}$ into weighted Pauli strings ($H = \sum h_k P_k$).
2.  **Pauli Gadget Circuit**: Convert Pauli terms into standard gadget circuits (Basis change $\to$ CNOTs $\to$ $R_z(2\theta)$ $\to$ Uncompute).
3.  **GridSynth**: Approximate continuous $R_z$ rotations using discrete Clifford+T sequences.
4.  **PyZX Reduction**: Apply global optimization using ZX-calculus rules to cancel redundant gates and simplify topology.
5.  **RMSynth**: Perform local optimization on blocks of phase gates using Reed-Muller synthesis techniques.

---

## Key Code Implementation

Below are the core functions from our optimization pipeline (`optimize_qasm_with_rmsynth.py`):

### 1. Pauli Decomposition

This function decomposes any unitary $U$ into a weighted sum of Pauli operators:

```python
def decompose_to_pauli(U):
    paulis = ['I', 'X', 'Y', 'Z']
    decomposition = {}
    
    # 1. Calculate Hamiltonian H such that U = exp(-iH)
    # H = i * log(U)
    U_matrix = U.data if hasattr(U, 'data') else U
    H_matrix = 1j * logm(U_matrix)
    
    # Ensure H is Hermitian (numerical error correction)
    H_matrix = (H_matrix + H_matrix.conj().T) / 2
    
    for p1 in paulis:
        for p2 in paulis:
            label = p1 + p2
            # Create Pauli operator P = P_i ⊗ P_j
            P = Operator(Pauli(label)).data
            
            # Project H onto Pauli basis
            # h_k = 1/4 * Tr(P @ H)
            coeff = (1/4) * np.trace(np.dot(P, H_matrix))
            
            # Keep only real part (H is Hermitian)
            real_coeff = np.real(coeff)
            
            if abs(real_coeff) > 1e-10:
                decomposition[label] = real_coeff
                
    return decomposition
```

### 2. Pauli Gadget Circuit Construction

Converts Pauli terms into executable quantum gates:

```python
def build_pauli_gadget_circuit(decomposition):
    """
    Transforms a Pauli decomposition into a circuit via Pauli Gadgets.
    Assumes U = exp(-i * sum(coeff * P))
    """
    qc = QuantumCircuit(2)
    
    for label, coeff in decomposition.items():
        theta = np.real(coeff) 
        
        if label == "II":
            # Global phase (I ⊗ I)
            qc.global_phase += theta
            continue
            
        # 1. Basis change to bring each Pauli to Z-basis
        for i, pauli in enumerate(label):
            if pauli == 'X':
                qc.h(i)
            elif pauli == 'Y':
                qc.sdg(i)
                qc.h(i)
        
        # 2. Entangler (CNOT chain to compute parity)
        qc.cx(0, 1)
        
        # 3. Phase rotation
        qc.rz(2 * theta, 1)
        
        # 4. Reverse entangler
        qc.cx(0, 1)
        
        # 5. Reverse basis change
        for i, pauli in enumerate(label):
            if pauli == 'X':
                qc.h(i)
            elif pauli == 'Y':
                qc.h(i)
                qc.s(i)
        
    return qc
```

### 3. PyZX Global Reduction

Applies ZX-calculus to massively reduce T-count:

```python
def apply_global_pyzx_reduction(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Transforms Qiskit circuit to PyZX graph, applies massive T-count reduction
    (full_reduce + teleport_reduce), and returns new Qiskit circuit.
    """
    # 1. Qiskit -> PyZX via QASM
    qasm_str = qasm2.dumps(circuit)
    zx_circ = zx.Circuit.from_qasm(qasm_str)
    
    # 2. Convert to ZX Graph
    g = zx_circ.to_graph()
    
    # 3. Global Reduction
    # full_reduce() applies all ZX-calculus simplification rules
    zx.full_reduce(g)
    
    # teleport_reduce() is often more efficient for extracting a clean circuit
    # after simplification, as it limits depth
    zx.simplify.teleport_reduce(g)
    
    # 4. Extract simplified circuit
    # Force extraction to basic gates (CNOT, H, T)
    optimized_zx_circ = zx.extract.extract_circuit(g.copy()).to_basic_gates()
    
    # 5. PyZX -> Qiskit
    new_qc = qasm2.loads(optimized_zx_circ.to_qasm())
    
    print(f"PyZX: Initial T-count: {zx_circ.tcount()}")
    print(f"PyZX: Final T-count after reduction: {optimized_zx_circ.tcount()}")
    
    return new_qc
```

### 4. RMSynth Block Optimization

Final phase polynomial optimization using Reed-Muller synthesis:

```python
def optimize_qasm_with_rmsynth(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Takes a circuit in QASM format, optimizes with rmsynth,
    and returns the result in QASM format.
    NOTE: Input is composed of H + CX + Rz + S + Sdg gates
    """
    # 1. Transform Rz to Clifford+T gates (Grid synthesis)
    qc = decompose_rz_circuit(circuit)

    # 2. Apply PyZX reduction
    qc = apply_global_pyzx_reduction(qc)

    # 3. Block-based optimization (to preserve H gates)
    # Strategy: Accumulate gates (CX, T, S, Z) in an rmsynth block
    # When we see an H gate (or other unsupported), optimize current block,
    # add it to final circuit, then add the H gate
    n = qc.num_qubits
    qc_output = QuantumCircuit(n, qc.num_clbits)
    
    rm_circ = Circuit(n_qubits=n)
    block_is_empty = True
    optimizer = Optimizer()

    def flush_rmsynth_block(rm_c, target_qc):
        """Optimizes current rmsynth block and adds it to target_qc."""
        nonlocal block_is_empty
        if block_is_empty:
            return
            
        # Block optimization
        optimized_rm, report = optimizer.optimize(rm_c)
        print(f"[Block] T-count: {report.before_t} -> {report.after_t}") 
        
        # Reconstruct Qiskit from optimized block
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

    for instruction in qc.data:
        gate = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        q_idx = [qc.find_bit(q).index for q in instruction.qubits]
        
        # Is this a gate optimizable by rmsynth?
        is_supported = False
        if gate.name == 'cx':
            rm_circ.add_cnot(q_idx[0], q_idx[1])
            is_supported = True
        elif gate.name in ['t', 'tdg', 's', 'sdg', 'z']:
            # Map to phase gate Rz(k*π/4)
            phase_map = {'t': 1, 'tdg': 7, 's': 2, 'sdg': 6, 'z': 4}
            rm_circ.add_phase(q_idx[0], phase_map[gate.name])
            is_supported = True
            
        if is_supported:
            block_is_empty = False
        else:
            # Unsupported gate (e.g., H, measure, barrier)
            flush_rmsynth_block(rm_circ, qc_output)
            rm_circ = Circuit(n_qubits=n)
            block_is_empty = True
            qc_output.append(gate, qubits, clbits)

    # Final flush
    flush_rmsynth_block(rm_circ, qc_output)
    return qc_output
```

---

## Detailed Approach by Challenge

Here is how we tackled each of the 11 specific tasks:

### 1. Controlled-Y Gate (Sanity Check)
**Target**: Implement a Controlled-Y gate.  
**Approach**: 
Instead of running the heavy pipeline, we used the analytical identity $Y = S X S^\dagger$.
*   **Circuit**: Applying $S$ on the target, then a $CNOT$, then $S^\dagger$ on the target implements $CY$.
*   **Result**: $S \cdot CX \cdot S^\dagger$.

### 2. Controlled-Ry(π/7)
**Target**: A controlled interaction with a specific rotation angle.  
**Approach**: 
This served as the first full test of our **End-to-End Pipeline**.
1.  Decomposed the unitary into Pauli terms.
2.  Generated Pauli gadgets.
3.  Approximated the rotation $R_y(\pi/7)$ (converted to Pauli basis) using `GridSynth`.
4.  Optimized with `RMSynth` to minimize T-count.

### 3. exp(iπ/7 Z⊗Z)
**Target**: Exponential of a Pauli string (Quantum Simulation).  
**Approach**: 
We recognized this as a standard **ZZ Interaction Gadget**.
*   **Decomposition**: $e^{i\theta Z \otimes Z}$ is implemented by $CNOT \to R_z(2\theta) \to CNOT$.
*   We built this skeleton explicitly and then used `GridSynth` to find the optimal Clifford+T sequence for the central $R_z(2\theta)$ rotation.

### 4. exp(iπ/7 (XX + YY))
**Target**: Hamiltonian with $XX$ and $YY$ terms (XY interaction).  
**Approach**: 
1.  Decomposed the Hamiltonian $H = XX + YY$ into its constituent Pauli terms.
2.  Built separate gadgets for the $XX$ and $YY$ parts (using Trotterization concepts if necessary, though direct decomposition works for commuting terms).
3.  Fed the resulting circuit through the full optimization pipeline (PyZX + RMSynth).

### 5. exp(iπ/4 (XX + YY + ZZ))
**Target**: 2-qubit Heisenberg model interaction.  
**Approach**: 
**Key Insight**: The operator $XX + YY + ZZ$ is mathematically related to the SWAP operator.
$$ XX + YY + ZZ = 2 \cdot \text{SWAP} - I $$
*   Therefore, the unitary is proportional to a partial SWAP.
*   We implemented the SWAP gate using 3 CNOTs and adjusted the global phase, resulting in a highly efficient circuit without needing complex approximations.

### 6. exp(iπ/7 (XX + ZI + IZ))
**Target**: Transverse Field Ising Model.  
**Approach**: 
1.  Decomposed the Hamiltonian into three non-commuting terms: $XX$, $ZI$, and $IZ$.
2.  Used our **Pauli Decomposition** engine to generate the weighted Sum of Paulis.
3.  Constructed the circuit using standard gadgets and ran the full optimizer to merge common sub-circuits and reduce T-count.

### 7. State Preparation
**Target**: Prepare a specific random 4-qubit state $|\psi\rangle$ from $|0000\rangle$.  
**Approach**: 
1.  Used Qiskit's `StatePreparation` library (based on Shende-Bullock-Markov) to generate a baseline circuit with standard gates ($U3, CX$).
2.  Decomposed these high-level gates into our Pauli basis representation.
3.  Optimized the resulting sequence using `RMSynth` to compress the T-gates introduced by the state prep algorithm.

### 8. Structured Unitary 1
**Target**: A specific $4 \times 4$ unitary matrix.  
**Approach**: 
**Key Insight**: We analyzed the matrix and identified it as the **Quantum Fourier Transform (QFT)** on 2 qubits.
*   **Formula**: $QFT_2 = (H \otimes I) \cdot CP(\pi/2) \cdot (I \otimes H) \cdot SWAP$.
*   We used Qiskit's `QFT` library to generate the standard circuit and then optimized the specific $CP(\pi/2)$ (Controlled-Phase) gate into Clifford+T.

### 9. Structured Unitary 2
**Target**: Another structured $4 \times 4$ matrix.  
**Approach**: 
1.  Performed matrix analysis to check for standard forms.
2.  Since no immediate named gate was obvious, we relied on our robust **General Compilation Strategy**:
    *   `decompose_to_pauli(U)` $\to$ `optimize_qasm_with_rmsynth`.
    *   This ensured any hidden structure was captured by the algebraic simplifications in PyZX and RMSynth.

### 10. Random Unitary
**Target**: Compile a completely random $4 \times 4$ unitary (Quantum Supremacy regime).  
**Approach**: 
This is the stress test for the compiler.
1.  Directly decomposed the $4 \times 4$ unitary into a sequence of Pauli rotations.
2.  Applied aggressive optimization levels.
3.  The pipeline handled the complexity by breaking it down into small, optimized chunks.

### 11. 4-Qubit Diagonal Unitary
**Target**: A phase polynomial $U|x\rangle = e^{i\phi(x)}|x\rangle$.  
**Approach**: 
**Key Insight**: Diagonal unitaries correspond to **Phase Polynomials**.
*   This is the ideal use case for `RMSynth` (Reed-Muller Synthesis) and `PyZX`.
*   We constructed the circuit using a `Diagonal` gate.
*   We leveraged the fact that the phases were multiples of $\pi/4$. This allows for exact synthesis using T-gates rather than approximations, resulting in a very precise and efficient circuit.

---

## Technology Stack

*   **Qiskit**: Core circuit handling and transpilation.
*   **RMSynth**: Specialized phase polynomial optimization.
*   **PyZX**: ZX-calculus based circuit reduction.
*   **PyGridSynth**: Clifford+T approximation for rotation gates.
*   **NumPy/SciPy**: Linear algebra and matrix decomposition.

---

## Implementation Note

> **Note**: Due to time constraints during the hackathon, the actual implementation for some challenges may vary slightly from the approaches described above. The core pipeline and optimization techniques remain consistent, but specific decomposition strategies or circuit constructions were adapted on a per-challenge basis to meet submission deadlines while maintaining correctness and optimization quality.

---

*Generated for Superquantum Challenge - January 2026*
