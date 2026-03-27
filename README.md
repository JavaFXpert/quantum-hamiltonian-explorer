# Quantum Hamiltonian Explorer

An interactive educational course that teaches high school students how simple molecules are represented as quantum Hamiltonians, how those Hamiltonians are diagonalized to find energy levels, and how quantum states evolve over time.

## Overview

The application is a single-file React JSX artifact designed to run in the Claude desktop/web app. It walks learners through 4 phases of quantum molecular simulation:

1. **Molecules and Quantum States** -- Atoms, bonds, molecular orbitals, and the qubit mapping
2. **Building the Hamiltonian** -- Pauli operators, tensor products, and constructing the Hamiltonian matrix
3. **Diagonalization and Energy Levels** -- Eigenvalues, eigenstates, and the potential energy curve
4. **Time Evolution** -- The Schrodinger equation and animated quantum state dynamics

### Molecules

Three molecules are included, all using 4 qubits (STO-3G basis, Jordan-Wigner mapping):

| Molecule | Description | Bond Length |
|----------|-------------|------------|
| H2 | Hydrogen -- the simplest molecule | Adjustable slider (0.5-3.0 A) |
| HeH+ | Helium hydride cation -- heteronuclear | Fixed at 0.772 A |
| LiH | Lithium hydride -- frozen-core active space | Fixed at 1.595 A |

All Pauli Hamiltonian coefficients were computed from first principles using PySCF (see `compute_*.py` scripts).

### Visualizations

Nine interactive visualizations are embedded throughout the course:

- **V1** Molecular Orbital Energy Level Diagram
- **V2** Superposition and Probability Explorer
- **V3** Pauli Matrix Explorer
- **V4** Hamiltonian Builder (animated Pauli decomposition)
- **V5** Full 16x16 Hamiltonian Heatmap
- **V6** Bond Length Explorer (H2, real-time matrix updates)
- **V7** Energy Level Diagram + Eigenstate Table
- **V8** Potential Energy Curve (H2)
- **V9** Time Evolution Playground (animated probability bar chart)

### Additional Features

- Collapsible sidebar navigation with progress tracking
- Global molecule selector affecting all visualizations
- AI chat tutor powered by the Anthropic API (context-aware)
- Practice quizzes at the end of each lesson
- Responsive layout with keyboard navigation

## Tech Stack

- **React** (single-file JSX component)
- **Tailwind CSS** (core utility classes only, no compiler)
- **All computation client-side** in JavaScript (complex arithmetic, Kronecker products, Jacobi eigenvalue algorithm, time evolution)
- **No external physics/chemistry libraries**

## Qubit Ordering Convention

This project uses the Qiskit convention: q0 is the least significant bit (rightmost in bitstrings). A Pauli string "ABCD" means A on q3, B on q2, C on q1, D on q0. The ground state bitstring |1100> means q3=1, q2=1 (bonding orbital occupied), q1=0, q0=0 (antibonding orbital empty).

## Generating Molecule Data

The Pauli Hamiltonian coefficients are precomputed using PySCF:

```bash
pip install pyscf
python compute_h2_hamiltonian.py       # H2 at 13 bond lengths
python compute_hamiltonians.py         # HeH+ and LiH
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
