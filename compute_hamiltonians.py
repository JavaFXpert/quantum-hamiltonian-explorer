#!/usr/bin/env python3
"""
Compute Jordan-Wigner Pauli Hamiltonians for HeH+ and LiH using PySCF.

Qubit ordering convention (Qiskit): q0 is LEAST significant bit (rightmost).
Pauli string "ABCD" means A on q3, B on q2, C on q1, D on q0.

Spin-orbital to qubit mapping:
  PySCF MOs: [sigma (bonding), sigma* (antibonding)]
  Spin-orbitals: [sigma_up, sigma_down, sigma*_up, sigma*_down]
  JW kron ordering: kron position 0 (leftmost) = sigma_up, ... position 3 = sigma*_down
  Qiskit mapping: kron pos 0 = q3, pos 1 = q2, pos 2 = q1, pos 3 = q0

  Result: q3 = sigma_up (bonding), q2 = sigma_down (bonding)
          q1 = sigma*_up (antibonding), q0 = sigma*_down (antibonding)

  Equivalently (since Hamiltonian is invariant under spin relabeling):
          [sigma*_up, sigma*_down, sigma_up, sigma_down] -> [q0, q1, q2, q3]

  Ground state: bonding occupied -> q3=1, q2=1, q1=0, q0=0 -> |1100>

This matches the user requirement:
  q2, q3 -> bonding orbital spin-up and spin-down (OCCUPIED in ground state)
  q0, q1 -> antibonding orbital spin-up and spin-down (UNOCCUPIED in ground state)
"""

import numpy as np
from pyscf import gto, scf, fci, mcscf, ao2mo
import json

# =============================================================================
# Matrix-based JW transformation
# =============================================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

SIGMA_PLUS = np.array([[0, 0], [1, 0]], dtype=complex)   # a†: |0> -> |1>
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)  # a:  |1> -> |0>

PAULI_LABELS = ['I', 'X', 'Y', 'Z']
PAULI_MATS = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}


def jw_creation(n_qubits, j):
    """JW creation operator a†_j as a 2^n x 2^n matrix."""
    ops = []
    for k in range(n_qubits):
        if k < j:
            ops.append(Z)
        elif k == j:
            ops.append(SIGMA_PLUS)
        else:
            ops.append(I2)
    result = ops[0]
    for m in ops[1:]:
        result = np.kron(result, m)
    return result


def jw_annihilation(n_qubits, j):
    """JW annihilation operator a_j as a 2^n x 2^n matrix."""
    ops = []
    for k in range(n_qubits):
        if k < j:
            ops.append(Z)
        elif k == j:
            ops.append(SIGMA_MINUS)
        else:
            ops.append(I2)
    result = ops[0]
    for m in ops[1:]:
        result = np.kron(result, m)
    return result


def build_jw_hamiltonian_matrix(h1_spin, h2_spin, n_qubits, constant=0.0):
    """
    Build the JW Hamiltonian as a full matrix.

    h1_spin[p,q]: one-electron integrals in spin-orbital basis
    h2_spin[p,q,r,s]: two-electron integrals in PHYSICIST notation <pq|rs>
    H = constant + sum_pq h1[p,q] a†_p a_q + (1/2) sum_pqrs h2[p,q,r,s] a†_p a†_q a_s a_r
    """
    dim = 2 ** n_qubits
    H = constant * np.eye(dim, dtype=complex)

    # Cache operators
    adag = [jw_creation(n_qubits, j) for j in range(n_qubits)]
    a = [jw_annihilation(n_qubits, j) for j in range(n_qubits)]

    # One-body
    for p in range(n_qubits):
        for q in range(n_qubits):
            if abs(h1_spin[p, q]) > 1e-15:
                H += h1_spin[p, q] * (adag[p] @ a[q])

    # Two-body
    for p in range(n_qubits):
        for q in range(n_qubits):
            if p == q:
                continue
            for r in range(n_qubits):
                for s in range(n_qubits):
                    if r == s:
                        continue
                    if abs(h2_spin[p, q, r, s]) < 1e-15:
                        continue
                    H += 0.5 * h2_spin[p, q, r, s] * (adag[p] @ adag[q] @ a[s] @ a[r])

    return H


def decompose_to_pauli(H_matrix, n_qubits):
    """
    Decompose a Hamiltonian matrix into Pauli string coefficients.
    H = sum_k c_k P_k, where c_k = Tr(P_k H) / 2^n

    Returns dict: {pauli_string: coefficient}
    """
    dim = 2 ** n_qubits
    terms = {}

    for indices in np.ndindex(*([4] * n_qubits)):
        label = ''.join(PAULI_LABELS[i] for i in indices)

        # Build Pauli matrix
        P = PAULI_MATS[PAULI_LABELS[indices[0]]]
        for k in range(1, n_qubits):
            P = np.kron(P, PAULI_MATS[PAULI_LABELS[indices[k]]])

        coeff = np.trace(P @ H_matrix).real / dim
        if abs(coeff) > 1e-10:
            terms[label] = coeff

    return terms


def spatial_to_spin_integrals(h1_mo, h2_mo_chemist, n_spatial):
    """
    Convert spatial MO integrals to spin-orbital integrals.

    h2_mo_chemist is in chemist notation (ij|kl).
    Returns h1_spin, h2_spin in PHYSICIST notation <pq|rs>.

    Spin-orbital ordering: [0_up, 0_down, 1_up, 1_down, ...]
    """
    n_spin = 2 * n_spatial

    h1_spin = np.zeros((n_spin, n_spin))
    for p in range(n_spin):
        for q in range(n_spin):
            if p % 2 == q % 2:
                h1_spin[p, q] = h1_mo[p // 2, q // 2]

    # Physicist: <pq|rs> = chemist (pr|qs)
    h2_spin = np.zeros((n_spin, n_spin, n_spin, n_spin))
    for p in range(n_spin):
        for q in range(n_spin):
            for r in range(n_spin):
                for s in range(n_spin):
                    if p % 2 == r % 2 and q % 2 == s % 2:
                        h2_spin[p, q, r, s] = h2_mo_chemist[p // 2, r // 2, q // 2, s // 2]

    return h1_spin, h2_spin


# =============================================================================
# HeH+
# =============================================================================

def compute_heh_plus():
    print("=" * 70)
    print("HeH+ (STO-3G, R = 0.772 A)")
    print("=" * 70)

    mol = gto.M(
        atom='He 0 0 0; H 0 0 0.772',
        basis='sto-3g',
        charge=1,
        spin=0,
        unit='Angstrom'
    )

    mf = scf.RHF(mol)
    mf.kernel()
    print(f"RHF energy: {mf.e_tot:.10f}")

    cisolver = fci.FCI(mf)
    e_fci, ci_vec = cisolver.kernel()
    print(f"FCI energy: {e_fci:.10f}")

    nuc_e = mol.energy_nuc()
    print(f"Nuclear repulsion: {nuc_e:.10f}")

    n_spatial = mf.mo_coeff.shape[1]
    print(f"Spatial orbitals: {n_spatial}")

    # MO integrals
    h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2_mo = ao2mo.restore(1, ao2mo.full(mol, mf.mo_coeff), n_spatial)

    print(f"\nh1_mo:\n{h1_mo}")

    # Spin-orbital integrals
    h1_spin, h2_spin = spatial_to_spin_integrals(h1_mo, h2_mo, n_spatial)

    # Build JW Hamiltonian matrix
    n_qubits = 4
    H_mat = build_jw_hamiltonian_matrix(h1_spin, h2_spin, n_qubits, nuc_e)

    assert np.allclose(H_mat, H_mat.conj().T), "Not Hermitian!"

    # Decompose into Pauli strings
    pauli_terms = decompose_to_pauli(H_mat, n_qubits)

    # Verify by reconstruction
    dim = 2 ** n_qubits
    H_check = np.zeros((dim, dim), dtype=complex)
    for label, coeff in pauli_terms.items():
        P = PAULI_MATS[label[0]]
        for k in range(1, n_qubits):
            P = np.kron(P, PAULI_MATS[label[k]])
        H_check += coeff * P
    recon_err = np.max(np.abs(H_mat - H_check))
    print(f"Pauli reconstruction error: {recon_err:.2e}")

    # Eigenvalues
    eigenvalues = np.sort(np.linalg.eigvalsh(H_mat.real))

    # 2-electron sector
    states_2e = [i for i in range(dim) if bin(i).count('1') == 2]
    P_proj = np.zeros((dim, len(states_2e)))
    for j, s in enumerate(states_2e):
        P_proj[s, j] = 1.0
    H_2e = P_proj.T @ H_mat.real @ P_proj
    eigs_2e = np.sort(np.linalg.eigvalsh(H_2e))

    print(f"\n2-electron sector eigenvalues: {eigs_2e}")
    print(f"2e ground state energy: {eigs_2e[0]:.10f}")
    print(f"FCI energy:             {e_fci:.10f}")
    print(f"Match: {abs(eigs_2e[0] - e_fci) < 1e-8}")

    # Ground state verification
    H_2e_eig = np.linalg.eigh(H_2e)
    gs_idx = np.argmin(H_2e_eig[0])
    gs_vec = H_2e_eig[1][:, gs_idx]
    gs_full = np.zeros(dim)
    for j, s in enumerate(states_2e):
        gs_full[s] = gs_vec[j]

    print(f"\n2e ground state vector (dominant components):")
    for i in range(dim):
        if abs(gs_full[i]) > 0.001:
            bits = format(i, '04b')
            print(f"  |{bits}> : {gs_full[i]:.6f}")

    # All eigenvalues
    print(f"\nAll 16 eigenvalues (sorted):")
    for i, e in enumerate(eigenvalues):
        print(f"  {i:2d}: {e:.10f}")

    # Pauli terms
    print(f"\n{len(pauli_terms)} Pauli terms:")
    for ops, c in sorted(pauli_terms.items(), key=lambda x: -abs(x[1])):
        print(f"  {c:+.12f} * {ops}")

    return pauli_terms, eigenvalues.tolist(), e_fci, nuc_e


# =============================================================================
# LiH
# =============================================================================

def compute_lih():
    print("\n" + "=" * 70)
    print("LiH (STO-3G, R = 1.595 A, CAS(2,2))")
    print("=" * 70)

    mol = gto.M(
        atom='Li 0 0 0; H 0 0 1.595',
        basis='sto-3g',
        charge=0,
        spin=0,
        unit='Angstrom'
    )

    mf = scf.RHF(mol)
    mf.kernel()
    print(f"RHF energy: {mf.e_tot:.10f}")

    nuc_e = mol.energy_nuc()
    print(f"Nuclear repulsion: {nuc_e:.10f}")

    # CASCI(2,2)
    mycas = mcscf.CASCI(mf, 2, 2)
    e_cas = mycas.kernel()[0]
    print(f"CASCI(2,2) energy: {e_cas:.10f}")

    ncore = mycas.ncore
    ncas = mycas.ncas
    nelecas = mycas.nelecas
    print(f"Core orbitals: {ncore}, Active orbitals: {ncas}, Active electrons: {nelecas}")

    # Active space integrals
    h1_cas, e_core = mycas.get_h1eff()
    print(f"Core energy: {e_core:.10f}")

    h2_cas = mycas.get_h2eff()
    if h2_cas.ndim != 4:
        h2_cas = ao2mo.restore(1, h2_cas, ncas)

    print(f"\nh1_cas:\n{h1_cas}")

    # Spin-orbital integrals
    h1_spin, h2_spin = spatial_to_spin_integrals(h1_cas, h2_cas, ncas)

    # Build JW Hamiltonian
    n_qubits = 4
    H_mat = build_jw_hamiltonian_matrix(h1_spin, h2_spin, n_qubits, e_core)

    assert np.allclose(H_mat, H_mat.conj().T), "Not Hermitian!"

    # Decompose
    pauli_terms = decompose_to_pauli(H_mat, n_qubits)

    # Verify reconstruction
    dim = 2 ** n_qubits
    H_check = np.zeros((dim, dim), dtype=complex)
    for label, coeff in pauli_terms.items():
        P = PAULI_MATS[label[0]]
        for k in range(1, n_qubits):
            P = np.kron(P, PAULI_MATS[label[k]])
        H_check += coeff * P
    recon_err = np.max(np.abs(H_mat - H_check))
    print(f"Pauli reconstruction error: {recon_err:.2e}")

    # Eigenvalues
    eigenvalues = np.sort(np.linalg.eigvalsh(H_mat.real))

    # 2-electron sector
    states_2e = [i for i in range(dim) if bin(i).count('1') == 2]
    P_proj = np.zeros((dim, len(states_2e)))
    for j, s in enumerate(states_2e):
        P_proj[s, j] = 1.0
    H_2e = P_proj.T @ H_mat.real @ P_proj
    eigs_2e = np.sort(np.linalg.eigvalsh(H_2e))

    print(f"\n2-electron sector eigenvalues: {eigs_2e}")
    print(f"2e ground state: {eigs_2e[0]:.10f}")
    print(f"CASCI energy:    {e_cas:.10f}")
    print(f"Match: {abs(eigs_2e[0] - e_cas) < 1e-8}")

    # Ground state verification
    H_2e_eig = np.linalg.eigh(H_2e)
    gs_idx = np.argmin(H_2e_eig[0])
    gs_vec = H_2e_eig[1][:, gs_idx]
    gs_full = np.zeros(dim)
    for j, s in enumerate(states_2e):
        gs_full[s] = gs_vec[j]

    print(f"\n2e ground state vector (dominant components):")
    for i in range(dim):
        if abs(gs_full[i]) > 0.001:
            bits = format(i, '04b')
            print(f"  |{bits}> : {gs_full[i]:.6f}")

    print(f"\nAll 16 eigenvalues (sorted):")
    for i, e in enumerate(eigenvalues):
        print(f"  {i:2d}: {e:.10f}")

    print(f"\n{len(pauli_terms)} Pauli terms:")
    for ops, c in sorted(pauli_terms.items(), key=lambda x: -abs(x[1])):
        print(f"  {c:+.12f} * {ops}")

    return pauli_terms, eigenvalues.tolist(), e_cas, e_core


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    heh_terms, heh_eigs, heh_fci, heh_nuc = compute_heh_plus()
    lih_terms, lih_eigs, lih_cas, lih_core = compute_lih()

    print("\n" + "=" * 70)
    print("JSON OUTPUT")
    print("=" * 70)

    def format_terms(terms_dict):
        return [{"coeff": round(c, 12), "ops": ops} for ops, c in
                sorted(terms_dict.items(), key=lambda x: -abs(x[1]))]

    heh_json = {
        "molecule": "HeH+",
        "basis": "STO-3G",
        "bond_length_angstrom": 0.772,
        "n_qubits": 4,
        "n_electrons": 2,
        "nuclear_repulsion": round(heh_nuc, 12),
        "fci_energy": round(heh_fci, 12),
        "qubit_mapping": "Jordan-Wigner",
        "spin_orbital_to_qubit": "[sigma*_up, sigma*_down, sigma_up, sigma_down] -> [q0, q1, q2, q3]",
        "pauli_string_convention": "ABCD = A@q3 B@q2 C@q1 D@q0",
        "ground_state_bitstring": "|1100>",
        "eigenvalues": [round(e, 12) for e in heh_eigs],
        "pauli_terms": format_terms(heh_terms)
    }

    lih_json = {
        "molecule": "LiH",
        "basis": "STO-3G",
        "bond_length_angstrom": 1.595,
        "active_space": "CAS(2,2)",
        "n_qubits": 4,
        "n_active_electrons": 2,
        "core_energy": round(lih_core, 12),
        "casci_energy": round(lih_cas, 12),
        "qubit_mapping": "Jordan-Wigner",
        "spin_orbital_to_qubit": "[sigma*_up, sigma*_down, sigma_up, sigma_down] -> [q0, q1, q2, q3]",
        "pauli_string_convention": "ABCD = A@q3 B@q2 C@q1 D@q0",
        "ground_state_bitstring": "|1100>",
        "eigenvalues": [round(e, 12) for e in lih_eigs],
        "pauli_terms": format_terms(lih_terms)
    }

    print("\n// HeH+ Hamiltonian")
    print(json.dumps(heh_json, indent=2))

    print("\n// LiH Hamiltonian")
    print(json.dumps(lih_json, indent=2))
