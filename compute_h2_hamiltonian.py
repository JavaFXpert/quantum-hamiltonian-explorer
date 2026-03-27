"""
Compute exact Jordan-Wigner Pauli Hamiltonian coefficients for H2/STO-3G
at multiple bond lengths.

Qubit ordering (Qiskit convention, q0 = least significant = rightmost):
  Spin-orbital ordering: [sigma*_up, sigma*_dn, sigma_up, sigma_dn] -> [q0, q1, q2, q3]
  Pauli string "ABCD" means A on q3, B on q2, C on q1, D on q0.
  Ground state bitstring: |1100> (q3=1, q2=1, q1=0, q0=0)
"""

import numpy as np
from pyscf import gto, scf
import json
import itertools

def compute_h2_integrals(bond_length):
    """Compute molecular integrals for H2 at given bond length (Angstroms)."""
    mol = gto.M(
        atom=f'H 0 0 0; H 0 0 {bond_length}',
        basis='sto-3g',
        symmetry=False,
        unit='Angstrom'
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    e_hf = mf.kernel()

    # Nuclear repulsion energy
    e_nuc = mol.energy_nuc()

    # MO coefficients
    mo_coeff = mf.mo_coeff

    # One-electron integrals in MO basis
    h1_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    h1_mo = mo_coeff.T @ h1_ao @ mo_coeff

    # Two-electron integrals in MO basis (chemist notation: (pq|rs))
    from pyscf import ao2mo
    h2_mo = ao2mo.kernel(mol, mo_coeff)
    # h2_mo is in compressed form, restore to full 4-index
    n_mo = mo_coeff.shape[1]
    h2_mo = ao2mo.restore(1, h2_mo, n_mo)  # (pq|rs) chemist notation

    return h1_mo, h2_mo, e_nuc, e_hf


def build_fermionic_hamiltonian(h1_mo, h2_mo, e_nuc):
    """
    Build the fermionic Hamiltonian in spin-orbital basis.

    PySCF gives spatial MO integrals for 2 spatial orbitals: sigma (bonding), sigma* (antibonding).
    We need spin-orbital integrals for 4 spin-orbitals.

    We want spin-orbital ordering: [sigma*_up, sigma*_dn, sigma_up, sigma_dn]
    mapping to qubits [q0, q1, q2, q3].

    PySCF MO ordering: 0 = sigma (bonding), 1 = sigma* (antibonding)

    Our desired spin-orbital ordering:
      spin-orb 0 (q0) = sigma*_up  -> spatial MO 1, alpha
      spin-orb 1 (q1) = sigma*_dn  -> spatial MO 1, beta
      spin-orb 2 (q2) = sigma_up   -> spatial MO 0, alpha
      spin-orb 3 (q3) = sigma_dn   -> spatial MO 0, beta
    """
    n_spatial = h1_mo.shape[0]  # 2 for H2/STO-3G
    n_spinorb = 2 * n_spatial   # 4

    # Map from our spin-orbital index to (spatial_mo, spin)
    # spin: 0=alpha, 1=beta
    spinorb_map = [
        (1, 0),  # spin-orb 0: sigma*_up
        (1, 1),  # spin-orb 1: sigma*_dn
        (0, 0),  # spin-orb 2: sigma_up
        (0, 1),  # spin-orb 3: sigma_dn
    ]

    # Build one-electron spin-orbital integrals
    h1_so = np.zeros((n_spinorb, n_spinorb))
    for p in range(n_spinorb):
        for q in range(n_spinorb):
            p_spat, p_spin = spinorb_map[p]
            q_spat, q_spin = spinorb_map[q]
            if p_spin == q_spin:
                h1_so[p, q] = h1_mo[p_spat, q_spat]

    # Build two-electron spin-orbital integrals
    # PySCF h2_mo is in chemist notation: (pq|rs) = <pr|qs> (physicist)
    # We need physicist notation: <pq|rs> = (pr|qs) in chemist
    # Or equivalently: <pq||rs> = <pq|rs> - <pq|sr>
    #                            = (pr|qs) - (ps|qr) in chemist notation

    # h2_phys[p,q,r,s] = <pq|rs> = (pr|qs) in chemist = h2_mo[p,r,q,s]
    h2_so = np.zeros((n_spinorb, n_spinorb, n_spinorb, n_spinorb))
    for p in range(n_spinorb):
        for q in range(n_spinorb):
            for r in range(n_spinorb):
                for s in range(n_spinorb):
                    p_spat, p_spin = spinorb_map[p]
                    q_spat, q_spin = spinorb_map[q]
                    r_spat, r_spin = spinorb_map[r]
                    s_spat, s_spin = spinorb_map[s]
                    # <pq|rs> in physicist notation = (pr|qs) in chemist notation
                    # requires spin conservation: p_spin==r_spin and q_spin==s_spin
                    if p_spin == r_spin and q_spin == s_spin:
                        h2_so[p, q, r, s] = h2_mo[p_spat, r_spat, q_spat, s_spat]

    return h1_so, h2_so, e_nuc


def jordan_wigner_transform(h1_so, h2_so, e_nuc):
    """
    Perform Jordan-Wigner transformation to get Pauli Hamiltonian.

    H = sum_{pq} h1[p,q] a†_p a_q + (1/2) sum_{pqrs} h2[p,q,r,s] a†_p a†_q a_s a_r + E_nuc

    where h2[p,q,r,s] = <pq|rs> in physicist notation.

    Returns dict mapping Pauli string -> coefficient.
    """
    n = h1_so.shape[0]  # number of spin-orbitals (4)

    # We'll accumulate Pauli terms as a dictionary: pauli_string -> coefficient
    pauli_dict = {}

    def add_term(pauli_str, coeff):
        if pauli_str in pauli_dict:
            pauli_dict[pauli_str] += coeff
        else:
            pauli_dict[pauli_str] = coeff

    # Identity from nuclear repulsion
    add_term('IIII', e_nuc)

    # One-body terms: h1[p,q] a†_p a_q
    for p in range(n):
        for q in range(n):
            if abs(h1_so[p, q]) < 1e-15:
                continue
            jw_one_body(p, q, h1_so[p, q], n, add_term)

    # Two-body terms: (1/2) h2[p,q,r,s] a†_p a†_q a_s a_r
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    if abs(h2_so[p, q, r, s]) < 1e-15:
                        continue
                    jw_two_body(p, q, r, s, 0.5 * h2_so[p, q, r, s], n, add_term)

    # Clean up near-zero terms
    result = {}
    for k, v in pauli_dict.items():
        if abs(v) > 1e-10:
            result[k] = v

    return result


def jw_one_body(p, q, coeff, n, add_term):
    """
    JW transform of coeff * a†_p a_q for n qubits.

    a†_p a_p = (I - Z_p) / 2

    For p != q (p < q):
    a†_p a_q = (1/2)(X_p Zstring X_q + Y_p Zstring Y_q)
             + (i/2)(Y_p Zstring X_q - X_p Zstring Y_q)  [but for real coeff this is hermitian]

    Actually for a†_p a_q + h.c. with real h1[p,q] = h1[q,p]:
    We handle each (p,q) pair individually since we're summing over all p,q.

    JW representation:
    a_p = (1/2)(X_p + iY_p) * Z_{p-1} * Z_{p-2} * ... * Z_0
    a†_p = (1/2)(X_p - iY_p) * Z_{p-1} * Z_{p-2} * ... * Z_0

    Note: In our convention, qubit index = spin-orbital index.
    The Z-string goes from qubit 0 to qubit p-1.
    """
    if p == q:
        # Number operator: a†_p a_p = (I - Z_p) / 2
        pauli_I = ['I'] * n
        pauli_Z = ['I'] * n
        pauli_Z[p] = 'Z'
        add_term(''.join(reversed(pauli_I)), coeff / 2.0)  # reversed for Qiskit convention
        add_term(''.join(reversed(pauli_Z)), -coeff / 2.0)
    else:
        # a†_p a_q for p != q
        # The Z-string is on all qubits strictly between min(p,q) and max(p,q)
        lo, hi = min(p, q), max(p, q)

        # a†_p a_q = (1/4) * product_of_operators
        # Using the JW transformation explicitly:
        # a†_p = (X_p - iY_p)/2 * Z_{0}...Z_{p-1}
        # a_q  = (X_q + iY_q)/2 * Z_{0}...Z_{q-1}
        #
        # a†_p a_q: the Z strings from 0..p-1 and 0..q-1 cancel below min(p,q)
        # leaving Z's on qubits from lo to hi-1 (exclusive of lo and hi? No...)
        #
        # Let me be precise. For p < q:
        # a†_p a_q = (X_p - iY_p)/2 * Z_0...Z_{p-1} * (X_q + iY_q)/2 * Z_0...Z_{q-1}
        #          = (1/4)(X_p - iY_p)(X_q + iY_q) * Z_0^2...Z_{p-1}^2 * Z_p...Z_{q-1}
        # Since Z_k^2 = I, the Z's from 0 to p-1 cancel.
        # Remaining: Z_p * Z_{p+1} * ... * Z_{q-1} on the middle qubits
        # But wait, Z_p is at the position of a†_p where we also have (X_p - iY_p)
        #
        # Actually the formula is:
        # a†_p = (1/2)(X_p - iY_p) * prod_{k<p} Z_k
        # a_q  = (1/2)(X_q + iY_q) * prod_{k<q} Z_k
        #
        # a†_p a_q = (1/4)(X_p - iY_p)(X_q + iY_q) * prod_{k<p} Z_k * prod_{k<q} Z_k
        #
        # The Z products: for k < min(p,q), Z_k appears twice -> I
        # For min(p,q) <= k < max(p,q), Z_k appears once
        # But if p < q: Z_k for p <= k < q
        # If p > q: Z_k for q <= k < p
        #
        # However, at position p we have X_p or Y_p (from the creation op),
        # and at position q we have X_q or Y_q (from the annihilation op).
        # The Z_k in the middle (strictly between p and q) remain as Z.
        # At positions p and q, the X/Y operators are what act.
        #
        # But there's a subtlety: if p < q, the Z string includes Z_p.
        # X_p * Z_p = X_p * Z_p = -iY_p... this gets complicated.
        #
        # Let me use the standard result directly.

        # Standard JW result for a†_p a_q (p != q):
        # Let lo = min(p,q), hi = max(p,q)
        # a†_p a_q + a†_q a_p = (1/2)[X_p (prod_{k=lo+1}^{hi-1} Z_k) X_q
        #                              + Y_p (prod_{k=lo+1}^{hi-1} Z_k) Y_q]
        #
        # The sign depends on ordering. For p < q:
        # a†_p a_q = (1/2)[X_p Z_{p+1}...Z_{q-1} X_q + Y_p Z_{p+1}...Z_{q-1} Y_q]
        #          + (i/2)[Y_p Z_{p+1}...Z_{q-1} X_q - X_p Z_{p+1}...Z_{q-1} Y_q]
        #
        # For real coefficients, the imaginary part cancels when we sum h[p,q] and h[q,p].
        # But since we're iterating over all p,q, let me handle each individually.

        # For a†_p a_q with p < q:
        # = (1/4)(X_p X_q + Y_p Y_q + iY_p X_q - iX_p Y_q) * Z_{p+1}...Z_{q-1}

        # For a†_p a_q with p > q:
        # = (1/4)(X_p X_q + Y_p Y_q - iY_p X_q + iX_p Y_q) * Z_{q+1}...Z_{p-1}
        # Note the sign flip on the imaginary terms.

        # Build the Z string between lo+1 and hi-1
        def make_pauli(op_lo, op_hi, lo, hi, n):
            """Create Pauli string with op_lo at position lo, op_hi at position hi, Z in between."""
            p_list = ['I'] * n
            p_list[lo] = op_lo
            p_list[hi] = op_hi
            for k in range(lo + 1, hi):
                p_list[k] = 'Z'
            # Reverse for Qiskit convention (q0 rightmost)
            return ''.join(reversed(p_list))

        if p < q:
            xx_str = make_pauli('X', 'X', p, q, n)
            yy_str = make_pauli('Y', 'Y', p, q, n)
            yx_str = make_pauli('Y', 'X', p, q, n)
            xy_str = make_pauli('X', 'Y', p, q, n)

            add_term(xx_str, coeff / 4.0)
            add_term(yy_str, coeff / 4.0)
            add_term(yx_str, 1j * coeff / 4.0)
            add_term(xy_str, -1j * coeff / 4.0)
        else:  # p > q
            xx_str = make_pauli('X', 'X', q, p, n)
            yy_str = make_pauli('Y', 'Y', q, p, n)
            yx_str = make_pauli('X', 'Y', q, p, n)  # note: lo=q gets X, hi=p gets Y
            xy_str = make_pauli('Y', 'X', q, p, n)  # lo=q gets Y, hi=p gets X

            add_term(xx_str, coeff / 4.0)
            add_term(yy_str, coeff / 4.0)
            add_term(yx_str, 1j * coeff / 4.0)  # +i for p > q
            add_term(xy_str, -1j * coeff / 4.0)


def jw_two_body(p, q, r, s, coeff, n, add_term):
    """
    JW transform of coeff * a†_p a†_q a_s a_r.

    We decompose this using:
    a†_p a†_q a_s a_r = a†_p a_r * a†_q a_s - delta_{qr} * a†_p a_s

    (Using the anticommutation relation: a†_q a_s = -a_s a†_q + delta_{qs})
    Wait, let me be more careful.

    a†_p a†_q a_s a_r
    We can normal order this using Wick's theorem, but it's simpler to
    express in terms of one-body operators:

    a†_p a†_q a_s a_r = a†_p a_r a†_q a_s - delta_{qr} a†_p a_s

    Wait: a†_q a_s a_r = a†_q (delta_{sr} - a_r a_s†)... this is getting complicated.
    Let me use the direct approach instead.

    Actually, the simplest approach: decompose the two-body operator into products of
    one-body JW operators. But that's complex because of the Z-strings.

    Better approach: use the known result that for the molecular Hamiltonian,
    the two-body term in JW produces specific Pauli terms.

    Let me use the direct numerical approach: build the full 16x16 Hamiltonian matrix
    and then decompose into Pauli basis.
    """
    # This function is called but we'll use the matrix decomposition approach instead
    pass


def build_hamiltonian_matrix(h1_so, h2_so, e_nuc):
    """
    Build the full Hamiltonian matrix in the computational basis.
    For 4 spin-orbitals, this is a 16x16 matrix.

    State |b3 b2 b1 b0> corresponds to integer b3*8 + b2*4 + b1*2 + b0
    where bi=1 means spin-orbital i is occupied.

    Qubit ordering: q0 (rightmost) = spin-orbital 0 = sigma*_up
    """
    n = h1_so.shape[0]  # 4
    dim = 2**n  # 16

    H = np.zeros((dim, dim))

    # Add nuclear repulsion to diagonal
    for i in range(dim):
        H[i, i] += e_nuc

    # Convert state index to occupation list
    def occ_list(state, n):
        """Return list of occupied orbitals for given state integer."""
        return [j for j in range(n) if (state >> j) & 1]

    def sign_create(state, p):
        """Sign and new state for a†_p |state>. Returns (sign, new_state) or (0, 0) if orbital occupied."""
        if (state >> p) & 1:
            return 0, 0  # orbital already occupied
        # Count number of occupied orbitals below p
        count = bin(state & ((1 << p) - 1)).count('1')
        new_state = state | (1 << p)
        return (-1)**count, new_state

    def sign_annihilate(state, p):
        """Sign and new state for a_p |state>. Returns (sign, new_state) or (0, 0) if orbital empty."""
        if not ((state >> p) & 1):
            return 0, 0  # orbital empty
        count = bin(state & ((1 << p) - 1)).count('1')
        new_state = state ^ (1 << p)
        return (-1)**count, new_state

    # One-body: h1[p,q] a†_p a_q
    for p in range(n):
        for q in range(n):
            if abs(h1_so[p, q]) < 1e-15:
                continue
            for state in range(dim):
                # Apply a_q then a†_p
                s1, new1 = sign_annihilate(state, q)
                if s1 == 0:
                    continue
                s2, new2 = sign_create(new1, p)
                if s2 == 0:
                    continue
                H[new2, state] += h1_so[p, q] * s1 * s2

    # Two-body: (1/2) h2[p,q,r,s] a†_p a†_q a_s a_r
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    if abs(h2_so[p, q, r, s]) < 1e-15:
                        continue
                    val = 0.5 * h2_so[p, q, r, s]
                    for state in range(dim):
                        # Apply a_r, then a_s, then a†_q, then a†_p
                        s1, st1 = sign_annihilate(state, r)
                        if s1 == 0:
                            continue
                        s2, st2 = sign_annihilate(st1, s)
                        if s2 == 0:
                            continue
                        s3, st3 = sign_create(st2, q)
                        if s3 == 0:
                            continue
                        s4, st4 = sign_create(st3, p)
                        if s4 == 0:
                            continue
                        H[st4, state] += val * s1 * s2 * s3 * s4

    return H


def decompose_to_pauli(H):
    """
    Decompose a 2^n x 2^n Hermitian matrix into Pauli basis.
    H = sum_i c_i P_i where P_i are n-qubit Pauli operators.
    c_i = Tr(P_i H) / 2^n

    For 4 qubits, we have 256 Pauli strings.

    Returns dict: pauli_string -> coefficient
    """
    n = 4
    dim = 2**n

    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    pauli_labels = ['I', 'X', 'Y', 'Z']

    result = {}

    for i3 in range(4):
        for i2 in range(4):
            for i1 in range(4):
                for i0 in range(4):
                    # Pauli string: label for q3, q2, q1, q0
                    label = pauli_labels[i3] + pauli_labels[i2] + pauli_labels[i1] + pauli_labels[i0]

                    # Build the tensor product
                    # In our matrix representation, the state |b3 b2 b1 b0> has index b3*8+b2*4+b1*2+b0
                    # The tensor product should be P_q3 ⊗ P_q2 ⊗ P_q1 ⊗ P_q0
                    # With q3 as most significant bit
                    P = np.kron(np.kron(np.kron(paulis[pauli_labels[i3]],
                                                 paulis[pauli_labels[i2]]),
                                        paulis[pauli_labels[i1]]),
                                paulis[pauli_labels[i0]])

                    coeff = np.trace(P @ H) / dim

                    if abs(coeff.imag) > 1e-10:
                        print(f"WARNING: imaginary coefficient for {label}: {coeff}")

                    if abs(coeff.real) > 1e-10:
                        result[label] = coeff.real

    return result


def verify_ground_state(H, expected_energy):
    """Verify the ground state energy and print the ground state."""
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    gs_energy = eigenvalues[0]
    gs_vector = eigenvectors[:, 0]

    # Find dominant component
    max_idx = np.argmax(np.abs(gs_vector))
    max_bitstring = format(max_idx, '04b')

    print(f"  Ground state energy: {gs_energy:.10f} Ha")
    print(f"  Expected (HF):       {expected_energy:.10f} Ha")
    print(f"  Dominant bitstring:  |{max_bitstring}> (amplitude: {gs_vector[max_idx]:.6f})")
    print(f"  All eigenvalues: {eigenvalues}")

    # Check that |1100> is the dominant state
    state_1100 = 0b1100  # q3=1,q2=1,q1=0,q0=0 = 12
    print(f"  Amplitude of |1100> (index {state_1100}): {gs_vector[state_1100]:.6f}")

    return gs_energy


def main():
    bond_lengths = [0.5, 0.6, 0.7, 0.735, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

    all_results = {}

    for R in bond_lengths:
        print(f"\n{'='*60}")
        print(f"Bond length: {R} Angstroms")
        print(f"{'='*60}")

        # Step 1: Get molecular integrals
        h1_mo, h2_mo, e_nuc, e_hf = compute_h2_integrals(R)

        print(f"  Nuclear repulsion: {e_nuc:.10f}")
        print(f"  HF energy: {e_hf:.10f}")
        print(f"  h1_mo:\n{h1_mo}")

        # Step 2: Build spin-orbital integrals with correct ordering
        h1_so, h2_so, e_nuc_val = build_fermionic_hamiltonian(h1_mo, h2_mo, e_nuc)

        # Step 3: Build full Hamiltonian matrix
        H_matrix = build_hamiltonian_matrix(h1_so, h2_so, e_nuc_val)

        # Step 4: Verify ground state
        gs_energy = verify_ground_state(H_matrix, e_hf)

        # Step 5: Decompose into Pauli basis
        pauli_terms = decompose_to_pauli(H_matrix)

        # Sort by Pauli string
        sorted_terms = sorted(pauli_terms.items())

        print(f"\n  Pauli terms ({len(sorted_terms)} terms):")
        for label, coeff in sorted_terms:
            print(f"    {label}: {coeff:+.10f}")

        # Verify: reconstruct energy from Pauli terms for |1100> state
        # |1100>: q3=1, q2=1, q1=0, q0=0
        # Z eigenvalue: +1 for |0>, -1 for |1>
        # q0=0 -> Z_q0 = +1
        # q1=0 -> Z_q1 = +1
        # q2=1 -> Z_q2 = -1
        # q3=1 -> Z_q3 = -1
        energy_check = 0
        for label, coeff in sorted_terms:
            # For diagonal (Z and I only) terms, compute expectation
            if all(c in 'IZ' for c in label):
                eigenvalue = 1.0
                z_vals = {'I': 1, 'Z': 1}  # default
                # label is "ABCD" for q3,q2,q1,q0
                qvals = [0, 0, 1, 1]  # q0=0, q1=0, q2=1, q3=1
                for i, c in enumerate(label):
                    qi = 3 - i  # label[0] = q3, label[1] = q2, etc.
                    if c == 'Z':
                        eigenvalue *= (1 - 2*qvals[qi])  # +1 for 0, -1 for 1
                energy_check += coeff * eigenvalue

        print(f"\n  Energy from diagonal Pauli terms for |1100>: {energy_check:.10f}")
        print(f"  Full ground state energy: {gs_energy:.10f}")

        # Store results
        terms_list = []
        for label, coeff in sorted_terms:
            terms_list.append({"coeff": round(coeff, 12), "ops": label})

        all_results[str(R)] = terms_list

    # Print final JSON
    print(f"\n\n{'='*60}")
    print("FINAL JSON OUTPUT")
    print(f"{'='*60}")
    print(json.dumps(all_results, indent=2))


if __name__ == '__main__':
    main()
