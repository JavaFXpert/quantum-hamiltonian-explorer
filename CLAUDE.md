# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Interactive educational course/playground teaching high school students about quantum molecular simulation. Covers how molecules become quantum Hamiltonians, diagonalization for energy levels, and time evolution of quantum states. Three molecules: H2 (with bond length slider), HeH+, and LiH (frozen core active space).

## Tech Stack & Constraints
- **Single-file React JSX** artifact for Claude desktop/web app
- **Tailwind CSS** core utility classes only (no compiler, no custom config)
- **All computation client-side** in JavaScript -- no external physics/chemistry libraries
- **Anthropic API** for AI chat tutor (model: `claude-sonnet-4-20250514`, no API key needed in artifact context)
- Complex numbers represented as `[real, imaginary]` pairs with inline arithmetic functions

## Architecture
The app follows the **interactive-course-builder** pattern: a single React component containing:
- **Sidebar navigation** (collapsible) with phases/lessons
- **Horizontal section tabs** within each lesson (4-6 sections per lesson)
- **Global state**: `selectedMolecule`, `bondLength`, `currentPhase/Lesson/Section`, `completedSections` (Set), `chatOpen/chatMessages`
- **Global molecule selector** at top of main content area, persists across all phases
- **Floating AI chat panel** (bottom-right, context-aware with current phase/lesson/section)

### Course Structure
4 phases (blue/purple/green/orange), 6 lessons, ~24 sections, 9 interactive visualizations (V1-V9). Full specification in `SPEC.md`.

### Molecule Data
All molecules use 4 qubits (4 spin-orbitals), Jordan-Wigner mapping, STO-3G basis:
- **H2**: R=0.735A equilibrium, bond length slider 0.5-3.0A (needs Pauli terms at ~10+ bond lengths, linear interpolation between them)
- **HeH+**: R=0.772A fixed geometry
- **LiH**: R=1.595A fixed geometry, (2e,2o) active space with frozen core

### Math Pipeline
1. Pauli terms (coefficients + operator strings like "XXYY") -> Kronecker product -> 16x16 Hamiltonian matrix
2. Diagonalization (precomputed eigenvalues/eigenvectors recommended for accuracy)
3. Time evolution: U(t)=exp(-iHt) via eigenbasis decomposition, animated with requestAnimationFrame

## Key Implementation Rules
- **Qiskit qubit ordering**: q0 is the least significant bit (rightmost in bitstrings). A Pauli string "XYZZ" means X on q3, Y on q2, Z on q1, Z on q0, and the bitstring |1100⟩ means q3=1, q2=1, q1=0, q0=0. Tensor products are ordered accordingly: q3 ⊗ q2 ⊗ q1 ⊗ q0. This convention must be consistent across all bitstring labels, Pauli string parsing, Kronecker products, MO diagram qubit labels, and eigenstate displays.
- Never use em-dashes anywhere in text content
- Use Dirac notation consistently throughout all text content (|0⟩, |1⟩, |ψ⟩)
- Every visualization must include a learner instruction panel with three sections: "What this shows", "How to use", "What to observe" on a colored background (bg-{phase-color}-100)
- Hamiltonian coefficients must be accurate published values for these well-studied benchmark systems
- Section completion tracked by navigation away (simple heuristic); show checkmarks on tabs and phase completion % in sidebar
- Visualizations must respond to the global molecule selector
- The bond length slider (H2 only) updates Hamiltonian, eigenvalues, and all dependent visualizations in real time
- Matrix math should work for any power-of-2 qubit count (not hardcoded to 16x16)
- Adding a new molecule should only require adding an entry to the MOLECULES object; all visualizations and math should derive from that data automatically
- AI tutor system prompt must include current phase, lesson, section title, section content, and selected molecule name for context-aware responses

## Build Approach
Per SPEC.md: build incrementally -- course framework/navigation first, then visualizations one at a time, chat tutor last. Read SPEC.md fully before writing any code.
