import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';

// ============================================================================
// PART 1: COMPLEX NUMBER & MATRIX UTILITIES
// ============================================================================

const cAdd = (a, b) => [a[0]+b[0], a[1]+b[1]];
const cSub = (a, b) => [a[0]-b[0], a[1]-b[1]];
const cMul = (a, b) => [a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]];
const cScale = (s, a) => [s*a[0], s*a[1]];
const cConj = (a) => [a[0], -a[1]];
const cAbs2 = (a) => a[0]*a[0]+a[1]*a[1];
const cExp = (a) => { const er=Math.exp(a[0]); return [er*Math.cos(a[1]), er*Math.sin(a[1])]; };
const C0 = [0,0], C1 = [1,0], Ci = [0,1], Cni = [0,-1];

function kronecker(A, B) {
  const ra=A.length, ca=A[0].length, rb=B.length, cb=B[0].length;
  const R=ra*rb, C=ca*cb;
  const result = Array.from({length:R}, ()=>Array(C).fill(C0));
  for(let i=0;i<ra;i++) for(let j=0;j<ca;j++)
    for(let k=0;k<rb;k++) for(let l=0;l<cb;l++)
      result[i*rb+k][j*cb+l] = cMul(A[i][j], B[k][l]);
  return result;
}

function matVecMul(M, v) {
  return M.map(row => row.reduce((s, m, j) => cAdd(s, cMul(m, v[j])), C0));
}

function adjoint(M) {
  const n=M.length, m=M[0].length;
  return Array.from({length:m}, (_,i) => Array.from({length:n}, (_,j) => cConj(M[j][i])));
}

// Pauli matrices as 2x2 complex matrices
const I2 = [[C1,C0],[C0,C1]];
const PX = [[C0,C1],[C1,C0]];
const PY = [[C0,Cni],[Ci,C0]];
const PZ = [[C1,C0],[C0,[-1,0]]];
const PAULI_MAP = {'I':I2,'X':PX,'Y':PY,'Z':PZ};

function pauliStringToMatrix(ops) {
  let result = PAULI_MAP[ops[0]];
  for(let i=1;i<ops.length;i++) result = kronecker(result, PAULI_MAP[ops[i]]);
  return result;
}

function buildHamiltonian(pauliTerms, numQubits) {
  const n = 1 << numQubits;
  const H = Array.from({length:n}, ()=>Array.from({length:n}, ()=>[...C0]));
  for(const {coeff, ops} of pauliTerms) {
    const M = pauliStringToMatrix(ops);
    for(let i=0;i<n;i++) for(let j=0;j<n;j++) H[i][j] = cAdd(H[i][j], cScale(coeff, M[i][j]));
  }
  return H;
}

// Jacobi eigenvalue algorithm for real symmetric matrices
function jacobiEigen(A_in) {
  const n = A_in.length;
  const A = A_in.map(r => r.map(c => c[0])); // take real part
  const V = Array.from({length:n}, (_,i) => {
    const row = new Array(n).fill(0); row[i]=1; return row;
  });
  const maxIter = 100;
  for(let iter=0; iter<maxIter; iter++) {
    let maxVal=0, p=0, q=1;
    for(let i=0;i<n;i++) for(let j=i+1;j<n;j++)
      if(Math.abs(A[i][j])>maxVal) { maxVal=Math.abs(A[i][j]); p=i; q=j; }
    if(maxVal < 1e-12) break;
    const theta = 0.5*Math.atan2(2*A[p][q], A[p][p]-A[q][q]);
    const c=Math.cos(theta), s=Math.sin(theta);
    const Ap = A.map(r=>[...r]);
    for(let i=0;i<n;i++) {
      A[i][p] = c*Ap[i][p] + s*Ap[i][q];
      A[i][q] = -s*Ap[i][p] + c*Ap[i][q];
    }
    const Ap2 = A.map(r=>[...r]);
    for(let j=0;j<n;j++) {
      A[p][j] = c*Ap2[p][j] + s*Ap2[q][j];
      A[q][j] = -s*Ap2[p][j] + c*Ap2[q][j];
    }
    const Vp = V.map(r=>[...r]);
    for(let i=0;i<n;i++) {
      V[i][p] = c*Vp[i][p] + s*Vp[i][q];
      V[i][q] = -s*Vp[i][p] + c*Vp[i][q];
    }
  }
  const eigenvalues = Array.from({length:n}, (_,i) => A[i][i]);
  const indices = eigenvalues.map((_,i)=>i).sort((a,b)=>eigenvalues[a]-eigenvalues[b]);
  const sortedVals = indices.map(i=>eigenvalues[i]);
  const sortedVecs = Array.from({length:n}, (_,i) =>
    Array.from({length:n}, (_,j) => [V[j][indices[i]], 0])
  );
  // sortedVecs[k] = k-th eigenvector as column, represented as array of [re,im]
  // Return as U matrix where U[row][col] = col-th eigenvector's row-th component
  const U = Array.from({length:n}, (_,r) =>
    Array.from({length:n}, (_,c) => [V[r][indices[c]], 0])
  );
  return { eigenvalues: sortedVals, U };
}

function evolveState(U, eigenvalues, psi0, t) {
  const Udagger = adjoint(U);
  const psiEig = matVecMul(Udagger, psi0);
  const psiEigEvolved = psiEig.map((amp, k) =>
    cMul(cExp([0, -eigenvalues[k]*t]), amp)
  );
  return matVecMul(U, psiEigEvolved);
}

// ============================================================================
// PART 2: MOLECULE DATA
// ============================================================================

// H2 compact representation: 8 independent parameters per bond length
// [g0(IIII), g1(IIIZ=IIZI), g2(IZII=ZIII), g3(IIZZ), g4(ZZII), g5(IZIZ=ZIZI), g6(IZZI=ZIIZ), gex(exchange)]
const H2_PARAMS = {
  0.5:  [0.379831351781,-0.369144315244,0.213935310245,0.186209842592,0.179926509764,0.134592403464,0.176809960386,0.042217556922],
  0.6:  [0.132366168028,-0.299205109236,0.194808677350,0.181126506123,0.175334432288,0.128765613415,0.172198274352,0.043432660936],
  0.7:  [-0.042078976478,-0.242742805131,0.177712874651,0.176276408043,0.170597383288,0.122933050562,0.167683194577,0.044750144015],
  0.735: [-0.090578986088,-0.225753492224,0.172183932619,0.174643430683,0.168927538701,0.120912632618,0.166145432564,0.045232799946],
  0.8:  [-0.167333989057,-0.197442936998,0.162516487489,0.171697883923,0.165832537216,0.117203647202,0.163360343091,0.046156695889],
  0.9:  [-0.259054122213,-0.160712491081,0.149074788447,0.167371259483,0.161138163782,0.111627234034,0.159270157474,0.047642923440],
  1.0:  [-0.327608189675,-0.130362920571,0.137165729371,0.163267686736,0.156600624882,0.106229044909,0.155426690780,0.049197645871],
  1.2:  [-0.419602368050,-0.083202862416,0.116986714353,0.155674636296,0.148270607853,0.096043673704,0.148491540860,0.052447867156],
  1.5:  [-0.491785777304,-0.035644816210,0.093456496677,0.145855190301,0.138175845766,0.082537054888,0.139921038903,0.057383984015],
  1.8:  [-0.523676596427,-0.006321810573,0.076087981016,0.137962552239,0.130927261070,0.071308321223,0.133312569593,0.062004248370],
  2.0:  [-0.533936348773,0.006651295688,0.067279304590,0.133666029882,0.127365703107,0.065015695812,0.129800314532,0.064784618720],
  2.5:  [-0.543599092020,0.025513881310,0.052648580736,0.125514947063,0.121420024659,0.052726264397,0.123278775890,0.070552511494],
  3.0:  [-0.545507714555,0.033928391841,0.045163652663,0.119982454376,0.117840552744,0.044071812609,0.118874698424,0.074802885815],
};

function expandH2Params(params) {
  const [g0,g1,g2,g3,g4,g5,g6,gex] = params;
  return [
    {coeff:g0, ops:"IIII"}, {coeff:g1, ops:"IIIZ"}, {coeff:g1, ops:"IIZI"},
    {coeff:g2, ops:"IZII"}, {coeff:g2, ops:"ZIII"}, {coeff:g3, ops:"IIZZ"},
    {coeff:g4, ops:"ZZII"}, {coeff:g5, ops:"IZIZ"}, {coeff:g5, ops:"ZIZI"},
    {coeff:g6, ops:"IZZI"}, {coeff:g6, ops:"ZIIZ"},
    {coeff:-gex, ops:"XXYY"}, {coeff:-gex, ops:"YYXX"},
    {coeff:gex, ops:"XYYX"}, {coeff:gex, ops:"YXXY"},
  ];
}

function interpolateH2(bondLength) {
  const bls = Object.keys(H2_PARAMS).map(Number).sort((a,b)=>a-b);
  if(bondLength <= bls[0]) return expandH2Params(H2_PARAMS[bls[0]]);
  if(bondLength >= bls[bls.length-1]) return expandH2Params(H2_PARAMS[bls[bls.length-1]]);
  let i=0;
  while(i<bls.length-1 && bls[i+1]<bondLength) i++;
  const t = (bondLength - bls[i])/(bls[i+1]-bls[i]);
  const p0 = H2_PARAMS[bls[i]], p1 = H2_PARAMS[bls[i+1]];
  const interp = p0.map((v,j) => v + t*(p1[j]-v));
  return expandH2Params(interp);
}

const HEH_PLUS_TERMS = [
  {coeff:-1.541975952897,ops:"IIII"},{coeff:0.758913613417,ops:"IZII"},{coeff:0.758913613417,ops:"ZIII"},
  {coeff:0.235787271538,ops:"ZZII"},{coeff:0.191400544735,ops:"IIIZ"},{coeff:0.191400544735,ops:"IIZI"},
  {coeff:0.188159055421,ops:"IIZZ"},{coeff:0.165286423411,ops:"IZZI"},{coeff:0.165286423411,ops:"ZIIZ"},
  {coeff:0.128876939852,ops:"IZIZ"},{coeff:0.128876939852,ops:"ZIZI"},
  {coeff:0.052465570376,ops:"IXZX"},{coeff:0.052465570376,ops:"IYZY"},
  {coeff:0.052465570376,ops:"XZXI"},{coeff:0.052465570376,ops:"YZYI"},
  {coeff:0.043229971464,ops:"XIXI"},{coeff:0.043229971464,ops:"YIYI"},
  {coeff:0.043229971464,ops:"ZXZX"},{coeff:0.043229971464,ops:"ZYZY"},
  {coeff:-0.036409483560,ops:"XXYY"},{coeff:0.036409483560,ops:"XYYX"},
  {coeff:0.036409483560,ops:"YXXY"},{coeff:-0.036409483560,ops:"YYXX"},
  {coeff:-0.009234661838,ops:"IXIX"},{coeff:-0.009234661838,ops:"IYIY"},
  {coeff:-0.009234661838,ops:"XZXZ"},{coeff:-0.009234661838,ops:"YZYZ"},
];

const LIH_TERMS = [
  {coeff:-7.508914538140,ops:"IIII"},{coeff:0.156139531201,ops:"ZIII"},{coeff:0.156139531201,ops:"IZII"},
  {coeff:0.121914456538,ops:"ZZII"},{coeff:0.084483749397,ops:"IIZZ"},
  {coeff:0.055938269342,ops:"ZIIZ"},{coeff:0.055938269342,ops:"IZZI"},
  {coeff:0.052684776986,ops:"IZIZ"},{coeff:0.052684776986,ops:"ZIZI"},
  {coeff:-0.014991186432,ops:"IIIZ"},{coeff:-0.014991186432,ops:"IIZI"},
  {coeff:0.013978294473,ops:"IXZX"},{coeff:0.013978294473,ops:"IYZY"},
  {coeff:0.013978294473,ops:"XZXI"},{coeff:0.013978294473,ops:"YZYI"},
  {coeff:0.012123738145,ops:"XIXI"},{coeff:0.012123738145,ops:"YIYI"},
  {coeff:0.012123738145,ops:"ZXZX"},{coeff:0.012123738145,ops:"ZYZY"},
  {coeff:-0.003253492355,ops:"XXYY"},{coeff:0.003253492355,ops:"XYYX"},
  {coeff:0.003253492355,ops:"YXXY"},{coeff:-0.003253492355,ops:"YYXX"},
  {coeff:-0.001854550186,ops:"IXIX"},{coeff:-0.001854550186,ops:"IYIY"},
  {coeff:-0.001854550186,ops:"XZXZ"},{coeff:-0.001854550186,ops:"YZYZ"},
];

const MOLECULES = {
  H2: {
    name: "Hydrogen (H\u2082)", formula: "H\u2082",
    description: "The simplest molecule: two hydrogen atoms sharing two electrons. The 'hello world' of quantum chemistry.",
    numQubits: 4, numElectrons: 2, numOrbitals: 2,
    orbitalLabels: ["\u03C3 (bonding)", "\u03C3* (antibonding)"],
    groundStateBitstring: "|1100\u27E9",
    bondLengthRange: [0.5, 3.0], equilibriumBondLength: 0.735,
    getPauliTerms: (bl) => interpolateH2(bl || 0.735),
    orbitalData: {
      atoms: [{symbol:"H",position:"left",atomicOrbitals:["1s"]},{symbol:"H",position:"right",atomicOrbitals:["1s"]}],
      molecularOrbitals: [
        {name:"\u03C3",type:"bonding",energy:-0.58,electrons:2,qubitIndices:[2,3],color:"#3b82f6"},
        {name:"\u03C3*",type:"antibonding",energy:0.67,electrons:0,qubitIndices:[0,1],color:"#ef4444"},
      ],
      connections: [
        {from:["H:1s","H:1s"],to:"\u03C3",type:"bonding"},
        {from:["H:1s","H:1s"],to:"\u03C3*",type:"antibonding"},
      ]
    },
  },
  "HeH+": {
    name: "Helium Hydride (HeH\u207A)", formula: "HeH\u207A",
    description: "A positively charged ion made of helium and hydrogen. Heteronuclear: the two atoms contribute differently to the molecular orbitals.",
    numQubits: 4, numElectrons: 2, numOrbitals: 2,
    orbitalLabels: ["\u03C3 (bonding)", "\u03C3* (antibonding)"],
    groundStateBitstring: "|1100\u27E9",
    bondLengthRange: null, equilibriumBondLength: 0.772,
    getPauliTerms: () => HEH_PLUS_TERMS,
    orbitalData: {
      atoms: [{symbol:"He",position:"left",atomicOrbitals:["1s"]},{symbol:"H",position:"right",atomicOrbitals:["1s"]}],
      molecularOrbitals: [
        {name:"\u03C3",type:"bonding",energy:-1.64,electrons:2,qubitIndices:[2,3],color:"#3b82f6"},
        {name:"\u03C3*",type:"antibonding",energy:-0.28,electrons:0,qubitIndices:[0,1],color:"#ef4444"},
      ],
      connections: [
        {from:["He:1s","H:1s"],to:"\u03C3",type:"bonding"},
        {from:["He:1s","H:1s"],to:"\u03C3*",type:"antibonding"},
      ]
    },
  },
  LiH: {
    name: "Lithium Hydride (LiH)", formula: "LiH",
    description: "Lithium bonded to hydrogen. Uses an 'active space' approximation: the core electrons on lithium are frozen, and we simulate only the 2 valence electrons in 2 active orbitals.",
    numQubits: 4, numElectrons: 2, numOrbitals: 2,
    orbitalLabels: ["\u03C3 (bonding)", "\u03C3* (antibonding)"],
    groundStateBitstring: "|1100\u27E9",
    bondLengthRange: null, equilibriumBondLength: 1.595,
    getPauliTerms: () => LIH_TERMS,
    frozenCore: true,
    frozenCoreNote: "LiH has 4 electrons total, but the 2 core electrons in lithium's 1s orbital are 'frozen' (not simulated). We model only the 2 valence electrons in the active space.",
    orbitalData: {
      atoms: [{symbol:"Li",position:"left",atomicOrbitals:["1s","2s"]},{symbol:"H",position:"right",atomicOrbitals:["1s"]}],
      molecularOrbitals: [
        {name:"\u03C3",type:"bonding",energy:-0.36,electrons:2,qubitIndices:[2,3],color:"#3b82f6"},
        {name:"\u03C3*",type:"antibonding",energy:0.24,electrons:0,qubitIndices:[0,1],color:"#ef4444"},
      ],
      frozenOrbitals: [{name:"1s (Li core)",energy:-2.48,electrons:2,color:"#9ca3af"}],
      connections: [
        {from:["Li:2s","H:1s"],to:"\u03C3",type:"bonding"},
        {from:["Li:2s","H:1s"],to:"\u03C3*",type:"antibonding"},
      ]
    },
  },
};

// ============================================================================
// PART 3: COURSE STRUCTURE
// ============================================================================

const PHASES = [
  { id: 1, title: "Molecules and Quantum States", color: "blue", icon: "\u269B\uFE0F",
    lessons: [
      { id: "1.1", title: "What Is a Molecule?", icon: "\u269B\uFE0F",
        sections: [
          { id: "1.1.1", title: "Atoms and Bonds", viz: null },
          { id: "1.1.2", title: "Meet the Electrons", viz: null },
          { id: "1.1.3", title: "Molecular Orbitals", viz: "V1" },
          { id: "1.1.4", title: "From Orbitals to Qubits", viz: "V1b" },
          { id: "1.1.5", title: "Key Takeaways & Practice", viz: null },
        ]
      },
      { id: "1.2", title: "Quantum States 101", icon: "\uD83C\uDFB2",
        sections: [
          { id: "1.2.1", title: "Bits and Qubits", viz: null },
          { id: "1.2.2", title: "Superposition and Probability", viz: "V2" },
          { id: "1.2.3", title: "Measuring a Quantum State", viz: null },
          { id: "1.2.4", title: "Key Takeaways & Practice", viz: null },
        ]
      },
    ]
  },
  { id: 2, title: "Building the Hamiltonian", color: "purple", icon: "\uD83D\uDCD0",
    lessons: [
      { id: "2.1", title: "The Hamiltonian -- Nature's Energy Rulebook", icon: "\uD83D\uDCD0",
        sections: [
          { id: "2.1.1", title: "What Is a Hamiltonian?", viz: null },
          { id: "2.1.2", title: "Where Does It Come From?", viz: null },
          { id: "2.1.3", title: "Pauli Operators -- The Building Blocks", viz: "V3" },
          { id: "2.1.4", title: "From Pauli Strings to the Hamiltonian Matrix", viz: "V4" },
          { id: "2.1.5", title: "Key Takeaways & Practice", viz: null },
        ]
      },
      { id: "2.2", title: "Exploring the Hamiltonian", icon: "\uD83D\uDD2C",
        sections: [
          { id: "2.2.1", title: "The Hamiltonian Matrix Up Close", viz: "V5" },
          { id: "2.2.2", title: "How the Molecule Changes the Matrix", viz: null },
          { id: "2.2.3", title: "Stretching the Bond (H\u2082 only)", viz: "V6" },
          { id: "2.2.4", title: "Key Takeaways & Practice", viz: null },
        ]
      },
    ]
  },
  { id: 3, title: "Diagonalization and Energy Levels", color: "green", icon: "\uD83D\uDCCA",
    lessons: [
      { id: "3.1", title: "Finding the Energy Levels", icon: "\uD83D\uDCCA",
        sections: [
          { id: "3.1.1", title: "What Is Diagonalization?", viz: null },
          { id: "3.1.2", title: "Eigenvalues = Energy Levels", viz: "V7" },
          { id: "3.1.3", title: "Eigenstates -- The Special States", viz: null },
          { id: "3.1.4", title: "The Potential Energy Curve (H\u2082)", viz: "V8" },
          { id: "3.1.5", title: "Key Takeaways & Practice", viz: null },
        ]
      },
    ]
  },
  { id: 4, title: "Time Evolution", color: "orange", icon: "\u23F1\uFE0F",
    lessons: [
      { id: "4.1", title: "Quantum Time Evolution", icon: "\u23F1\uFE0F",
        sections: [
          { id: "4.1.1", title: "The Schr\u00F6dinger Equation (Conceptual)", viz: null },
          { id: "4.1.2", title: "Why Eigenstates Are Special", viz: null },
          { id: "4.1.3", title: "The Time Evolution Playground", viz: "V9" },
          { id: "4.1.4", title: "Experiments to Try", viz: null },
          { id: "4.1.5", title: "Key Takeaways & Practice", viz: null },
        ]
      },
    ]
  },
];

const PHASE_COLORS = { blue: {bg:"bg-blue-50",border:"border-blue-300",text:"text-blue-700",accent:"bg-blue-500",light:"bg-blue-100",tab:"bg-blue-600"},
  purple: {bg:"bg-purple-50",border:"border-purple-300",text:"text-purple-700",accent:"bg-purple-500",light:"bg-purple-100",tab:"bg-purple-600"},
  green: {bg:"bg-green-50",border:"border-green-300",text:"text-green-700",accent:"bg-green-500",light:"bg-green-100",tab:"bg-green-600"},
  orange: {bg:"bg-orange-50",border:"border-orange-300",text:"text-orange-700",accent:"bg-orange-500",light:"bg-orange-100",tab:"bg-orange-600"},
};

function basisLabel(index, n) {
  return "|" + index.toString(2).padStart(n, "0") + "\u27E9";
}

// ============================================================================
// PART 4: SECTION CONTENT
// ============================================================================

function SectionContent({ sectionId, molecule, bondLength, setBondLength, phaseColor }) {
  const mol = MOLECULES[molecule];
  const colors = PHASE_COLORS[phaseColor];

  const contentMap = {
    "1.1.1": () => (
      <div className="space-y-4">
        <p>Everything around you is made of <strong>atoms</strong>. When atoms get close together, they can share electrons and form <strong>molecules</strong>. This sharing of electrons is called a <strong>chemical bond</strong>.</p>
        <p>In this course, we'll explore three simple molecules:</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(MOLECULES).map(([key, m]) => (
            <div key={key} className="p-4 border rounded-xl bg-white shadow-sm">
              <div className="text-2xl font-bold mb-1">{m.formula}</div>
              <div className="text-sm font-medium text-gray-600 mb-2">{m.name}</div>
              <p className="text-sm text-gray-500">{m.description}</p>
            </div>
          ))}
        </div>
        <p>Use the <strong>molecule selector</strong> at the top of the page to switch between these molecules. Your choice will affect all the visualizations throughout the course!</p>
        <BallAndStick molecule={molecule} bondLength={bondLength} />
      </div>
    ),
    "1.1.2": () => (
      <div className="space-y-4">
        <p>Electrons are tiny particles that orbit the nuclei of atoms. They carry negative charge and are the key to understanding chemistry. The way electrons are arranged determines a molecule's properties: its shape, color, reactivity, and more.</p>
        <p>For our molecules, we only need to track a small number of electrons:</p>
        <ul className="list-disc pl-6 space-y-1">
          <li><strong>H{"\u2082"}</strong>: 2 electrons (one from each hydrogen atom)</li>
          <li><strong>HeH{"\u207A"}</strong>: 2 electrons (two from helium, one from hydrogen, minus one for the positive charge)</li>
          <li><strong>LiH</strong>: 2 <em>active</em> electrons (lithium's core electrons are "frozen")</li>
        </ul>
        {mol.frozenCore && <div className={`p-3 ${colors.light} rounded-lg text-sm`}>{mol.frozenCoreNote}</div>}
        <p>Electrons live in <strong>orbitals</strong>: regions of space around the nuclei where electrons are most likely to be found. Think of them as "rooms" that electrons can occupy.</p>
      </div>
    ),
    "1.1.3": () => (
      <div className="space-y-4">
        <p>When atoms bond to form a molecule, their individual atomic orbitals combine into <strong>molecular orbitals</strong> (MOs). Two key types:</p>
        <ul className="list-disc pl-6 space-y-1">
          <li><strong className="text-blue-600">Bonding orbitals ({"\u03C3"})</strong>: Lower energy, stabilize the molecule. Electrons here hold the atoms together.</li>
          <li><strong className="text-red-600">Antibonding orbitals ({"\u03C3"}*)</strong>: Higher energy, destabilize the molecule. Electrons here push the atoms apart.</li>
        </ul>
        <p>Electrons fill orbitals from lowest to highest energy (the <strong>Aufbau principle</strong>). Each orbital can hold 2 electrons with opposite spins ({"\u2191\u2193"}).</p>
        <InstructionPanel color={phaseColor} show={`The molecular orbital energy level diagram for ${mol.formula}. Atomic orbitals on the sides combine to form molecular orbitals in the center.`} use="Select different molecules to see how the diagram changes. Hover over orbitals for details." observe="The bonding orbital is always lower energy than the antibonding orbital. Electrons fill the bonding orbital first." />
        <MODiagram molecule={molecule} showQubitMapping={false} />
      </div>
    ),
    "1.1.4": () => (
      <div className="space-y-4">
        <p>Here's the key bridge to quantum computing: each <strong>spin-orbital</strong> (an orbital + a spin direction) becomes one <strong>qubit</strong>.</p>
        <ul className="list-disc pl-6 space-y-1">
          <li>An occupied spin-orbital = |1{"\u27E9"}</li>
          <li>An empty spin-orbital = |0{"\u27E9"}</li>
        </ul>
        <p>For {mol.formula}: 2 spatial orbitals {"\u00D7"} 2 spins = <strong>4 qubits</strong> (q0, q1, q2, q3).</p>
        <p>The bonding orbital maps to qubits q2 (spin-up) and q3 (spin-down). The antibonding orbital maps to q0 (spin-up) and q1 (spin-down).</p>
        <p>The ground state configuration {mol.groundStateBitstring} means: q3=1, q2=1 (bonding orbital filled), q1=0, q0=0 (antibonding orbital empty).</p>
        <InstructionPanel color={phaseColor} show="The MO diagram with qubit labels." use="Toggle 'Show Qubit Mapping' to see how each spin-orbital corresponds to a qubit." observe={"The bitstring |1100\u27E9 directly encodes which orbitals are occupied."} />
        <MODiagram molecule={molecule} showQubitMapping={true} />
      </div>
    ),
    "1.1.5": () => (
      <div className="space-y-4">
        <h3 className="font-bold text-lg">Key Takeaways</h3>
        <ul className="list-disc pl-6 space-y-1">
          <li>Molecules form when atoms share electrons through chemical bonds.</li>
          <li>Atomic orbitals combine into bonding (lower energy) and antibonding (higher energy) molecular orbitals.</li>
          <li>Each spin-orbital maps to one qubit: occupied = |1{"\u27E9"}, empty = |0{"\u27E9"}.</li>
          <li>For our molecules, 2 spatial orbitals {"\u00D7"} 2 spins = 4 qubits.</li>
        </ul>
        <PracticeQuiz questions={[
          {q: "In the ground state of H\u2082, which orbitals are occupied?", a: "Both spin-orbitals of the bonding (\u03C3) orbital are occupied. The antibonding (\u03C3*) is empty."},
          {q: "What bitstring represents the ground state?", a: "|1100\u27E9: q3=1, q2=1 (bonding filled), q1=0, q0=0 (antibonding empty)."},
          {q: "If a molecule has 3 spatial orbitals, how many qubits do we need?", a: "3 \u00D7 2 = 6 qubits (each spatial orbital has spin-up and spin-down)."},
        ]} />
      </div>
    ),
    "1.2.1": () => (
      <div className="space-y-4">
        <p>A classical <strong>bit</strong> is either 0 or 1. A quantum bit (<strong>qubit</strong>) can be |0{"\u27E9"}, |1{"\u27E9"}, or a <strong>superposition</strong> of both.</p>
        <p>With multiple qubits, we can have many basis states. For 2 qubits: |00{"\u27E9"}, |01{"\u27E9"}, |10{"\u27E9"}, |11{"\u27E9"} (4 states). For n qubits: 2{"\u207F"} possible basis states.</p>
        <p>For our 4-qubit molecules: 2{"\u2074"} = <strong>16 basis states</strong>, from |0000{"\u27E9"} to |1111{"\u27E9"}.</p>
        <p>We write quantum states using <strong>Dirac notation</strong>: the ket |{"\u03C8\u27E9"} represents a quantum state. The bra {"\u27E8\u03C8"}| is its conjugate transpose.</p>
      </div>
    ),
    "1.2.2": () => (
      <div className="space-y-4">
        <p>A quantum state can be a combination of basis states, each with an <strong>amplitude</strong> (a number). The probability of measuring a particular basis state equals the square of its amplitude's magnitude.</p>
        <p>For example: |{"\u03C8\u27E9"} = 0.6|00{"\u27E9"} + 0.8|01{"\u27E9"} means P(|00{"\u27E9"}) = 0.36 and P(|01{"\u27E9"}) = 0.64.</p>
        <p>All probabilities must add to 1 (the <strong>normalization</strong> condition).</p>
        <InstructionPanel color={phaseColor} show="A 2-qubit quantum state with adjustable amplitudes and real-time probability display." use="Move the amplitude sliders and watch the probabilities update. Try the auto-normalize button." observe="The probabilities (squared amplitudes) always sum to 1 for a valid quantum state." />
        <SuperpositionExplorer />
      </div>
    ),
    "1.2.3": () => (
      <div className="space-y-4">
        <p><strong>Measurement</strong> collapses a superposition into a single definite basis state. Which state you get is random, but the probabilities are determined by the amplitudes.</p>
        <p>After measurement, the quantum state <em>becomes</em> the measured state. The superposition is gone.</p>
        <p>Think of it like spinning a weighted wheel: you know the probability of each outcome, but you can't predict which one you'll get on any single spin.</p>
        <p>This is why quantum computing is probabilistic: you often run the same quantum circuit many times and collect statistics.</p>
      </div>
    ),
    "1.2.4": () => (
      <div className="space-y-4">
        <h3 className="font-bold text-lg">Key Takeaways</h3>
        <ul className="list-disc pl-6 space-y-1">
          <li>Qubits can exist in superpositions of |0{"\u27E9"} and |1{"\u27E9"}.</li>
          <li>The probability of measuring a state equals the squared magnitude of its amplitude.</li>
          <li>All probabilities must sum to 1 (normalization).</li>
          <li>Measurement collapses the superposition into one definite state.</li>
        </ul>
        <PracticeQuiz questions={[
          {q: "If |\u03C8\u27E9 = 0.5|00\u27E9 + 0.5|01\u27E9 + 0.5|10\u27E9 + 0.5|11\u27E9, what are the probabilities?", a: "Each has probability 0.25 (= 0.5\u00B2). Total = 1.0. This is a valid, equally-weighted superposition."},
          {q: "Is |\u03C8\u27E9 = 0.8|0\u27E9 + 0.8|1\u27E9 a valid quantum state?", a: "No! 0.8\u00B2 + 0.8\u00B2 = 1.28 \u2260 1. It's not normalized."},
        ]} />
      </div>
    ),
    "2.1.1": () => (
      <div className="space-y-4">
        <p>The <strong>Hamiltonian</strong> is a mathematical object (a matrix) that encodes all the energy information of a quantum system. Think of it as nature's rulebook for the molecule.</p>
        <p>The Hamiltonian tells you two crucial things:</p>
        <ol className="list-decimal pl-6 space-y-1">
          <li><strong>Energy levels</strong>: What are the possible energies the molecule can have?</li>
          <li><strong>Time evolution</strong>: How does the quantum state change over time?</li>
        </ol>
        <p>For our 4-qubit molecules, the Hamiltonian is a <strong>16{"\u00D7"}16 matrix</strong>. Each row and column corresponds to one of the 16 basis states (|0000{"\u27E9"} through |1111{"\u27E9"}).</p>
      </div>
    ),
    "2.1.2": () => (
      <div className="space-y-4">
        <p>The molecular Hamiltonian captures several energy contributions:</p>
        <ul className="list-disc pl-6 space-y-1">
          <li><strong>Kinetic energy</strong> of electrons (how fast they move)</li>
          <li><strong>Electron-nucleus attraction</strong> (negative energy, stabilizing)</li>
          <li><strong>Electron-electron repulsion</strong> (positive energy, destabilizing)</li>
          <li><strong>Nucleus-nucleus repulsion</strong> (positive, constant for fixed geometry)</li>
        </ul>
        <p>Quantum chemistry software (like PySCF) computes these interactions from the laws of physics and produces a Hamiltonian. In this course, we use pre-computed Hamiltonians for our three molecules.</p>
        <p>The beautiful thing: all this complex physics gets encoded into a single matrix!</p>
      </div>
    ),
    "2.1.3": () => (
      <div className="space-y-4">
        <p>Pauli operators are the building blocks of quantum Hamiltonians. There are four single-qubit Pauli operators:</p>
        <ul className="list-disc pl-6 space-y-1">
          <li><strong>I</strong> (Identity): Does nothing. Leaves the qubit unchanged.</li>
          <li><strong>X</strong> (Pauli-X): Flips |0{"\u27E9"} to |1{"\u27E9"} and vice versa. Like a NOT gate.</li>
          <li><strong>Y</strong> (Pauli-Y): Flips with a phase. Involves imaginary numbers.</li>
          <li><strong>Z</strong> (Pauli-Z): Leaves |0{"\u27E9"} alone but flips the sign of |1{"\u27E9"}.</li>
        </ul>
        <p>For multiple qubits, we combine Pauli operators using the <strong>tensor product</strong> (Kronecker product). For example, "ZI" means "apply Z to the first qubit and I to the second."</p>
        <InstructionPanel color={phaseColor} show="The four Pauli matrices and their tensor products for 2 qubits." use={"Click the Pauli tabs to see each 2\u00D72 matrix. Use the dropdowns to build 2-qubit tensor products and see the resulting 4\u00D74 matrix."} observe="I leaves everything unchanged. X swaps rows/columns. Z flips signs. The tensor product 'tiles' the matrices." />
        <PauliExplorer />
      </div>
    ),
    "2.1.4": () => (
      <div className="space-y-4">
        <p>The molecular Hamiltonian is expressed as a <strong>weighted sum of Pauli strings</strong>:</p>
        <p className="text-center font-mono bg-gray-100 p-3 rounded-lg">H = c{"\u2081"}{"\u00D7"}P{"\u2081"} + c{"\u2082"}{"\u00D7"}P{"\u2082"} + ... + c{"\u2099"}{"\u00D7"}P{"\u2099"}</p>
        <p>Each term has a coefficient (a real number) and a 4-character Pauli string (like "IZZI" or "XXYY"). The coefficient tells you how much that interaction contributes to the total energy.</p>
        <p>For {mol.formula}, the Hamiltonian has {mol.getPauliTerms(bondLength).length} Pauli terms. Watch the "Build" animation to see how they combine into the full 16{"\u00D7"}16 matrix!</p>
        <InstructionPanel color={phaseColor} show={`The Pauli decomposition of the Hamiltonian for ${mol.formula} and how the terms combine into the full matrix.`} use="Hover over a Pauli term to highlight its contribution. Click 'Build Animation' to watch the matrix assemble term by term." observe="Some terms contribute to the diagonal (energy of individual states), while others create off-diagonal connections (transitions between states)." />
        <HamiltonianBuilder molecule={molecule} bondLength={bondLength} />
      </div>
    ),
    "2.1.5": () => (
      <div className="space-y-4">
        <h3 className="font-bold text-lg">Key Takeaways</h3>
        <ul className="list-disc pl-6 space-y-1">
          <li>The Hamiltonian is a matrix that encodes all energy information about a molecule.</li>
          <li>It is built from Pauli operators (I, X, Y, Z) using tensor products.</li>
          <li>The molecular Hamiltonian = weighted sum of Pauli strings.</li>
          <li>Diagonal elements represent state energies; off-diagonal elements represent transitions.</li>
        </ul>
        <PracticeQuiz questions={[
          {q: "What does the Pauli string 'IZZI' mean?", a: "Apply I to q3, Z to q2, Z to q1, I to q0. It measures whether q2 and q1 have the same or opposite values."},
          {q: "If a Hamiltonian term has coefficient 0, what does it contribute?", a: "Nothing! A zero coefficient means that particular Pauli interaction doesn't exist for this molecule."},
        ]} />
      </div>
    ),
    "2.2.1": () => (
      <div className="space-y-4">
        <p>Let's look at the full Hamiltonian matrix for {mol.formula}. This is a <strong>16{"\u00D7"}16 matrix</strong> where each row and column represents one of the 16 computational basis states.</p>
        <p>Key features to notice:</p>
        <ul className="list-disc pl-6 space-y-1">
          <li>It's <strong>Hermitian</strong> (symmetric for real entries): H[i][j] = H[j][i].</li>
          <li>The <strong>diagonal</strong> gives the "self-energy" of each basis state.</li>
          <li><strong>Off-diagonal</strong> entries represent transitions between states.</li>
          <li>Many entries are <strong>zero</strong> (the matrix is sparse).</li>
        </ul>
        <InstructionPanel color={phaseColor} show={`The full 16\u00D716 Hamiltonian matrix for ${mol.formula} as a color-coded heatmap.`} use="Hover over cells to see exact values and basis state labels. Click 'Diagonal Only' to focus on the energy of each state." observe="The matrix is symmetric. Blue means negative (stabilizing), red means positive (destabilizing), white means zero." />
        <HamiltonianHeatmap molecule={molecule} bondLength={bondLength} />
      </div>
    ),
    "2.2.2": () => (
      <div className="space-y-4">
        <p>Try switching between molecules using the selector at the top! Each molecule produces a different Hamiltonian matrix:</p>
        <ul className="list-disc pl-6 space-y-1">
          <li><strong>H{"\u2082"}</strong>: Simplest structure. High symmetry (homonuclear) means many related coefficients.</li>
          <li><strong>HeH{"\u207A"}</strong>: Heteronuclear, so the symmetry between bonding/antibonding is broken. More distinct Pauli terms.</li>
          <li><strong>LiH</strong>: Active-space Hamiltonian. Smaller energy scale in the active space since core electrons are frozen.</li>
        </ul>
        <p>The physical differences (different atoms, charges, electron configurations) are encoded entirely in the coefficients of the Pauli terms. Different molecules, different numbers!</p>
      </div>
    ),
    "2.2.3": () => (
      <div className="space-y-4">
        <p>For H{"\u2082"}, we can change the distance between the two hydrogen atoms. This is the <strong>bond length</strong>, measured in Angstroms ({"\u00C5"}).</p>
        <p>As you stretch or compress the bond:</p>
        <ul className="list-disc pl-6 space-y-1">
          <li><strong>Short distances</strong> ({`<`}0.5 {"\u00C5"}): Nuclear repulsion dominates. Very high energy.</li>
          <li><strong>Equilibrium</strong> (~0.735 {"\u00C5"}): The "sweet spot" where total energy is minimized.</li>
          <li><strong>Large distances</strong> ({`>`}2.0 {"\u00C5"}): The molecule approaches two independent hydrogen atoms.</li>
        </ul>
        {molecule === "H2" ? (
          <>
            <InstructionPanel color={phaseColor} show={"How the H\u2082 Hamiltonian changes with bond length."} use="Drag the bond length slider and watch the matrix update in real time." observe="At the equilibrium distance, the off-diagonal elements (coupling) are strongest. At large distances, the matrix becomes more diagonal." />
            <BondLengthExplorer bondLength={bondLength} setBondLength={setBondLength} />
          </>
        ) : (
          <div className={`p-4 ${colors.light} rounded-lg`}>
            <p className="font-medium">Switch to H{"\u2082"} to use the bond length slider! HeH{"\u207A"} and LiH use fixed equilibrium geometries.</p>
          </div>
        )}
      </div>
    ),
    "2.2.4": () => (
      <div className="space-y-4">
        <h3 className="font-bold text-lg">Key Takeaways</h3>
        <ul className="list-disc pl-6 space-y-1">
          <li>The Hamiltonian matrix is Hermitian (symmetric) and typically sparse.</li>
          <li>Different molecules produce different Hamiltonian structures.</li>
          <li>Changing bond length changes the Hamiltonian continuously.</li>
          <li>There's an equilibrium bond length where energy is minimized.</li>
        </ul>
        <PracticeQuiz questions={[
          {q: "A matrix is Hermitian if H[i][j] = H[j][i]* (complex conjugate of transpose). For real matrices, what does this simplify to?", a: "For real matrices, Hermitian = symmetric: H[i][j] = H[j][i]."},
          {q: "What happens to the Hamiltonian matrix as H\u2082 bond length goes to infinity?", a: "It approaches the Hamiltonian of two independent atoms. The coupling terms become weaker."},
        ]} />
      </div>
    ),
    "3.1.1": () => (
      <div className="space-y-4">
        <p><strong>Diagonalization</strong> is the process of finding a special basis where the Hamiltonian matrix becomes diagonal. In this eigenbasis, each diagonal entry is an <strong>energy eigenvalue</strong>.</p>
        <p>Mathematically: H = U D U{"\u2020"}, where:</p>
        <ul className="list-disc pl-6 space-y-1">
          <li><strong>D</strong> is a diagonal matrix of eigenvalues (energy levels)</li>
          <li><strong>U</strong> is a unitary matrix whose columns are the eigenvectors</li>
          <li>U{"\u2020"} is the conjugate transpose of U</li>
        </ul>
        <p>Think of diagonalization as finding the "natural coordinates" of the system. In these coordinates, the physics becomes simple: each eigenstate has a definite energy.</p>
      </div>
    ),
    "3.1.2": () => (
      <div className="space-y-4">
        <p>The eigenvalues of the Hamiltonian are the <strong>energy levels</strong> of the molecule. The lowest eigenvalue is the <strong>ground state energy</strong>; higher ones are <strong>excited states</strong>.</p>
        <p>For our 4-qubit system, there are 16 eigenvalues. Most quantum chemistry focuses on the lowest few.</p>
        <InstructionPanel color={phaseColor} show={`The energy levels (eigenvalues) and corresponding eigenstates for ${mol.formula}.`} use="Click on an energy level in the diagram to highlight it in the table. Switch molecules to compare." observe="The ground state has the lowest energy. The energy gap between the ground and first excited state determines many physical properties." />
        <EnergyLevelDiagram molecule={molecule} bondLength={bondLength} />
      </div>
    ),
    "3.1.3": () => (
      <div className="space-y-4">
        <p><strong>Eigenstates</strong> are special quantum states that don't change character over time. If you prepare a system in an eigenstate, it stays in that eigenstate (it only picks up a phase factor, which doesn't affect probabilities).</p>
        <p>That's why eigenstates are called <strong>"stationary states"</strong>: their probability distribution is frozen in time.</p>
        <p>The ground state eigenstate is the most stable configuration of the molecule. It's not simply |1100{"\u27E9"} but rather a superposition of basis states that minimizes the total energy.</p>
        <p>In the next phase, we'll see that non-eigenstates <em>do</em> change over time, with probabilities oscillating at rates determined by the energy gaps!</p>
      </div>
    ),
    "3.1.4": () => (
      <div className="space-y-4">
        <p>For H{"\u2082"}, we can plot the ground state energy as a function of bond length. This <strong>potential energy curve</strong> is one of the most important plots in quantum chemistry!</p>
        <p>The minimum of this curve gives:</p>
        <ul className="list-disc pl-6 space-y-1">
          <li>The <strong>equilibrium bond length</strong> (~0.735 {"\u00C5"} for H{"\u2082"})</li>
          <li>The <strong>ground state energy</strong> at equilibrium (~-1.137 Hartree)</li>
        </ul>
        <InstructionPanel color={phaseColor} show={"The potential energy curve for H\u2082: ground state energy vs. bond length."} use="Drag the marker along the curve. The energy and bond length update in real time." observe={"The curve has a minimum near 0.735 \u00C5. At short distances, energy rises steeply (repulsion). At large distances, it asymptotes (dissociation)."} />
        <PotentialEnergyCurve bondLength={bondLength} setBondLength={setBondLength} />
      </div>
    ),
    "3.1.5": () => (
      <div className="space-y-4">
        <h3 className="font-bold text-lg">Key Takeaways</h3>
        <ul className="list-disc pl-6 space-y-1">
          <li>Diagonalization finds the eigenvalues (energy levels) and eigenstates of the Hamiltonian.</li>
          <li>The ground state has the lowest energy; excited states have higher energies.</li>
          <li>Eigenstates are "stationary" - their probability distribution doesn't change over time.</li>
          <li>The potential energy curve shows how ground state energy varies with bond length.</li>
        </ul>
        <PracticeQuiz questions={[
          {q: "If two eigenvalues are E\u2080 = -1.14 Ha and E\u2081 = -0.59 Ha, what is the energy gap?", a: "\u0394E = E\u2081 - E\u2080 = -0.59 - (-1.14) = 0.55 Hartree."},
          {q: "Why is the equilibrium bond length special?", a: "It minimizes the total energy. The molecule naturally 'prefers' this distance."},
        ]} />
      </div>
    ),
    "4.1.1": () => (
      <div className="space-y-4">
        <p>The Hamiltonian doesn't just tell you about energy levels; it also governs how quantum states <strong>change over time</strong>. The rule is:</p>
        <p className="text-center font-mono bg-gray-100 p-3 rounded-lg">|{"\u03C8"}(t){"\u27E9"} = e^(-iHt) |{"\u03C8"}(0){"\u27E9"}</p>
        <p>The operator U(t) = e^(-iHt) is the <strong>time evolution operator</strong>. Don't worry about the exponential of a matrix; think of it as:</p>
        <ul className="list-disc pl-6 space-y-1">
          <li>The Hamiltonian generates a "rotation" in quantum state space.</li>
          <li>Larger energy differences cause faster rotation.</li>
          <li>The time parameter t is in natural units (where {"\u0127"} = 1).</li>
        </ul>
      </div>
    ),
    "4.1.2": () => (
      <div className="space-y-4">
        <p>Eigenstates are special under time evolution. An eigenstate |E_k{"\u27E9"} evolves as:</p>
        <p className="text-center font-mono bg-gray-100 p-3 rounded-lg">|E_k{"\u27E9"} {"\u2192"} e^(-iE_k t) |E_k{"\u27E9"}</p>
        <p>The factor e^(-iE_k t) is just a <strong>phase</strong>. It changes the "direction" of the amplitude in the complex plane but doesn't affect the probability: |e^(-iE_k t)|{"\u00B2"} = 1. So the probability distribution <strong>doesn't change</strong>!</p>
        <p>For a <strong>superposition</strong> of eigenstates, each component rotates at a different rate (proportional to its energy). This causes the probability distribution to oscillate over time.</p>
        <p>The oscillation period is related to the <strong>energy gap</strong>: larger gaps mean faster oscillation.</p>
      </div>
    ),
    "4.1.3": () => (
      <div className="space-y-4">
        <InstructionPanel color={phaseColor} show={`Real-time quantum state evolution under the Hamiltonian of ${mol.formula}.`} use="Select an initial state, press Play, and watch the probabilities evolve. Adjust speed with the slider. Try different initial states and molecules!" observe={"Eigenstates don't change (stationary). Basis states like |1100\u27E9 oscillate because they're superpositions of eigenstates. The oscillation speed depends on energy gaps."} />
        <TimeEvolutionPlayground molecule={molecule} bondLength={bondLength} />
      </div>
    ),
    "4.1.4": () => (
      <div className="space-y-4">
        <h3 className="font-bold text-lg">Experiments to Try</h3>
        <p>Go back to the Time Evolution Playground and try these experiments:</p>
        <div className="space-y-3">
          <div className="p-3 bg-orange-50 rounded-lg border border-orange-200">
            <p className="font-medium">Experiment 1: Stationary State</p>
            <p className="text-sm">Select "Ground state (E{"\u2080"})" as the initial state. Press Play. Nothing changes! It's stationary.</p>
          </div>
          <div className="p-3 bg-orange-50 rounded-lg border border-orange-200">
            <p className="font-medium">Experiment 2: Oscillating Configuration</p>
            <p className="text-sm">Select "|1100{"\u27E9"}" (the ground configuration bitstring). Press Play. Watch the probabilities oscillate. This happens because |1100{"\u27E9"} is NOT an eigenstate.</p>
          </div>
          <div className="p-3 bg-orange-50 rounded-lg border border-orange-200">
            <p className="font-medium">Experiment 3: Two-State Superposition</p>
            <p className="text-sm">Select "E{"\u2080"} + E{"\u2081"}" (superposition of ground and first excited state). See smooth oscillation. The period relates to the energy gap {"\u0394"}E = E{"\u2081"} - E{"\u2080"}.</p>
          </div>
          <div className="p-3 bg-orange-50 rounded-lg border border-orange-200">
            <p className="font-medium">Experiment 4: Compare Molecules</p>
            <p className="text-sm">Switch between H{"\u2082"}, HeH{"\u207A"}, and LiH with the same initial state type. Larger energy gaps = faster evolution.</p>
          </div>
        </div>
      </div>
    ),
    "4.1.5": () => (
      <div className="space-y-4">
        <h3 className="font-bold text-lg">Course Complete! Key Takeaways</h3>
        <ul className="list-disc pl-6 space-y-1">
          <li>The Hamiltonian governs time evolution: |{"\u03C8"}(t){"\u27E9"} = e^(-iHt) |{"\u03C8"}(0){"\u27E9"}.</li>
          <li>Eigenstates are stationary: their probabilities don't change over time.</li>
          <li>Superpositions of eigenstates oscillate at rates determined by energy gaps.</li>
          <li>Larger energy gaps produce faster oscillation.</li>
        </ul>
        <p className="mt-4 text-lg font-medium">Congratulations! You've learned how molecules are represented as quantum Hamiltonians, how to find their energy levels through diagonalization, and how quantum states evolve over time. These are the foundations of quantum molecular simulation!</p>
        <PracticeQuiz questions={[
          {q: "Why does the ground state eigenstate not change over time?", a: "Under time evolution, it picks up a phase e^(-iE\u2080t), but the probability |e^(-iE\u2080t)|\u00B2 = 1 is unchanged. Only the 'direction' in complex space rotates, not the magnitude."},
          {q: "If \u0394E = 0.55 Ha between two eigenstates, what is the oscillation period?", a: "T = 2\u03C0/\u0394E = 2\u03C0/0.55 \u2248 11.4 (in natural units where \u0127=1)."},
        ]} />
      </div>
    ),
  };

  const render = contentMap[sectionId];
  return render ? render() : <p className="text-gray-500">Content for section {sectionId} is coming soon.</p>;
}

// ============================================================================
// PART 5: HELPER COMPONENTS
// ============================================================================

function InstructionPanel({ color, show, use, observe }) {
  const colors = PHASE_COLORS[color] || PHASE_COLORS.blue;
  return (
    <div className={`${colors.light} p-4 rounded-xl space-y-2 text-sm`}>
      <div><span className="font-bold">{"\uD83C\uDFAF"} What this shows: </span>{show}</div>
      <div><span className="font-bold">{"\uD83C\uDFAE"} How to use: </span>{use}</div>
      <div><span className="font-bold">{"\uD83D\uDC40"} What to observe: </span>{observe}</div>
    </div>
  );
}

function NarrativeHelp({ children }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button onClick={() => setOpen(!open)} title="Explain this diagram"
        className={`w-7 h-7 rounded-full flex items-center justify-center text-sm font-bold transition-colors ${open ? 'bg-amber-200 text-amber-800 border-amber-400' : 'bg-amber-50 text-amber-600 border-amber-200 hover:bg-amber-100'} border`}>?</button>
      {open && (
        <div className="mt-3 p-4 bg-amber-50 border border-amber-200 rounded-xl text-sm text-amber-900 leading-relaxed space-y-2">
          {children}
        </div>
      )}
    </>
  );
}

function PracticeQuiz({ questions }) {
  const [revealed, setRevealed] = useState({});
  return (
    <div className="space-y-3 mt-4">
      <h4 className="font-bold">Practice</h4>
      {questions.map((q, i) => (
        <div key={i} className="p-3 bg-gray-50 rounded-lg border">
          <p className="font-medium text-sm mb-2">{q.q}</p>
          {revealed[i] ? (
            <p className="text-sm text-green-700 bg-green-50 p-2 rounded">{q.a}</p>
          ) : (
            <button onClick={() => setRevealed(r => ({...r, [i]: true}))}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium">Show Answer</button>
          )}
        </div>
      ))}
    </div>
  );
}

function BallAndStick({ molecule, bondLength }) {
  const mol = MOLECULES[molecule];
  const atoms = mol.orbitalData.atoms;
  const bl = bondLength || mol.equilibriumBondLength;
  const scale = 60;
  const cx = 200, cy = 60;
  const d = bl * scale / 2;
  return (
    <div className="flex justify-center">
      <svg width="400" height="120" className="bg-white rounded-lg border">
        <line x1={cx-d+15} y1={cy} x2={cx+d-15} y2={cy} stroke="#94a3b8" strokeWidth="4" />
        <circle cx={cx-d} cy={cy} r="20" fill="#3b82f6" stroke="#1e40af" strokeWidth="2" />
        <text x={cx-d} y={cy+5} textAnchor="middle" fill="white" fontWeight="bold" fontSize="14">{atoms[0].symbol}</text>
        <circle cx={cx+d} cy={cy} r="20" fill={atoms[1].symbol==="H"?"#3b82f6":"#10b981"} stroke={atoms[1].symbol==="H"?"#1e40af":"#047857"} strokeWidth="2" />
        <text x={cx+d} y={cy+5} textAnchor="middle" fill="white" fontWeight="bold" fontSize="14">{atoms[1].symbol}</text>
        <text x={cx} y={cy+50} textAnchor="middle" fill="#64748b" fontSize="12">Bond length: {bl.toFixed(3)} {"\u00C5"}</text>
      </svg>
    </div>
  );
}

// ============================================================================
// PART 6: VISUALIZATIONS
// ============================================================================

// V1: Molecular Orbital Energy Level Diagram
function MODiagram({ molecule, showQubitMapping }) {
  const [showQubits, setShowQubits] = useState(showQubitMapping);
  const [hoveredOrb, setHoveredOrb] = useState(null);
  const mol = MOLECULES[molecule];
  const od = mol.orbitalData;
  const mos = od.molecularOrbitals;
  const frozen = od.frozenOrbitals || [];
  const w = 500, h = 320;
  const eMin = Math.min(...mos.map(m=>m.energy), ...frozen.map(f=>f.energy)) - 0.5;
  const eMax = Math.max(...mos.map(m=>m.energy)) + 0.5;
  const yScale = (e) => 40 + (h-80)*(1-(e-eMin)/(eMax-eMin));

  return (
    <div className="bg-white p-4 rounded-xl border shadow-sm">
      <div className="flex justify-between items-center mb-3">
        <h4 className="font-bold text-sm">Molecular Orbital Diagram: {mol.formula}</h4>
        <div className="flex items-center gap-2">
          <NarrativeHelp>
            {molecule === "H2" ? (<>
              <p>In the diagram, you can see <strong>H 1s</strong> on the left and <strong>H 1s</strong> on the right, representing the two separate hydrogen atoms. The dashed lines show what happens when those atoms come together to form H{"\u2082"}: their two 1s orbitals combine to create two new molecular orbitals in the center.</p>
              <p>The lower one (<strong>{"\u03C3"}, in blue</strong>) is the <em>bonding orbital</em>, which is lower in energy than either atomic orbital. This is where the two electrons end up (shown by the up/down arrows), and it's what holds the molecule together.</p>
              <p>The upper one (<strong>{"\u03C3"}*, in red</strong>) is the <em>antibonding orbital</em>, which is higher in energy. It's empty in the ground state.</p>
              <p>So the diagram is telling a story from left to right and right to left: two isolated hydrogen atoms, each with a 1s orbital, come together and their orbitals merge into bonding and antibonding molecular orbitals. It's the visual version of "two atoms forming a bond."</p>
            </>) : molecule === "HeH+" ? (<>
              <p>This diagram shows how <strong>helium</strong> and <strong>hydrogen</strong> combine to form HeH{"\u207A"}. Notice that the He 1s orbital on the left sits <em>lower in energy</em> than the H 1s on the right. That's because helium's nucleus has two protons, so it pulls its electron in more tightly.</p>
              <p>When these two different atomic orbitals combine, they still form one bonding ({"\u03C3"}) and one antibonding ({"\u03C3"}*) molecular orbital. But because He contributes more to the lower-energy bonding orbital, the bond is <em>polarized</em>: the electron density is shifted toward helium. This asymmetry is what makes HeH{"\u207A"} a heteronuclear molecule.</p>
              <p>Both electrons sit in the bonding orbital, just like H{"\u2082"}, but the unequal atomic contributions give this molecule a different character.</p>
            </>) : (<>
              <p>LiH is more complex because lithium has more electrons. The diagram shows the <strong>Li 1s core orbital</strong> (grayed out, marked "frozen") at very low energy. These two core electrons are tightly bound and don't participate in bonding, so we freeze them out of the simulation.</p>
              <p>The active chemistry happens between lithium's <strong>2s orbital</strong> and hydrogen's <strong>1s orbital</strong>. These combine to form the bonding ({"\u03C3"}) and antibonding ({"\u03C3"}*) molecular orbitals, just as in H{"\u2082"} and HeH{"\u207A"}.</p>
              <p>The two valence electrons fill the bonding orbital. By focusing only on these 2 active electrons in 2 active orbitals (a "CAS(2,2)" active space), we capture the essential bonding physics with just 4 qubits while avoiding the cost of simulating the inert core.</p>
            </>)}
          </NarrativeHelp>
          <button onClick={() => setShowQubits(!showQubits)}
            className={`text-xs px-3 py-1 rounded-full border ${showQubits ? 'bg-blue-100 border-blue-300 text-blue-700' : 'bg-gray-100'}`}>
            {showQubits ? "Hide" : "Show"} Qubit Mapping
          </button>
        </div>
      </div>
      <svg width={w} height={h} className="mx-auto block">
        {/* Energy axis */}
        <line x1="40" y1="30" x2="40" y2={h-30} stroke="#94a3b8" strokeWidth="1" />
        <text x="15" y={h/2} textAnchor="middle" transform={`rotate(-90,15,${h/2})`} fill="#64748b" fontSize="11">Energy</text>
        <polygon points={`40,25 37,35 43,35`} fill="#94a3b8" />

        {/* Frozen orbitals */}
        {frozen.map((fo, i) => (
          <g key={`frozen-${i}`}>
            <line x1="200" y1={yScale(fo.energy)} x2="300" y2={yScale(fo.energy)} stroke={fo.color} strokeWidth="3" strokeDasharray="4,2" />
            <text x="250" y={yScale(fo.energy)-10} textAnchor="middle" fill="#9ca3af" fontSize="10">{fo.name} (frozen)</text>
            <text x="260" y={yScale(fo.energy)+4} textAnchor="start" fill="#9ca3af" fontSize="14">{"\u2191\u2193"}</text>
          </g>
        ))}

        {/* Atomic orbitals */}
        {od.atoms.map((atom, ai) => {
          const x = ai === 0 ? 90 : 410;
          const aoEnergy = atom.symbol === "He" ? -0.92 : atom.symbol === "Li" ? -0.20 : -0.50;
          return (
            <g key={`atom-${ai}`}>
              <line x1={x-30} y1={yScale(aoEnergy)} x2={x+30} y2={yScale(aoEnergy)} stroke="#94a3b8" strokeWidth="2" />
              <text x={x} y={yScale(aoEnergy)-10} textAnchor="middle" fill="#64748b" fontSize="11">{atom.symbol} {atom.atomicOrbitals[atom.atomicOrbitals.length-1]}</text>
            </g>
          );
        })}

        {/* Molecular orbitals */}
        {mos.map((mo, i) => {
          const y = yScale(mo.energy);
          const x1 = 200, x2 = 300;
          return (
            <g key={`mo-${i}`} onMouseEnter={() => setHoveredOrb(mo)} onMouseLeave={() => setHoveredOrb(null)} className="cursor-pointer">
              <line x1={x1} y1={y} x2={x2} y2={y} stroke={mo.color} strokeWidth="3" />
              <text x={x1-5} y={y+4} textAnchor="end" fill={mo.color} fontSize="11" fontWeight="bold">{mo.name}</text>
              {mo.electrons >= 1 && <text x={235} y={y+5} textAnchor="middle" fill={mo.color} fontSize="16">{"\u2191"}</text>}
              {mo.electrons >= 2 && <text x={265} y={y+5} textAnchor="middle" fill={mo.color} fontSize="16">{"\u2193"}</text>}
              {showQubits && (
                <>
                  <text x={230} y={y+20} textAnchor="middle" fill="#6b7280" fontSize="9">q{mo.qubitIndices[0]}</text>
                  <text x={270} y={y+20} textAnchor="middle" fill="#6b7280" fontSize="9">q{mo.qubitIndices[1]}</text>
                </>
              )}
              {/* Connection lines to atomic orbitals */}
              {od.connections.filter(c => c.to === mo.name).map((conn, ci) => {
                const lx = 120, rx = 380;
                return (
                  <g key={ci}>
                    <line x1={lx} y1={yScale(od.atoms[0].symbol==="He"?-0.92:od.atoms[0].symbol==="Li"?-0.20:-0.50)} x2={x1} y2={y} stroke="#d1d5db" strokeWidth="1" strokeDasharray="4,2" />
                    <line x1={rx} y1={yScale(-0.50)} x2={x2} y2={y} stroke="#d1d5db" strokeWidth="1" strokeDasharray="4,2" />
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* Qubit mapping annotation */}
        {showQubits && (
          <text x={250} y={h-5} textAnchor="middle" fill="#374151" fontSize="11" fontWeight="bold">
            Ground state: {mol.groundStateBitstring}
          </text>
        )}
      </svg>

      {/* Hover tooltip */}
      {hoveredOrb && (
        <div className="mt-2 p-2 bg-gray-50 rounded text-sm">
          <strong>{hoveredOrb.name}</strong> ({hoveredOrb.type}) | Energy: {hoveredOrb.energy.toFixed(2)} Ha | Electrons: {hoveredOrb.electrons} | Qubits: q{hoveredOrb.qubitIndices[0]}, q{hoveredOrb.qubitIndices[1]}
        </div>
      )}
    </div>
  );
}

// V2: Superposition and Probability Explorer
function SuperpositionExplorer() {
  const [amps, setAmps] = useState([0.5, 0.5, 0.5, 0.5]);
  const labels = ["|00\u27E9", "|01\u27E9", "|10\u27E9", "|11\u27E9"];
  const sumSq = amps.reduce((s, a) => s + a*a, 0);
  const normalized = sumSq > 0;
  const probs = amps.map(a => a*a / (sumSq || 1));

  const autoNormalize = () => {
    const norm = Math.sqrt(sumSq);
    if(norm > 0) setAmps(amps.map(a => a/norm));
  };

  return (
    <div className="bg-white p-4 rounded-xl border shadow-sm space-y-4">
      <div className="flex justify-between items-center">
        <h4 className="font-bold text-sm">2-Qubit Superposition Explorer</h4>
        <NarrativeHelp>
          <p>On the left are four <strong>amplitude sliders</strong>, one for each of the four 2-qubit basis states: |00{"\u27E9"}, |01{"\u27E9"}, |10{"\u27E9"}, and |11{"\u27E9"}. Each amplitude is a number that describes how much of that basis state is "in the mix."</p>
          <p>On the right, the <strong>probability bars</strong> show the chance of measuring each state. Probability = amplitude squared, so an amplitude of 0.5 gives a probability of 0.25 (25%).</p>
          <p>The normalization indicator at the bottom tells you whether your amplitudes form a valid quantum state. For a real physical state, all probabilities must add up to exactly 1. If they don't, hit <strong>Auto-Normalize</strong> to fix them.</p>
          <p>Try setting one amplitude to 1 and the rest to 0: you get a "definite" state with 100% probability. Then try making all four equal: you get a uniform superposition where every outcome is equally likely.</p>
        </NarrativeHelp>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-3">
          <p className="text-xs font-medium text-gray-600">Amplitudes</p>
          {amps.map((a, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="font-mono text-sm w-10">{labels[i]}</span>
              <input type="range" min="-1" max="1" step="0.01" value={a}
                onChange={e => { const n=[...amps]; n[i]=parseFloat(e.target.value); setAmps(n); }}
                className="flex-1" />
              <span className="text-sm w-12 text-right">{a.toFixed(2)}</span>
            </div>
          ))}
          <div className={`text-sm p-2 rounded ${Math.abs(sumSq-1)<0.01 ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700'}`}>
            {"\u2211"}|a|{"\u00B2"} = {sumSq.toFixed(3)} {Math.abs(sumSq-1)<0.01 ? "\u2705 Normalized" : "\u26A0\uFE0F Not normalized"}
          </div>
          <button onClick={autoNormalize} className="text-xs px-3 py-1 bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200">Auto-Normalize</button>
        </div>
        <div>
          <p className="text-xs font-medium text-gray-600 mb-2">Probabilities</p>
          <div className="space-y-2">
            {probs.map((p, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="font-mono text-sm w-10">{labels[i]}</span>
                <div className="flex-1 bg-gray-100 rounded-full h-5 overflow-hidden">
                  <div className="bg-blue-500 h-full rounded-full transition-all" style={{width: `${p*100}%`}} />
                </div>
                <span className="text-sm w-14 text-right">{(p*100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// V3: Pauli Matrix Explorer
function PauliExplorer() {
  const [selected, setSelected] = useState('I');
  const [q0op, setQ0op] = useState('I');
  const [q1op, setQ1op] = useState('I');

  const pauliNames = ['I','X','Y','Z'];
  const pauliDisplays = {
    I: [[1,0],[0,1]], X: [[0,1],[1,0]], Y: [["0","-i"],["i","0"]], Z: [[1,0],[0,-1]]
  };
  const pauliActions = {
    I: {"|0\u27E9": "|0\u27E9", "|1\u27E9": "|1\u27E9"},
    X: {"|0\u27E9": "|1\u27E9", "|1\u27E9": "|0\u27E9"},
    Y: {"|0\u27E9": "i|1\u27E9", "|1\u27E9": "-i|0\u27E9"},
    Z: {"|0\u27E9": "|0\u27E9", "|1\u27E9": "-|1\u27E9"},
  };
  const pauliDescriptions = {
    I: "Identity: leaves the qubit unchanged.",
    X: "Pauli-X: flips the qubit (like a NOT gate).",
    Y: "Pauli-Y: flips with a phase factor of i.",
    Z: "Pauli-Z: adds a minus sign to |1\u27E9.",
  };

  const tensor = useMemo(() => {
    const A = PAULI_MAP[q1op]; // q1 is more significant
    const B = PAULI_MAP[q0op];
    return kronecker(A, B);
  }, [q0op, q1op]);

  const formatComplex = (c) => {
    const [re, im] = c;
    if(Math.abs(re) < 1e-10 && Math.abs(im) < 1e-10) return "0";
    if(Math.abs(im) < 1e-10) return re === 1 ? "1" : re === -1 ? "-1" : re.toFixed(1);
    if(Math.abs(re) < 1e-10) return im === 1 ? "i" : im === -1 ? "-i" : im.toFixed(1)+"i";
    return `${re.toFixed(1)}${im>0?"+":""}${im.toFixed(1)}i`;
  };

  const cellColor = (val) => {
    const [re, im] = val;
    if(Math.abs(re) < 1e-10 && Math.abs(im) < 1e-10) return "bg-gray-50";
    if(Math.abs(im) > 1e-10) return "bg-purple-100";
    return re > 0 ? "bg-blue-100" : "bg-red-100";
  };

  return (
    <div className="bg-white p-4 rounded-xl border shadow-sm space-y-4">
      <div className="flex justify-between items-center">
        <h4 className="font-bold text-sm">Pauli Matrix Explorer</h4>
        <NarrativeHelp>
          <p>The top section shows the four <strong>single-qubit Pauli matrices</strong>. Each is a 2{"\u00D7"}2 grid of numbers that describes an operation on one qubit. Click <strong>I, X, Y, Z</strong> to see each one and what it does to |0{"\u27E9"} and |1{"\u27E9"}.</p>
          <p><strong>I</strong> (identity) does nothing. <strong>X</strong> is a bit-flip (like a NOT gate). <strong>Z</strong> leaves |0{"\u27E9"} alone but flips the sign of |1{"\u27E9"}. <strong>Y</strong> is a combination of X and Z with an imaginary factor.</p>
          <p>The bottom section shows the <strong>tensor product</strong>: what happens when you apply Pauli operators to two qubits simultaneously. Select an operator for each qubit using the dropdowns, and the resulting 4{"\u00D7"}4 matrix appears. This is how single-qubit building blocks scale up to multi-qubit systems.</p>
          <p>The color coding helps: blue cells are positive real numbers, red cells are negative, and purple cells involve imaginary numbers (from the Y operator).</p>
        </NarrativeHelp>
      </div>
      {/* Single-qubit section */}
      <div>
        <div className="flex gap-2 mb-3">
          {pauliNames.map(name => (
            <button key={name} onClick={() => setSelected(name)}
              className={`px-4 py-2 rounded-lg font-mono font-bold text-sm ${selected===name ? 'bg-blue-600 text-white' : 'bg-gray-100 hover:bg-gray-200'}`}>{name}</button>
          ))}
        </div>
        <div className="flex gap-6 items-start">
          <div>
            <p className="text-xs text-gray-500 mb-1">Matrix</p>
            <div className="grid grid-cols-2 gap-1">
              {pauliDisplays[selected].map((row, i) => row.map((val, j) => (
                <div key={`${i}-${j}`} className={`w-12 h-12 flex items-center justify-center text-sm font-mono rounded ${typeof val === 'string' && val.includes('i') ? 'bg-purple-100' : val === 0 ? 'bg-gray-50' : 'bg-blue-50'} border`}>{val}</div>
              )))}
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-xs text-gray-500">{pauliDescriptions[selected]}</p>
            <div className="space-y-1">
              {Object.entries(pauliActions[selected]).map(([input, output]) => (
                <div key={input} className="text-sm font-mono">{selected}{input} = {output}</div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Two-qubit tensor product */}
      <div className="border-t pt-4">
        <p className="text-sm font-medium mb-2">2-Qubit Tensor Product</p>
        <div className="flex items-center gap-3 mb-3">
          <div>
            <label className="text-xs text-gray-500">Qubit 1 (left):</label>
            <select value={q1op} onChange={e=>setQ1op(e.target.value)} className="ml-1 border rounded px-2 py-1 text-sm">
              {pauliNames.map(n=><option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <span className="font-mono text-lg">{"\u2297"}</span>
          <div>
            <label className="text-xs text-gray-500">Qubit 0 (right):</label>
            <select value={q0op} onChange={e=>setQ0op(e.target.value)} className="ml-1 border rounded px-2 py-1 text-sm">
              {pauliNames.map(n=><option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <span className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">= "{q1op}{q0op}"</span>
        </div>
        <div className="overflow-x-auto">
          <div className="grid grid-cols-4 gap-1 w-fit">
            {tensor.map((row, i) => row.map((val, j) => (
              <div key={`${i}-${j}`} className={`w-14 h-10 flex items-center justify-center text-xs font-mono rounded border ${cellColor(val)}`}>{formatComplex(val)}</div>
            )))}
          </div>
        </div>
        <div className="flex gap-4 mt-2 text-xs text-gray-500">
          <span>Rows/Cols: |00{"\u27E9"} |01{"\u27E9"} |10{"\u27E9"} |11{"\u27E9"}</span>
        </div>
      </div>
    </div>
  );
}

// V4: Hamiltonian Builder
function HamiltonianBuilder({ molecule, bondLength }) {
  const mol = MOLECULES[molecule];
  const terms = mol.getPauliTerms(bondLength);
  const [hoveredTerm, setHoveredTerm] = useState(null);
  const [buildStep, setBuildStep] = useState(-1);
  const [isBuilding, setIsBuilding] = useState(false);

  const n = 1 << mol.numQubits;
  const fullH = useMemo(() => buildHamiltonian(terms, mol.numQubits), [terms]);
  const partialH = useMemo(() => {
    if(buildStep < 0) return fullH;
    return buildHamiltonian(terms.slice(0, buildStep+1), mol.numQubits);
  }, [terms, buildStep, fullH]);
  const highlightH = useMemo(() => {
    if(hoveredTerm === null) return null;
    return buildHamiltonian([terms[hoveredTerm]], mol.numQubits);
  }, [terms, hoveredTerm]);

  const displayH = buildStep >= 0 ? partialH : fullH;

  // The IIII (identity) term is a uniform diagonal offset that dominates the color scale
  // (e.g. -7.509 for LiH vs 0.15 for the next largest term). Subtract it so the
  // physically meaningful structure drives the colors.
  const iiiiCoeff = useMemo(() => terms.find(t => t.ops === "IIII")?.coeff || 0, [terms]);
  const fullMaxVal = useMemo(() => {
    let mx = 0;
    for(let i=0;i<n;i++) for(let j=0;j<n;j++) {
      const v = i === j ? fullH[i][j][0] - iiiiCoeff : fullH[i][j][0];
      mx = Math.max(mx, Math.abs(v));
    }
    return mx || 1;
  }, [fullH, n, iiiiCoeff]);

  // Highlight the term being added during build animation
  const buildHighlightH = useMemo(() => {
    if(buildStep < 0) return null;
    return buildHamiltonian([terms[buildStep]], mol.numQubits);
  }, [terms, buildStep, mol.numQubits]);

  useEffect(() => {
    if(!isBuilding) return;
    if(buildStep >= terms.length - 1) { setIsBuilding(false); setBuildStep(-1); return; }
    const timer = setTimeout(() => setBuildStep(s => s + 1), 300);
    return () => clearTimeout(timer);
  }, [isBuilding, buildStep, terms.length]);

  const startBuild = () => { setBuildStep(0); setIsBuilding(true); };
  const resetBuild = () => { setBuildStep(-1); setIsBuilding(false); };

  return (
    <div className="bg-white p-4 rounded-xl border shadow-sm">
      <div className="flex justify-between items-center mb-3">
        <h4 className="font-bold text-sm">Hamiltonian Builder: {mol.formula}</h4>
        <div className="flex items-center gap-2">
          <NarrativeHelp>
            <p>The <strong>left panel</strong> lists every Pauli term that makes up the Hamiltonian for {mol.formula}. Each term has a coefficient (a number) and a 4-character Pauli string like "IZZI". Positive coefficients are in blue, negative in red.</p>
            <p>The <strong>right panel</strong> shows the resulting 16{"\u00D7"}16 Hamiltonian matrix as a heatmap. Each cell represents the interaction between two basis states. Blue cells are negative (stabilizing), red cells are positive (destabilizing), and white cells are zero.</p>
            <p>Hover over any Pauli term on the left and you'll see its specific contribution highlighted in the matrix with a purple border. This shows you exactly which matrix entries that term affects.</p>
            <p>Click <strong>"Build Animation"</strong> to watch the matrix assemble one term at a time. You'll see how each Pauli term adds its contribution, gradually building up the full Hamiltonian. It's like watching a recipe come together ingredient by ingredient.</p>
          </NarrativeHelp>
          <button onClick={startBuild} className="text-xs px-3 py-1 bg-purple-100 text-purple-700 rounded-full hover:bg-purple-200">Build Animation</button>
          <button onClick={resetBuild} className="text-xs px-3 py-1 bg-gray-100 rounded-full hover:bg-gray-200">Reset</button>
        </div>
      </div>
      <div className="flex gap-4">
        {/* Pauli terms list */}
        <div className="w-64 max-h-80 overflow-y-auto border rounded-lg p-2 space-y-1 flex-shrink-0">
          <p className="text-xs font-medium text-gray-500 mb-1">{terms.length} Pauli terms</p>
          {terms.map((term, i) => (
            <div key={i}
              onMouseEnter={() => setHoveredTerm(i)}
              onMouseLeave={() => setHoveredTerm(null)}
              className={`flex justify-between text-xs font-mono p-1 rounded cursor-pointer transition-colors
                ${hoveredTerm === i ? 'bg-purple-100' : buildStep >= 0 && i <= buildStep ? 'bg-green-50' : buildStep >= 0 && i > buildStep ? 'opacity-30' : 'hover:bg-gray-50'}`}>
              <span className="text-purple-700">{term.ops}</span>
              <span className={term.coeff >= 0 ? 'text-blue-600' : 'text-red-600'}>{term.coeff >= 0 ? '+' : ''}{term.coeff.toFixed(6)}</span>
            </div>
          ))}
        </div>
        {/* Matrix heatmap */}
        <div className="flex-1">
          <MatrixHeatmap H={displayH} n={n} highlightH={hoveredTerm !== null ? highlightH : buildHighlightH} fixedMaxVal={fullMaxVal} diagOffset={iiiiCoeff} />
          {buildStep >= 0 && <p className="text-xs text-gray-500 mt-1">Terms added: {buildStep+1}/{terms.length}</p>}
        </div>
      </div>
    </div>
  );
}

// Shared matrix heatmap component
function MatrixHeatmap({ H, n, highlightH, showDiagonalOnly, fixedMaxVal, diagOffset }) {
  const [tooltip, setTooltip] = useState(null);
  const computedMaxVal = useMemo(() => {
    let mx = 0;
    const off = diagOffset || 0;
    for(let i=0;i<n;i++) for(let j=0;j<n;j++) {
      const v = i === j ? H[i][j][0] - off : H[i][j][0];
      mx = Math.max(mx, Math.abs(v));
    }
    return mx || 1;
  }, [H, n, diagOffset]);
  const maxVal = fixedMaxVal || computedMaxVal;

  const cellSize = Math.min(28, Math.floor(400/n));

  const getColor = (rawVal, i, j) => {
    // Subtract the uniform identity offset from diagonal for coloring so that
    // the interesting structure isn't drowned out by the constant energy shift
    const val = (diagOffset !== undefined && i === j) ? rawVal - (diagOffset || 0) : rawVal;
    const v = val / maxVal;
    if(Math.abs(v) < 0.01) return '#f8fafc';
    if(v > 0) return `rgba(239,68,68,${Math.min(Math.abs(v),1)})`;
    return `rgba(59,130,246,${Math.min(Math.abs(v),1)})`;
  };

  return (
    <div className="overflow-x-auto">
      <div style={{display:'grid', gridTemplateColumns:`30px repeat(${n}, ${cellSize}px)`, gap:'1px'}}>
        <div />
        {Array.from({length:n}, (_,j) => (
          <div key={j} className="text-center" style={{fontSize:'7px',transform:'rotate(-45deg)',transformOrigin:'center',height:cellSize}}>
            {basisLabel(j, Math.log2(n))}
          </div>
        ))}
        {Array.from({length:n}, (_,i) => (
          <React.Fragment key={i}>
            <div className="flex items-center justify-end pr-1" style={{fontSize:'7px',height:cellSize}}>
              {basisLabel(i, Math.log2(n))}
            </div>
            {Array.from({length:n}, (_,j) => {
              const val = H[i][j][0];
              const isHighlighted = highlightH && Math.abs(highlightH[i][j][0]) > 1e-10;
              const dimmed = showDiagonalOnly && i !== j;
              return (
                <div key={j}
                  style={{width:cellSize,height:cellSize,backgroundColor: dimmed ? '#f8fafc' : getColor(val, i, j),
                    border: isHighlighted ? '2px solid #7c3aed' : '1px solid #e2e8f0'}}
                  className="cursor-pointer transition-all"
                  onMouseEnter={() => setTooltip({i,j,val})}
                  onMouseLeave={() => setTooltip(null)}
                />
              );
            })}
          </React.Fragment>
        ))}
      </div>
      {tooltip && (
        <div className="mt-2 text-xs bg-gray-50 p-2 rounded">
          {"\u27E8"}{basisLabel(tooltip.i, Math.log2(n)).slice(1,-1)}|H|{basisLabel(tooltip.j, Math.log2(n)).slice(1,-1)}{"\u27E9"} = {tooltip.val.toFixed(6)} Ha
        </div>
      )}
      <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
        <div className="flex items-center gap-1"><div className="w-3 h-3 bg-blue-500 rounded" />Negative</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 bg-gray-100 border rounded" />Zero</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 bg-red-500 rounded" />Positive</div>
      </div>
    </div>
  );
}

// V5: Full Hamiltonian Heatmap
function HamiltonianHeatmap({ molecule, bondLength }) {
  const mol = MOLECULES[molecule];
  const terms = mol.getPauliTerms(bondLength);
  const n = 1 << mol.numQubits;
  const H = useMemo(() => buildHamiltonian(terms, mol.numQubits), [terms]);
  const [diagOnly, setDiagOnly] = useState(false);

  return (
    <div className="bg-white p-4 rounded-xl border shadow-sm">
      <div className="flex justify-between items-center mb-3">
        <h4 className="font-bold text-sm">Hamiltonian Matrix: {mol.formula}</h4>
        <div className="flex items-center gap-2">
          <NarrativeHelp>
            <p>This is the full <strong>16{"\u00D7"}16 Hamiltonian matrix</strong> for {mol.formula}. Each row and column corresponds to one of the 16 computational basis states, from |0000{"\u27E9"} (top-left) to |1111{"\u27E9"} (bottom-right).</p>
            <p>The <strong>diagonal entries</strong> (top-left to bottom-right) represent the energy of each individual basis state in isolation. Click "Diagonal Only" to see just these values. The off-diagonal entries represent <strong>couplings</strong> between states: non-zero off-diagonal values mean those two states can transition into each other.</p>
            <p>The color scale uses blue for negative values (which tend to be stabilizing) and red for positive values (destabilizing). Hover over any cell to see its exact numerical value and the two basis states it connects.</p>
            <p>Notice the <strong>symmetry</strong>: the matrix looks the same if you flip it across the diagonal. This is because the Hamiltonian is Hermitian, a fundamental requirement of quantum mechanics that ensures energies are real numbers.</p>
          </NarrativeHelp>
          <button onClick={() => setDiagOnly(!diagOnly)}
            className={`text-xs px-3 py-1 rounded-full border ${diagOnly ? 'bg-green-100 border-green-300 text-green-700' : 'bg-gray-100'}`}>
            {diagOnly ? "Show All" : "Diagonal Only"}
          </button>
        </div>
      </div>
      <MatrixHeatmap H={H} n={n} showDiagonalOnly={diagOnly} />
    </div>
  );
}

// V6: Bond Length Explorer (H2 only)
function BondLengthExplorer({ bondLength, setBondLength }) {
  const terms = interpolateH2(bondLength);
  const H = useMemo(() => buildHamiltonian(terms, 4), [terms]);
  const n = 16;

  const trace = useMemo(() => {
    let t = 0; for(let i=0;i<n;i++) t += H[i][i][0]; return t;
  }, [H]);
  const maxOffDiag = useMemo(() => {
    let mx = 0;
    for(let i=0;i<n;i++) for(let j=0;j<n;j++) if(i!==j) mx = Math.max(mx, Math.abs(H[i][j][0]));
    return mx;
  }, [H]);

  return (
    <div className="bg-white p-4 rounded-xl border shadow-sm space-y-4">
      <div className="flex justify-between items-center">
        <h4 className="font-bold text-sm">Bond Length Explorer: H{"\u2082"}</h4>
        <NarrativeHelp>
          <p>This visualization connects the physical picture (two hydrogen atoms at some distance) to the mathematical picture (the Hamiltonian matrix). The <strong>ball-and-stick diagram</strong> at the top shows the two atoms, and the slider lets you stretch or compress the bond.</p>
          <p>As you drag the slider, the <strong>Hamiltonian matrix</strong> below updates in real time. Watch how the colors and magnitudes shift. At <strong>short distances</strong> (below 0.5 {"\u00C5"}), the nuclear repulsion dominates and everything gets large and red (positive). At the <strong>equilibrium</strong> (~0.735 {"\u00C5"}), the system reaches its energy minimum. At <strong>large distances</strong> (above 2.0 {"\u00C5"}), the off-diagonal couplings weaken as the two atoms stop "talking" to each other.</p>
          <p>The <strong>properties panel</strong> on the right shows summary statistics: the trace (sum of diagonal elements), the largest off-diagonal element (a measure of how strongly the states are coupled), and the number of Pauli terms.</p>
        </NarrativeHelp>
      </div>
      <BallAndStick molecule="H2" bondLength={bondLength} />
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium">Bond Length:</span>
        <input type="range" min="0.5" max="3.0" step="0.05" value={bondLength}
          onChange={e => setBondLength(parseFloat(e.target.value))} className="flex-1" />
        <span className="text-sm font-mono w-16">{bondLength.toFixed(2)} {"\u00C5"}</span>
      </div>
      <div className="flex gap-4">
        <div className="flex-1">
          <MatrixHeatmap H={H} n={n} />
        </div>
        <div className="w-48 space-y-2 text-sm">
          <div className="p-2 bg-gray-50 rounded">
            <p className="text-xs text-gray-500">Matrix Properties</p>
            <p>Trace: {trace.toFixed(4)}</p>
            <p>Max |off-diag|: {maxOffDiag.toFixed(4)}</p>
            <p>Pauli terms: {terms.length}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

// V7: Energy Level Diagram + Eigenstate Table
function EnergyLevelDiagram({ molecule, bondLength }) {
  const mol = MOLECULES[molecule];
  const terms = mol.getPauliTerms(bondLength);
  const H = useMemo(() => buildHamiltonian(terms, mol.numQubits), [terms]);
  const {eigenvalues, U} = useMemo(() => jacobiEigen(H), [H]);
  const [selectedLevel, setSelectedLevel] = useState(0);
  const nq = mol.numQubits;
  const dim = 1 << nq;

  // Get dominant basis states for each eigenvector
  const getEigenstateStr = (k) => {
    const vec = Array.from({length:dim}, (_,i) => U[i][k]);
    const probs = vec.map(c => cAbs2(c));
    const sorted = probs.map((p,i) => ({p,i})).sort((a,b) => b.p - a.p);
    const dominants = sorted.filter(x => x.p > 0.01).slice(0, 4);
    return dominants.map(d => `${(d.p).toFixed(3)}${basisLabel(d.i,nq)}`).join(" + ");
  };

  const eMin = Math.min(...eigenvalues);
  const eMax = Math.max(...eigenvalues);
  const eRange = eMax - eMin || 1;

  return (
    <div className="bg-white p-4 rounded-xl border shadow-sm">
      <div className="flex justify-between items-center mb-3">
        <h4 className="font-bold text-sm">Energy Levels: {mol.formula}</h4>
        <NarrativeHelp>
          <p>On the <strong>left</strong> is an energy level diagram. Each horizontal line represents one eigenvalue of the Hamiltonian, that is, one allowed energy level for the molecule. The lowest line (green, labeled E{"\u2080"}) is the <strong>ground state</strong>, the most stable energy. Lines above it are <strong>excited states</strong>, colored from green (low) to red (high).</p>
          <p>On the <strong>right</strong> is a table showing each energy level's numerical value in Hartree and electron-volts, along with its <strong>eigenstate</strong>: the quantum state (superposition of basis states) that has that energy. The dominant components tell you which computational basis states contribute most.</p>
          <p>Click any energy level line or table row to highlight the correspondence between the two views. The ground state is usually dominated by |1100{"\u27E9"} (bonding orbital filled), but it's not purely |1100{"\u27E9"}: there's always some admixture of other states due to electron correlation.</p>
          <p>The <strong>energy gap</strong> between E{"\u2080"} and E{"\u2081"} is particularly important: it determines how difficult it is to excite the molecule and sets the timescale for quantum dynamics.</p>
        </NarrativeHelp>
      </div>
      <div className="flex gap-4">
        {/* Energy level diagram */}
        <div className="w-48 flex-shrink-0">
          <svg width="180" height="320">
            <line x1="20" y1="10" x2="20" y2="310" stroke="#94a3b8" strokeWidth="1" />
            <text x="8" y="160" textAnchor="middle" transform="rotate(-90,8,160)" fill="#64748b" fontSize="10">Energy (Ha)</text>
            {eigenvalues.map((e, i) => {
              const y = 300 - (e - eMin)/eRange * 280;
              const hue = (i / (dim-1)) * 120; // green to red
              const show = i < 6 || i === dim - 1;
              if(!show && i !== 6) return null;
              if(i === 6) return <text key={i} x="100" y={y} textAnchor="middle" fill="#94a3b8" fontSize="10">...</text>;
              return (
                <g key={i} className="cursor-pointer" onClick={() => setSelectedLevel(i)}>
                  <line x1="40" y1={y} x2="160" y2={y} stroke={`hsl(${120-hue},70%,45%)`} strokeWidth={selectedLevel===i ? 3 : 2} />
                  <text x="35" y={y-4} textAnchor="end" fill="#64748b" fontSize="8">E{"\u2080".charCodeAt(0)+i > 0x2089 ? i : String.fromCharCode(0x2080+i)}</text>
                  {selectedLevel === i && <circle cx="165" cy={y} r="4" fill={`hsl(${120-hue},70%,45%)`} />}
                </g>
              );
            })}
          </svg>
        </div>
        {/* Eigenstate table */}
        <div className="flex-1 overflow-y-auto max-h-80">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="text-left p-1">Level</th>
                <th className="text-left p-1">Energy (Ha)</th>
                <th className="text-left p-1">Energy (eV)</th>
                <th className="text-left p-1">Eigenstate (dominant components)</th>
              </tr>
            </thead>
            <tbody>
              {eigenvalues.map((e, i) => (
                <tr key={i} className={`border-b cursor-pointer ${selectedLevel===i ? 'bg-blue-50' : 'hover:bg-gray-50'}`}
                  onClick={() => setSelectedLevel(i)}>
                  <td className="p-1 font-medium">{i===0?"Ground":i===1?"1st excited":i===2?"2nd excited":`E${i}`}</td>
                  <td className="p-1 font-mono">{e.toFixed(6)}</td>
                  <td className="p-1 font-mono">{(e*27.2114).toFixed(4)}</td>
                  <td className="p-1 font-mono text-xs">{getEigenstateStr(i)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// V8: Potential Energy Curve (H2)
function PotentialEnergyCurve({ bondLength, setBondLength }) {
  const bondLengths = Object.keys(H2_PARAMS).map(Number).sort((a,b)=>a-b);

  const energyData = useMemo(() => {
    return bondLengths.map(bl => {
      const terms = expandH2Params(H2_PARAMS[bl]);
      const H = buildHamiltonian(terms, 4);
      const {eigenvalues} = jacobiEigen(H);
      // Find ground state in 2-electron sector
      return { bl, e0: eigenvalues[0], e1: eigenvalues.length > 1 ? eigenvalues[1] : null };
    });
  }, []);

  const currentTerms = interpolateH2(bondLength);
  const currentH = useMemo(() => buildHamiltonian(currentTerms, 4), [currentTerms]);
  const currentEigen = useMemo(() => jacobiEigen(currentH), [currentH]);
  const currentE0 = currentEigen.eigenvalues[0];

  const w = 500, h = 280, pad = {l:60,r:20,t:20,b:40};
  const pw = w-pad.l-pad.r, ph = h-pad.t-pad.b;

  const blMin = 0.5, blMax = 3.0;
  const eVals = energyData.map(d => d.e0);
  const eMin = Math.min(...eVals) - 0.1;
  const eMax = Math.max(...eVals) + 0.1;

  const sx = (bl) => pad.l + (bl - blMin)/(blMax-blMin)*pw;
  const sy = (e) => pad.t + (1-(e-eMin)/(eMax-eMin))*ph;

  const pathStr = energyData.map((d,i) => `${i===0?'M':'L'}${sx(d.bl).toFixed(1)},${sy(d.e0).toFixed(1)}`).join(' ');

  // Equilibrium annotation
  const eqData = energyData.reduce((a,b) => a.e0 < b.e0 ? a : b);

  const handleDrag = (e) => {
    const svg = e.currentTarget.closest('svg');
    const rect = svg.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const bl = blMin + (x - pad.l)/pw*(blMax-blMin);
    setBondLength(Math.max(blMin, Math.min(blMax, bl)));
  };

  return (
    <div className="bg-white p-4 rounded-xl border shadow-sm">
      <div className="flex justify-between items-center mb-3">
        <h4 className="font-bold text-sm">Potential Energy Curve: H{"\u2082"}</h4>
        <NarrativeHelp>
          <p>This is one of the most important plots in quantum chemistry. The horizontal axis is the <strong>bond length</strong> (distance between the two hydrogen nuclei). The vertical axis is the <strong>ground state energy</strong> (the lowest eigenvalue of the Hamiltonian at that distance).</p>
          <p>The curve has a characteristic <strong>well shape</strong>. On the left (short distances), the energy rises steeply because the nuclei repel each other strongly. On the right (large distances), the energy levels off as the molecule dissociates into two separate atoms.</p>
          <p>The <strong>minimum of the curve</strong> (marked with a green dashed line) is the equilibrium bond length: the distance at which the molecule is most stable. For H{"\u2082"} in the STO-3G basis, this is approximately 0.735 {"\u00C5"}.</p>
          <p>Click anywhere on the plot (or drag the slider below) to move the red marker. The displayed energy updates to show you the exact value at that bond length. Every point on this curve required building a Hamiltonian matrix and diagonalizing it!</p>
        </NarrativeHelp>
      </div>
      <svg width={w} height={h} className="mx-auto block cursor-crosshair" onClick={handleDrag}>
        {/* Axes */}
        <line x1={pad.l} y1={h-pad.b} x2={w-pad.r} y2={h-pad.b} stroke="#94a3b8" strokeWidth="1" />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={h-pad.b} stroke="#94a3b8" strokeWidth="1" />
        <text x={w/2} y={h-5} textAnchor="middle" fill="#64748b" fontSize="11">Bond Length ({"\u00C5"})</text>
        <text x="12" y={h/2} textAnchor="middle" transform={`rotate(-90,12,${h/2})`} fill="#64748b" fontSize="11">Energy (Hartree)</text>

        {/* Grid lines and tick labels */}
        {[0.5,1.0,1.5,2.0,2.5,3.0].map(bl => (
          <g key={bl}>
            <line x1={sx(bl)} y1={h-pad.b} x2={sx(bl)} y2={h-pad.b+5} stroke="#94a3b8" />
            <text x={sx(bl)} y={h-pad.b+15} textAnchor="middle" fill="#64748b" fontSize="9">{bl}</text>
          </g>
        ))}
        {Array.from({length:5}, (_,i) => {
          const e = eMin + i*(eMax-eMin)/4;
          return (
            <g key={i}>
              <line x1={pad.l-5} y1={sy(e)} x2={pad.l} y2={sy(e)} stroke="#94a3b8" />
              <text x={pad.l-8} y={sy(e)+3} textAnchor="end" fill="#64748b" fontSize="9">{e.toFixed(2)}</text>
            </g>
          );
        })}

        {/* PES curve */}
        <path d={pathStr} fill="none" stroke="#3b82f6" strokeWidth="2.5" />

        {/* Equilibrium annotation */}
        <line x1={sx(eqData.bl)} y1={sy(eqData.e0)} x2={sx(eqData.bl)} y2={h-pad.b} stroke="#10b981" strokeWidth="1" strokeDasharray="4,2" />
        <text x={sx(eqData.bl)+5} y={h-pad.b-5} fill="#10b981" fontSize="9">R_eq = {eqData.bl.toFixed(3)} {"\u00C5"}</text>

        {/* Current position marker */}
        <circle cx={sx(bondLength)} cy={sy(currentE0)} r="6" fill="#ef4444" stroke="white" strokeWidth="2" />
      </svg>
      <div className="flex items-center gap-3 mt-3">
        <span className="text-sm">Bond Length:</span>
        <input type="range" min="0.5" max="3.0" step="0.01" value={bondLength}
          onChange={e => setBondLength(parseFloat(e.target.value))} className="flex-1" />
        <span className="text-sm font-mono">{bondLength.toFixed(3)} {"\u00C5"}</span>
        <span className="text-sm font-mono text-blue-600">E = {currentE0.toFixed(6)} Ha</span>
      </div>
    </div>
  );
}

// V9: Time Evolution Playground
function TimeEvolutionPlayground({ molecule, bondLength }) {
  const mol = MOLECULES[molecule];
  const terms = mol.getPauliTerms(bondLength);
  const H = useMemo(() => buildHamiltonian(terms, mol.numQubits), [terms]);
  const eigen = useMemo(() => jacobiEigen(H), [H]);
  const nq = mol.numQubits;
  const dim = 1 << nq;

  const [initialStateKey, setInitialStateKey] = useState("basis_1100");
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [time, setTime] = useState(0);
  const tMax = 40;
  const animRef = useRef(null);
  const lastTimeRef = useRef(null);

  // Build initial state options
  const initialStates = useMemo(() => {
    const states = {};
    // Basis states
    const interesting = [0b1100, 0b0011, 0b1010, 0b0110, 0b0101, 0b0000];
    interesting.forEach(idx => {
      if(idx < dim) {
        const key = `basis_${idx.toString(2).padStart(nq,'0')}`;
        const vec = Array.from({length:dim}, (_,i) => i===idx ? [...C1] : [...C0]);
        states[key] = { label: basisLabel(idx, nq), category: "Basis States", vec };
      }
    });
    // Eigenstates
    for(let k=0; k<Math.min(3,dim); k++) {
      const vec = Array.from({length:dim}, (_,i) => [...eigen.U[i][k]]);
      states[`eigen_${k}`] = { label: `E${String.fromCharCode(0x2080+k)} (${eigen.eigenvalues[k].toFixed(4)} Ha)`, category: "Eigenstates", vec };
    }
    // Superpositions
    const norm = 1/Math.sqrt(2);
    for(let k=1; k<Math.min(3,dim); k++) {
      const vec = Array.from({length:dim}, (_,i) => cScale(norm, cAdd(eigen.U[i][0], eigen.U[i][k])));
      states[`super_0_${k}`] = { label: `E\u2080 + E${String.fromCharCode(0x2080+k)}`, category: "Superpositions", vec };
    }
    return states;
  }, [eigen, dim, nq]);

  const psi0 = initialStates[initialStateKey]?.vec || initialStates["basis_1100"]?.vec || Array.from({length:dim}, (_,i) => i===0b1100 ? [...C1] : [...C0]);

  const psiT = useMemo(() => evolveState(eigen.U, eigen.eigenvalues, psi0, time), [eigen, psi0, time]);
  const probs = psiT.map(c => cAbs2(c));
  const totalProb = probs.reduce((s,p) => s+p, 0);

  // Check if stationary
  const isEigenstate = initialStateKey.startsWith("eigen_");

  // Animation loop
  useEffect(() => {
    if(!playing) { lastTimeRef.current = null; return; }
    const animate = (timestamp) => {
      if(lastTimeRef.current === null) lastTimeRef.current = timestamp;
      const dt = (timestamp - lastTimeRef.current) / 1000 * speed * 2;
      lastTimeRef.current = timestamp;
      setTime(t => {
        const nt = t + dt;
        return nt > tMax ? 0 : nt;
      });
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
    return () => { if(animRef.current) cancelAnimationFrame(animRef.current); };
  }, [playing, speed]);

  const reset = () => { setPlaying(false); setTime(0); lastTimeRef.current = null; };

  // Group states by category for select
  const categories = {};
  Object.entries(initialStates).forEach(([k, v]) => {
    if(!categories[v.category]) categories[v.category] = [];
    categories[v.category].push({key: k, label: v.label});
  });

  return (
    <div className="bg-white p-4 rounded-xl border shadow-sm space-y-4">
      <div className="flex justify-between items-center">
        <h4 className="font-bold text-sm">Time Evolution Playground: {mol.formula}</h4>
        <NarrativeHelp>
          <p>This is the capstone visualization of the course. The <strong>bar chart</strong> shows the probability of finding the system in each of the 16 computational basis states at the current time. Taller bars mean higher probability.</p>
          <p>Use the <strong>initial state dropdown</strong> to choose a starting state. There are three categories:</p>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>Basis states</strong> like |1100{"\u27E9"}: these are definite configurations, not eigenstates, so they <em>will</em> evolve over time.</li>
            <li><strong>Eigenstates</strong> like E{"\u2080"}: these are stationary states that <em>won't</em> change, since they're the Hamiltonian's "natural modes."</li>
            <li><strong>Superpositions</strong> like E{"\u2080"} + E{"\u2081"}: equal mixes of two eigenstates that oscillate smoothly at a rate set by the energy gap.</li>
          </ul>
          <p>Press <strong>Play</strong> to start the animation and watch probabilities flow between states. The speed slider controls how fast time passes. You can also drag the time scrubber to jump to any moment.</p>
          <p>The info panel at the bottom shows the current state as a superposition, the total probability (should always be 1.0), and a note if you've picked a stationary eigenstate.</p>
        </NarrativeHelp>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3 items-center">
        <select value={initialStateKey} onChange={e => { setInitialStateKey(e.target.value); reset(); }}
          className="border rounded px-2 py-1 text-sm flex-1 min-w-48">
          {Object.entries(categories).map(([cat, items]) => (
            <optgroup key={cat} label={cat}>
              {items.map(({key, label}) => <option key={key} value={key}>{label}</option>)}
            </optgroup>
          ))}
        </select>
        <button onClick={() => setPlaying(!playing)}
          className={`px-4 py-1 rounded-lg text-sm font-medium ${playing ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
          {playing ? "\u23F8 Pause" : "\u25B6 Play"}
        </button>
        <button onClick={reset} className="px-3 py-1 rounded-lg text-sm bg-gray-100 hover:bg-gray-200">Reset</button>
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-500">Speed:</span>
          <input type="range" min="0.1" max="4" step="0.1" value={speed}
            onChange={e => setSpeed(parseFloat(e.target.value))} className="w-20" />
          <span className="text-xs font-mono w-8">{speed.toFixed(1)}x</span>
        </div>
      </div>

      {/* Time scrubber */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500">t =</span>
        <input type="range" min="0" max={tMax} step="0.1" value={time}
          onChange={e => { setTime(parseFloat(e.target.value)); setPlaying(false); }} className="flex-1" />
        <span className="text-sm font-mono w-16">{time.toFixed(1)}</span>
      </div>

      {/* Probability bar chart */}
      <div className="overflow-x-auto">
        <svg width={Math.max(dim * 32 + 40, 400)} height="200" className="block">
          {probs.map((p, i) => {
            const barW = 24;
            const gap = 8;
            const x = 30 + i * (barW + gap);
            const barH = p * 150;
            const hue = (i / dim) * 280;
            return (
              <g key={i}>
                <rect x={x} y={170-barH} width={barW} height={barH} rx="2"
                  fill={`hsl(${hue},60%,55%)`} opacity="0.8" />
                <text x={x+barW/2} y={185} textAnchor="middle" fontSize="7" fill="#64748b"
                  transform={`rotate(-45,${x+barW/2},185)`}>
                  {basisLabel(i, nq)}
                </text>
                {p > 0.05 && <text x={x+barW/2} y={168-barH} textAnchor="middle" fontSize="8" fill="#374151">{(p*100).toFixed(0)}%</text>}
              </g>
            );
          })}
          {/* Y axis */}
          <line x1="28" y1="20" x2="28" y2="170" stroke="#94a3b8" strokeWidth="1" />
          {[0, 0.25, 0.5, 0.75, 1.0].map(v => (
            <g key={v}>
              <line x1="24" y1={170-v*150} x2="28" y2={170-v*150} stroke="#94a3b8" />
              <text x="22" y={174-v*150} textAnchor="end" fontSize="8" fill="#94a3b8">{v}</text>
            </g>
          ))}
          <text x="8" y="100" textAnchor="middle" transform="rotate(-90,8,100)" fill="#64748b" fontSize="9">Probability</text>
        </svg>
      </div>

      {/* Info panel */}
      <div className="text-xs space-y-1 bg-gray-50 p-3 rounded-lg">
        {isEigenstate && (
          <p className="text-green-700 font-medium">This is a stationary state: probabilities don't change!</p>
        )}
        <p>Total probability: {totalProb.toFixed(6)} {Math.abs(totalProb-1)<0.001 ? "\u2705" : "\u26A0\uFE0F"}</p>
        <p>Current state (terms {`>`} 1%): {
          probs.map((p,i) => ({p,i})).filter(x=>x.p>0.01).sort((a,b)=>b.p-a.p)
            .map(x => `${x.p.toFixed(3)}${basisLabel(x.i,nq)}`).join(" + ")
        }</p>
      </div>
    </div>
  );
}

// ============================================================================
// PART 7: AI CHAT TUTOR
// ============================================================================

function MarkdownRenderer({ text }) {
  // Simple markdown renderer for chat responses
  const lines = text.split('\n');
  return (
    <div className="space-y-1 text-sm">
      {lines.map((line, i) => {
        if(line.startsWith('# ')) return <h3 key={i} className="font-bold text-base">{line.slice(2)}</h3>;
        if(line.startsWith('## ')) return <h4 key={i} className="font-bold text-sm">{line.slice(3)}</h4>;
        if(line.startsWith('- ')) return <li key={i} className="ml-4">{line.slice(2)}</li>;
        if(line.startsWith('**') && line.endsWith('**')) return <p key={i} className="font-bold">{line.slice(2,-2)}</p>;
        if(line.trim() === '') return <br key={i} />;
        // Bold inline
        const parts = line.split(/(\*\*.*?\*\*)/g);
        return <p key={i}>{parts.map((p, j) =>
          p.startsWith('**') && p.endsWith('**') ? <strong key={j}>{p.slice(2,-2)}</strong> : p
        )}</p>;
      })}
    </div>
  );
}

function ChatPanel({ currentSection, molecule, phaseColor }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({behavior:'smooth'}); }, [messages]);

  const sendMessage = async () => {
    if(!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput('');
    setMessages(m => [...m, {role:'user', content:userMsg}]);
    setLoading(true);

    // Find section info
    let sectionTitle = "", lessonTitle = "", phaseName = "";
    PHASES.forEach(phase => {
      phase.lessons.forEach(lesson => {
        lesson.sections.forEach(sec => {
          if(sec.id === currentSection) {
            sectionTitle = sec.title;
            lessonTitle = lesson.title;
            phaseName = phase.title;
          }
        });
      });
    });

    const systemPrompt = `You are a friendly, encouraging tutor helping a high school student explore quantum molecular simulation. The student is using an interactive playground application.

They are currently in:
- Phase: ${phaseName}
- Lesson: ${lessonTitle}
- Section: ${sectionTitle}

Currently selected molecule: ${MOLECULES[molecule]?.name || molecule}

Guidelines:
- Use simple language appropriate for a high school student
- Reference what they can see in the current visualization when relevant
- Use Dirac notation (|0\u27E9, |1\u27E9) consistently
- Encourage experimentation with the interactive controls
- If they ask about advanced topics, give a brief honest answer and note it's beyond the current scope
- Be concise but thorough
- Use analogies when helpful
- Avoid using em-dashes`;

    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'anthropic-version': '2023-06-01',
          'x-api-key': typeof window !== 'undefined' && window.__ANTHROPIC_API_KEY__ || '',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 1000,
          system: systemPrompt,
          messages: [...messages, {role:'user',content:userMsg}].map(m=>({role:m.role,content:m.content})),
        })
      });
      const data = await response.json();
      const reply = data.content?.[0]?.text || "I'm having trouble connecting right now. Try again later!";
      setMessages(m => [...m, {role:'assistant', content:reply}]);
    } catch {
      setMessages(m => [...m, {role:'assistant', content:"I'm not able to connect to the AI tutor right now. This feature works best in the Claude artifact environment. In the meantime, feel free to explore the interactive visualizations!"}]);
    }
    setLoading(false);
  };

  if(!open) {
    return (
      <button onClick={() => setOpen(true)}
        className="fixed bottom-4 right-4 w-14 h-14 rounded-full bg-blue-600 text-white shadow-lg flex items-center justify-center text-2xl hover:bg-blue-700 z-50">
        {"\uD83D\uDCAC"}
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 w-96 h-[500px] bg-white rounded-2xl shadow-2xl border flex flex-col z-50">
      <div className="flex items-center justify-between p-3 border-b bg-blue-600 text-white rounded-t-2xl">
        <span className="font-medium text-sm">{"\uD83E\uDD16"} AI Tutor</span>
        <div className="flex gap-2">
          <button onClick={() => setOpen(false)} className="text-white/80 hover:text-white text-lg">{"\u2212"}</button>
          <button onClick={() => { setOpen(false); setMessages([]); }} className="text-white/80 hover:text-white text-lg">{"\u00D7"}</button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.length === 0 && (
          <div className="text-sm text-gray-400 text-center mt-8">
            <p>Hi! I'm your AI tutor. {"\uD83D\uDC4B"}</p>
            <p className="mt-1">Ask me anything about what you're learning!</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-2 rounded-lg text-sm ${msg.role === 'user' ? 'bg-blue-100 text-blue-900' : 'bg-gray-100 text-gray-800'}`}>
              {msg.role === 'assistant' ? <MarkdownRenderer text={msg.content} /> : msg.content}
            </div>
          </div>
        ))}
        {loading && <div className="text-sm text-gray-400 animate-pulse">Thinking...</div>}
        <div ref={messagesEndRef} />
      </div>
      <div className="p-3 border-t flex gap-2">
        <input value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Ask a question..."
          className="flex-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300" />
        <button onClick={sendMessage} disabled={loading}
          className="px-3 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50">{"\u2191"}</button>
      </div>
    </div>
  );
}

// ============================================================================
// PART 8: MAIN APP
// ============================================================================

export default function QuantumHamiltonianExplorer() {
  const [selectedMolecule, setSelectedMolecule] = useState("H2");
  const [bondLength, setBondLength] = useState(0.735);
  const [currentSection, setCurrentSection] = useState("1.1.1");
  const [completedSections, setCompletedSections] = useState(new Set());
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Track section completion on navigate
  const navigateToSection = useCallback((sectionId) => {
    setCompletedSections(prev => {
      const next = new Set(prev);
      next.add(currentSection);
      return next;
    });
    setCurrentSection(sectionId);
  }, [currentSection]);

  // Find current phase/lesson
  let currentPhase = null, currentLesson = null, currentSectionObj = null;
  PHASES.forEach(phase => {
    phase.lessons.forEach(lesson => {
      lesson.sections.forEach(section => {
        if(section.id === currentSection) {
          currentPhase = phase;
          currentLesson = lesson;
          currentSectionObj = section;
        }
      });
    });
  });

  const phaseColor = currentPhase?.color || "blue";
  const colors = PHASE_COLORS[phaseColor];

  // Phase completion percentages
  const getPhaseCompletion = (phase) => {
    let total = 0, done = 0;
    phase.lessons.forEach(l => l.sections.forEach(s => { total++; if(completedSections.has(s.id)) done++; }));
    return total > 0 ? Math.round(done/total*100) : 0;
  };

  // Navigation: next/prev section
  const allSections = PHASES.flatMap(p => p.lessons.flatMap(l => l.sections.map(s => s.id)));
  const currentIdx = allSections.indexOf(currentSection);
  const prevSection = currentIdx > 0 ? allSections[currentIdx-1] : null;
  const nextSection = currentIdx < allSections.length-1 ? allSections[currentIdx+1] : null;

  return (
    <div className="flex h-screen bg-gray-50 text-gray-800 overflow-hidden">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-72' : 'w-0'} transition-all duration-300 bg-white border-r overflow-hidden flex-shrink-0`}>
        <div className="w-72 h-full overflow-y-auto p-4">
          <h1 className="text-lg font-bold mb-4">Quantum Hamiltonian Explorer</h1>
          {PHASES.map(phase => {
            const pc = getPhaseCompletion(phase);
            const isCurrentPhase = currentPhase?.id === phase.id;
            const phColors = PHASE_COLORS[phase.color];
            return (
              <div key={phase.id} className="mb-3">
                <div className={`flex items-center gap-2 px-2 py-1 rounded-lg ${isCurrentPhase ? phColors.light : ''}`}>
                  <span>{phase.icon}</span>
                  <span className={`text-sm font-medium flex-1 ${phColors.text}`}>{phase.title}</span>
                  <span className="text-xs text-gray-400">{pc}%</span>
                </div>
                {phase.lessons.map(lesson => (
                  <div key={lesson.id} className="ml-4 mt-1">
                    <p className="text-xs font-medium text-gray-500 px-2">{lesson.icon} {lesson.title}</p>
                    {lesson.sections.map(section => {
                      const isCurrent = section.id === currentSection;
                      const isDone = completedSections.has(section.id);
                      return (
                        <button key={section.id} onClick={() => navigateToSection(section.id)}
                          className={`w-full text-left text-xs px-2 py-1 rounded flex items-center gap-1 ${isCurrent ? `${phColors.light} font-medium` : 'hover:bg-gray-50'}`}>
                          <span className="w-4">{isDone ? "\u2705" : "\u25CB"}</span>
                          <span className="truncate">{section.title}</span>
                        </button>
                      );
                    })}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <div className="bg-white border-b px-4 py-2 flex items-center gap-4 flex-shrink-0">
          <button onClick={() => setSidebarOpen(!sidebarOpen)}
            className="text-gray-500 hover:text-gray-700 text-lg">{sidebarOpen ? "\u2630" : "\u2630"}</button>

          {/* Molecule selector */}
          <div className="flex items-center gap-2 bg-gray-50 rounded-lg px-3 py-1">
            <span className="text-sm text-gray-500">Molecule:</span>
            {Object.entries(MOLECULES).map(([key, mol]) => (
              <button key={key} onClick={() => { setSelectedMolecule(key); if(key !== "H2") setBondLength(mol.equilibriumBondLength); }}
                className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${selectedMolecule === key ? `${colors.tab} text-white` : 'bg-white border hover:bg-gray-100'}`}>
                {mol.formula}
              </button>
            ))}
          </div>

          {/* Bond length (H2 only) */}
          {selectedMolecule === "H2" && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">R =</span>
              <input type="range" min="0.5" max="3.0" step="0.01" value={bondLength}
                onChange={e => setBondLength(parseFloat(e.target.value))} className="w-24" />
              <span className="text-xs font-mono">{bondLength.toFixed(2)} {"\u00C5"}</span>
            </div>
          )}
        </div>

        {/* Section tabs */}
        {currentLesson && (
          <div className="bg-white border-b px-4 py-1 flex gap-1 overflow-x-auto flex-shrink-0">
            {currentLesson.sections.map(section => {
              const isCurrent = section.id === currentSection;
              const isDone = completedSections.has(section.id);
              return (
                <button key={section.id} onClick={() => navigateToSection(section.id)}
                  className={`px-3 py-2 text-xs rounded-t-lg whitespace-nowrap border-b-2 transition-colors ${isCurrent ? `${colors.tab} text-white border-transparent` : isDone ? 'bg-green-50 border-green-300 text-green-700' : 'bg-gray-50 border-transparent hover:bg-gray-100'}`}>
                  {isDone && !isCurrent ? "\u2705 " : ""}{section.title}
                </button>
              );
            })}
          </div>
        )}

        {/* Content area */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-4xl mx-auto">
            {currentPhase && currentLesson && (
              <div className="mb-4">
                <div className={`text-sm ${colors.text} font-medium`}>
                  Phase {currentPhase.id}: {currentPhase.title} {"\u203A"} {currentLesson.title}
                </div>
                <h2 className="text-2xl font-bold mt-1">{currentSectionObj?.title}</h2>
              </div>
            )}

            <SectionContent sectionId={currentSection} molecule={selectedMolecule} bondLength={bondLength} setBondLength={setBondLength} phaseColor={phaseColor} />

            {/* Navigation buttons */}
            <div className="flex justify-between mt-8 pt-4 border-t">
              {prevSection ? (
                <button onClick={() => navigateToSection(prevSection)}
                  className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg">{"\u2190"} Previous</button>
              ) : <div />}
              {nextSection ? (
                <button onClick={() => navigateToSection(nextSection)}
                  className={`px-4 py-2 text-sm ${colors.tab} text-white hover:opacity-90 rounded-lg`}>Next {"\u2192"}</button>
              ) : <div />}
            </div>
          </div>
        </div>
      </div>

      {/* Chat panel */}
      <ChatPanel currentSection={currentSection} molecule={selectedMolecule} phaseColor={phaseColor} />
    </div>
  );
}
