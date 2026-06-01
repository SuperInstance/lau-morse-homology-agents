# lau-morse-homology-agents

**Morse theory applied to agent fitness landscapes — the topology of the landscape determines learning dynamics.**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

124 tests · 2,918 lines of Rust · 10 modules

---

## What This Does

This crate treats agent learning as gradient flow on a **fitness landscape**, then applies the full machinery of **Morse theory** to extract topological invariants from that landscape. The key insight:

> **Critical points of the fitness function = Nash equilibria.**  
> **The Morse index of a critical point = the number of unstable directions.**  
> **Morse inequalities bound the number of equilibria from below by the topology of strategy space.**

You get:
- A complete **Morse chain complex** from critical points and gradient flow lines
- **Morse inequalities** (weak and strong) bounding Nash equilibrium counts
- **Witten deformation** to isolate critical points and connect to quantum-mechanical tunneling
- **Morse–Smale complex** decomposition of strategy space into stable/unstable cells
- **Nash equilibrium finder** for 2×2 games with Morse-theoretic classification

---

## Key Idea

In classical Morse theory, a smooth function *f : M → ℝ* on a closed manifold satisfies the **Morse inequalities**:

$$M_k \geq \beta_k \quad \forall\, k$$

where *M_k* is the number of critical points of index *k* and *β_k* are the Betti numbers. When *f* is the **fitness function** on an agent's joint strategy space, these critical points are exactly the **Nash equilibria**. The Morse index tells you how many directions an equilibrium is unstable in — a saddle-point equilibrium with index 3 has 3 unstable directions, meaning 3 dimensions of strategy space where small perturbations cause the agent to drift away.

This crate implements every layer of this correspondence:

| Morse Theory Concept | Agent Theory Counterpart |
|---|---|
| Morse function *f* | Fitness / payoff function |
| Critical point (∇f = 0) | Nash equilibrium |
| Morse index λ | Number of unstable directions |
| Gradient flow | Learning dynamics (gradient ascent) |
| Stable manifold | Basin of attraction |
| Morse inequalities | Lower bounds on # of equilibria |
| Witten deformation | Reward shaping / potential-based shaping |

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-morse-homology-agents = "0.1"
```

Requires Rust 2021 edition. Dependencies: `nalgebra` (with serde), `serde`, `serde_json`.

---

## Quick Start

```rust
use lau_morse_homology_agents::*;

// 1. Define a 2×2 game (Prisoner's Dilemma)
use nalgebra::DMatrix;
let row_payoffs = DMatrix::from_row_slice(2, 2, &[-1.0, -3.0, 0.0, -2.0]);
let col_payoffs = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, -3.0, -2.0]);
let landscape = FitnessLandscape::from_two_player_game(&row_payoffs, &col_payoffs);

// 2. Find Nash equilibria
let nash = landscape.find_nash_equilibria_2x2();
println!("Nash equilibria: {:?}", nash);
// → [(1.0, 0.0, 1.0, 0.0)]  i.e. (Defect, Defect)

// 3. Count equilibria from topology
let betti = vec![1, 0, 1]; // Betti numbers of S²
let counter = NashEquilibriumCounter::count_from_topology(&betti);
let lower_bound = NashEquilibriumCounter::lower_bound_weak(&betti);
println!("At least {} equilibria", lower_bound);

// 4. Run gradient flow (agent learning)
let gf = GradientFlow::new().with_learning_rate(0.1).with_momentum(0.9);
let traj = gf.flow_polynomial_1d(2.0, &[0.0, 0.0, -1.0]);
println!("Converged: {} in {} steps", traj.converged, traj.steps);
```

---

## API Reference

### `MorseFunction`
The core abstraction. A smooth function on agent state space with non-degenerate critical points.

```rust
let f = MorseFunction::from_polynomial(1, vec![1.0, 0.0, -1.0]); // 1 - x²
let val = f.evaluate_1d(0.5);     // 0.75
let grad = f.gradient_1d(0.5);    // -1.0
let hess = f.hessian_1d(0.5);     // -2.0
```

### `CriticalPoint`
A point where ∇f = 0, classified by Morse index.

```rust
let cp = CriticalPoint {
    position: DVector::from_vec(vec![0.0]),
    value: 1.0,
    index: 1,
    hessian: DMatrix::from_vec(1, 1, vec![-2.0]),
};
assert_eq!(cp.point_type(), CriticalPointType::Maximum);
assert!(cp.is_non_degenerate());
```

### `MorseIndex`
Computes the index from a Hessian matrix.

```rust
let hessian = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, 2.0]);
let mi = MorseIndex::from_hessian(&hessian);
assert_eq!(mi.index, 1);         // one negative eigenvalue
assert_eq!(mi.co_index(), 1);    // one positive eigenvalue
assert_eq!(mi.signature(), (1, 0, 1));
```

### `MorseChainComplex`
Builds the chain complex from critical points and gradient flow lines.

```rust
let mut complex = MorseChainComplex::new();
complex.add_generator(MorseChainGenerator { id: 0, index: 0, position: ..., value: 0.0 });
complex.add_generator(MorseChainGenerator { id: 1, index: 1, position: ..., value: 1.0 });
complex.add_boundary(GradientFlowLine { source_id: 1, target_id: 0, count: 1 });
let homology = complex.compute_homology(); // Returns (Betti numbers, torsion)
```

### `MorseInequalities`
Verifies both weak and strong Morse inequalities, computes Euler characteristic and Poincaré polynomial.

```rust
let mi = MorseInequalities::new(
    vec![1, 2, 1],  // Morse polynomial: critical points per index
    vec![1, 0, 1],  // Betti numbers
);
let weak_ok = mi.verify_weak_inequalities();  // [true, true, true]
let strong_ok = mi.verify_strong_inequalities(); // [true, true]
let euler = mi.euler_characteristic();         // 1 - 0 + 1 = 2
```

### `MorseLemmaCoordinates`
The Morse lemma: near a critical point, the function has a canonical normal form.

```rust
let coords = MorseLemmaCoordinates::from_hessian(position, value, &hessian);
let normal_val = coords.evaluate_normal_form(&y); // f(p) - y₁² + y₂²
let recovered = coords.evaluate(&x);              // in original coordinates
```

### `MorseSmaleComplex`
Decomposes the manifold into cells from intersecting stable/unstable manifolds.

```rust
let mut msc = MorseSmaleComplex::new(2);
msc.add_cell(MorseSmaleCell { id: 0, source_index: 0, target_index: 1, dimension: 0, center: ... });
let betti = msc.betti_numbers();      // [1, 0, 1]
let euler = msc.euler_characteristic(); // 2
```

### `WittenDeformation`
Twists the Laplacian by *e^{-tf}* to isolate critical points.

```rust
let mut witten = WittenDeformation::new(vec![0.0, 1.0, 2.0], vec![0, 1, 0], 2);
witten.with_t(10.0);
let gap = witten.spectral_gap(0, 1);          // ≈ 10·(1-0) = 10
let tunneling = witten.tunneling_amplitude(0, 1); // ≈ e^{-10}
```

### `FitnessLandscape`
Agent fitness as a Morse function on joint strategy space.

```rust
let fl = FitnessLandscape::from_two_player_game(&row, &col);
let nash = fl.find_nash_equilibria_2x2();
let fitness = fl.evaluate_fitness(&strategy);
let grad = fl.fitness_gradient(&strategy);
let eq_type = fl.classify_equilibrium(&pos, &hessian);
```

### `GradientFlow`
Agent learning as gradient ascent on the fitness landscape.

```rust
let gf = GradientFlow::new()
    .with_learning_rate(0.01)
    .with_momentum(0.9);
let traj = gf.flow(&start, |x| gradient_fn(x));
println!("Converged: {} in {} steps", traj.converged, traj.steps);
```

### `NashEquilibriumCounter`
Counts equilibria via Morse inequalities.

```rust
let counter = NashEquilibriumCounter::new(landscape);
let lower = NashEquilibriumCounter::lower_bound_weak(&[1, 0, 1]); // 2
let upper = counter.upper_bound_from_strong(&[1, 0, 1]);          // from strong inequalities
```

---

## How It Works

The crate is structured as a 10-module pipeline:

```
MorseFunction  →  CriticalPoint  →  MorseIndex  →  MorseLemmaCoordinates
       ↓                                    ↓
FitnessLandscape  →  GradientFlow  →  MorseSmaleComplex
       ↓                                    ↓
NashEquilibrium  ←  MorseInequalities  ←  MorseChainComplex  ←  WittenDeformation
```

1. **`MorseFunction`**: Defines the function, finds critical points (∇f = 0), checks non-degeneracy.
2. **`MorseIndex`**: Eigendecomposes the Hessian at each critical point to compute the index.
3. **`MorseLemmaCoordinates`**: Applies the Morse lemma to get normal form coordinates near each critical point.
4. **`MorseChainComplex`**: Builds the chain complex *C_k* from critical points, with boundary maps from gradient flow lines.
5. **`MorseInequalities`**: Verifies that *M_k ≥ β_k* (weak) and the stronger polynomial inequalities.
6. **`MorseSmaleComplex`**: Decomposes strategy space into cells from transversally intersecting stable/unstable manifolds.
7. **`WittenDeformation`**: Applies the Witten twist *d_t = e^{-tf}de^{tf}* to spectrally isolate critical points.
8. **`FitnessLandscape`**: Maps game theory (payoff matrices) to Morse theory (fitness function).
9. **`GradientFlow`**: Implements gradient ascent with momentum and adaptive learning rates.
10. **`NashEquilibrium`**: Uses Morse inequalities to bound and classify equilibria.

---

## The Math

### Morse Theory in 60 Seconds

Given a smooth function *f : Mⁿ → ℝ* on a closed *n*-manifold:

1. **Critical point**: *p* where ∇f(*p*) = 0
2. **Non-degenerate**: Hessian *H_f(p)* has no zero eigenvalues
3. **Morse index**: *λ(p)* = number of negative eigenvalues of *H_f(p)*
4. **Morse lemma**: Near *p*, coordinates exist where *f = f(p) - y₁² - ⋯ - y_λ² + y_{λ+1}² + ⋯ + y_n²*

The **Morse inequalities** relate critical points to homology:

- **Weak**: *M_k ≥ β_k* for all *k*
- **Strong**: *M₀ - M₁ + ⋯ + (-1)^k M_k ≥ β₀ - β₁ + ⋯ + (-1)^k β_k* for all *k*
- **Equality**: *Σ(-1)^k M_k = Σ(-1)^k β_k = χ(M)* (Euler characteristic)

### Witten Deformation

The **Witten-deformed Laplacian** Δ_t = (*d_t + d_t**)², where *d_t = e^{-tf}de^{tf}*, has the property that for large *t*:

- Low-lying eigenforms concentrate near critical points
- The spectral gap between "small" and "large" eigenvalues grows as ~*t*
- Tunneling amplitudes between critical points *p, q* decay as *e^{-t|f(p)-f(q)|}*

This gives a **supersymmetric quantum mechanics** interpretation: critical points are ground states, gradient flow lines are instanton tunneling paths.

### Application to Agent Theory

For an *n*-player game with joint strategy space *S*:

- The **fitness function** *F : S → ℝ* is the Morse function
- **Critical points** of *F* are **Nash equilibria** (no player can improve by deviating)
- The **Morse index** of an equilibrium counts unstable directions (dimensions where perturbation causes drift)
- **Gradient flow** on *F* models **learning dynamics** (fictitious play, gradient ascent)
- **Morse inequalities** bound the **minimum number of Nash equilibria** from the topology of *S*

For example, if the strategy space is an *n*-sphere (β₀ = 1, β_n = 1), there are at least 2 Nash equilibria — corresponding to the minimum and maximum of fitness.

---

## Module Overview

| Module | Tests | Key Types |
|--------|-------|-----------|
| `morse_function` | 13 | `MorseFunction`, `CriticalPoint`, `CriticalPointType` |
| `morse_index` | 12 | `MorseIndex` |
| `morse_lemma` | 9 | `MorseLemmaCoordinates` |
| `morse_inequalities` | 15 | `MorseInequalities`, `BettiNumbers` |
| `morse_homology` | 13 | `MorseChainComplex`, `MorseHomology` |
| `morse_smale` | 12 | `MorseSmaleComplex`, `MorseSmaleCell` |
| `witten` | 15 | `WittenDeformation` |
| `fitness_landscape` | 12 | `FitnessLandscape`, `AgentState` |
| `gradient_flow` | 10 | `GradientFlow`, `FlowTrajectory` |
| `nash_equilibrium` | 13 | `NashEquilibriumCounter`, `EquilibriumInfo` |

---

## License

MIT
