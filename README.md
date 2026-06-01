# lau-morse-homology-agents

**Morse theory applied to agent fitness landscapes — the topology of the landscape determines learning dynamics.**

A Rust library that connects differential topology (Morse theory, handle decompositions, Morse–Smale complexes) to multi-agent learning and game theory. Critical points of a fitness function reveal the structure of possible agent behaviors, and Morse theory tells you exactly how many critical points *must* exist, what types they are, and how gradient-flow learning trajectories connect them.

[![124 tests passing](https://img.shields.io/badge/tests-124%20passing-brightgreen)]()

---

## Table of Contents

- [What This Does](#what-this-does)
- [Key Idea](#key-idea)
- [Install](#install)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [MorseFunction](#morsefunction)
  - [CriticalPoint](#criticalpoint)
  - [CriticalPointType](#criticalpointtype)
  - [MorseLemmaCoordinates](#morselemmacoordinates)
  - [MorseIndex](#morseindex)
  - [MorseInequalities](#morseinequalities)
  - [BettiNumbers](#bettinumbers)
  - [MorseChainComplex](#morsechaincomplex)
  - [MorseHomology](#morsehomology-groups)
  - [MorseSmaleComplex](#morsesmalecomplex)
  - [WittenDeformation](#wittendeformation)
  - [FitnessLandscape](#fitnesslandscape)
  - [AgentState](#agentstate)
  - [GradientFlow](#gradientflow)
  - [NashEquilibriumCounter](#nashequilibriumcounter)
  - [EquilibriumInfo](#equilibriuminfo)
- [How It Works](#how-it-works)
- [The Math](#the-math)
  - [Morse Functions and Critical Points](#morse-functions-and-critical-points)
  - [Morse Lemma](#morse-lemma)
  - [Morse Index and Handle Decomposition](#morse-index-and-handle-decomposition)
  - [Morse Inequalities](#morse-inequalities-1)
  - [Morse Homology](#morse-homology)
  - [Morse–Smale Complex](#morse-smale-complex)
  - [Witten Deformation](#witten-deformation-1)
  - [Fitness Landscapes and Game Theory](#fitness-landscapes-and-game-theory)
- [License](#license)

---

## What This Does

This library provides:

1. **Morse function analysis** — define fitness functions on agent strategy spaces, find their critical points (minima, maxima, saddles), and classify them by the Morse index.
2. **Morse lemma coordinates** — near every critical point, compute the canonical normal form that reveals the local landscape geometry.
3. **Morse inequalities** — given the Betti numbers (homology) of your strategy space, compute lower bounds on the number of critical points that *must* exist. No amount of game design can eliminate them.
4. **Morse homology** — build a chain complex from critical points connected by gradient-flow lines, compute homology groups, and verify ∂² = 0.
5. **Morse–Smale complexes** — decompose the strategy space into cells from the intersections of stable and unstable manifolds, and compute cell-complex homology.
6. **Witten deformation** — deform the Laplacian by a parameter `t` to isolate critical points; use tunneling amplitudes for reward shaping.
7. **Fitness landscapes** — model multi-agent games as Morse functions; find and classify Nash equilibria using topology.
8. **Gradient flow** — simulate agent learning as gradient ascent on the fitness landscape, with momentum and adaptive learning rates.
9. **Nash equilibrium counting** — use Morse inequalities and the Wilson oddness theorem to bound the number of Nash equilibria from topology.

---

## Key Idea

**Morse theory** is the observation that a smooth function `f : M → ℝ` on a manifold encodes the topology of `M` through its critical points (where ∇f = 0). For an *agent* navigating a fitness landscape:

- **Critical points = equilibria** — places where the agent has no incentive to move (Nash equilibria, evolutionary stable strategies).
- **Morse index = instability** — the number of "descending" directions at a critical point tells you how many ways an agent can escape. Index 0 = stable minimum, index n = unstable maximum, intermediate = saddle.
- **Gradient flow lines = learning trajectories** — the paths agents follow when doing gradient ascent connect critical points, forming the boundary maps of a chain complex.
- **Morse inequalities = impossibility results** — the topology of the strategy space forces a minimum number of equilibria. You can't design a game with fewer Nash equilibria than the sum of Betti numbers.

This library makes all of that computable.

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-morse-homology-agents = "0.1.0"
```

Or use `cargo add`:

```bash
cargo add lau-morse-homology-agents
```

Requires Rust 2021 edition or later. Dependencies: `nalgebra` (with serde), `serde`, `serde_json`.

---

## Quick Start

### Find critical points of a polynomial fitness function

```rust
use lau_morse_homology_agents::MorseFunction;

// f(x) = x³/3 - x  (classic double-well-like potential)
let mut f = MorseFunction::from_polynomial(1, vec![0.0, -1.0, 0.0, 1.0/3.0]);
let count = f.find_critical_points_1d((-5.0, 5.0), 30);
println!("Found {} critical points", count);
for cp in &f.critical_points {
    println!("  x={:.3} f={:.3} index={} type={:?}",
        cp.position[0], cp.value, cp.index, cp.point_type());
}
```

### Compute Morse inequalities for a sphere

```rust
use lau_morse_homology_agents::MorseInequalities;

// S² has Betti numbers [1, 0, 1], and a perfect Morse function has M = [1, 0, 1]
let mi = MorseInequalities::new(vec![1, 0, 1], vec![1, 0, 1]);
println!("Perfect? {}", mi.is_perfect()); // true
println!("χ = {}", mi.euler_from_morse()); // 2
println!("Weak inequalities hold? {:?}", mi.verify_weak_inequalities()); // [true, true, true]
```

### Find Nash equilibria in a 2×2 game

```rust
use lau_morse_homology_agents::{FitnessLandscape, NashEquilibriumCounter};
use nalgebra::DMatrix;

// Prisoner's Dilemma
let row = DMatrix::from_row_slice(2, 2, &[-1.0, -3.0, 0.0, -2.0]);
let col = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, -3.0, -2.0]);
let fl = FitnessLandscape::from_two_player_game(&row, &col);
let mut counter = NashEquilibriumCounter::new(fl);
let n = counter.find_equilibria_2x2();
println!("Found {} Nash equilibria", n);
```

### Run gradient flow (agent learning)

```rust
use lau_morse_homology_agents::GradientFlow;

let gf = GradientFlow::new()
    .with_learning_rate(0.1)
    .with_momentum(0.9);
let traj = gf.flow_polynomial_1d(5.0, &[0.0, 0.0, -1.0]);
println!("Converged: {} in {} steps to x={:.4}",
    traj.converged, traj.steps, traj.final_position[0]);
```

### Compute Morse homology

```rust
use lau_morse_homology_agents::{MorseChainComplex, MorseHomology as MorseHomologyGroups};
use lau_morse_homology_agents::morse_homology::{MorseChainGenerator, GradientFlowLine};
use nalgebra::DVector;

// Circle: 1 minimum (index 0), 1 maximum (index 1)
let mut cc = MorseChainComplex::new();
cc.add_generator(MorseChainGenerator { id: 0, index: 0, position: DVector::from_vec(vec![0.0]), value: 0.0 });
cc.add_generator(MorseChainGenerator { id: 1, index: 1, position: DVector::from_vec(vec![3.14]), value: 1.0 });
cc.add_boundary(GradientFlowLine { source_id: 1, target_id: 0, count: 0 }); // Z/2: two flow lines cancel

let mh = MorseHomologyGroups::from_chain_complex(&cc);
println!("Betti numbers: {:?}, χ = {}", mh.betti, mh.euler_characteristic());
// Betti: [1, 1], χ = 0
```

---

## API Reference

### MorseFunction

```rust
pub struct MorseFunction {
    pub dimension: usize,
    pub critical_points: Vec<CriticalPoint>,
    pub coefficients: Vec<f64>,
}
```

A Morse function on agent state space. Stores polynomial coefficients for evaluation and a list of known critical points.

| Method | Description |
|--------|-------------|
| `new(dimension)` | Create with given dimension |
| `from_polynomial(dimension, coefficients)` | Create from polynomial coeffs `[c₀, c₁, ...]` for `f(x) = Σ cᵢxⁱ` |
| `evaluate_1d(x)` | Evaluate 1D polynomial |
| `gradient_1d(x)` | Gradient of 1D polynomial |
| `hessian_1d(x)` | Second derivative (1D) |
| `find_critical_points_1d(range, num_starts)` | Newton's method to find all critical points |
| `evaluate_quadratic(x, A, b, c)` | Evaluate `f(x) = xᵀAx + bᵀx + c` |
| `gradient_quadratic(A, b, x)` | Gradient of quadratic form |
| `hessian_quadratic(A)` | Hessian of quadratic form |
| `verify_non_degenerate()` | Check all critical points have invertible Hessian |
| `add_critical_point(cp)` | Manually add a critical point |
| `critical_points_of_index(k)` | Filter by Morse index |
| `morse_polynomial()` | `[M₀, M₁, ...]` where `Mₖ` = number of critical points with index `k` |
| `total_critical_points()` | Count of all critical points |

### CriticalPoint

```rust
pub struct CriticalPoint {
    pub position: DVector<f64>,
    pub value: f64,
    pub index: usize,
    pub hessian: DMatrix<f64>,
}
```

A critical point of the Morse function.

| Method | Description |
|--------|-------------|
| `point_type()` | Classify as `Minimum`, `Maximum`, or `Saddle` |
| `is_non_degenerate()` | Check Hessian has no zero eigenvalues |
| `stability()` | Ratio of negative to positive eigenvalue magnitudes |

### CriticalPointType

```rust
pub enum CriticalPointType { Minimum, Maximum, Saddle }
```

Classification from `(index, dim)`: index 0 → Minimum, index = dim → Maximum, else → Saddle.

### MorseLemmaCoordinates

```rust
pub struct MorseLemmaCoordinates {
    pub critical_point: DVector<f64>,
    pub critical_value: f64,
    pub morse_index: usize,
    pub dimension: usize,
    pub transformation: DMatrix<f64>,
    pub eigenvalues: DVector<f64>,
}
```

The canonical coordinate system near a critical point from the Morse Lemma.

| Method | Description |
|--------|-------------|
| `from_hessian(position, value, hessian)` | Compute eigendecomposition and transformation |
| `evaluate_normal_form(y)` | `f(p) - y₁² - ... - yλ² + yλ₊₁² + ...` |
| `to_morse_coords(x)` | Transform `x → y = Qᵀ(x - p)` |
| `from_morse_coords(y)` | Transform `y → x = Qy + p` |
| `verify(f_actual, x)` | Compare actual vs. normal form |
| `descending_directions()` | Eigenvectors with negative eigenvalues (unstable directions) |
| `ascending_directions()` | Eigenvectors with positive eigenvalues (stable directions) |
| `isotropy_dimension()` | Dimension of the stable manifold |

### MorseIndex

```rust
pub struct MorseIndex {
    pub index: usize,
    pub dimension: usize,
    pub eigenvalues: DVector<f64>,
    pub eigenvectors: DMatrix<f64>,
}
```

The Morse index = number of negative Hessian eigenvalues.

| Method | Description |
|--------|-------------|
| `from_hessian(hessian)` | Compute from eigendecomposition |
| `co_index()` | `dim - index` (stable manifold dimension) |
| `nullity()` | Number of zero eigenvalues (0 for non-degenerate) |
| `is_non_degenerate()` | `nullity() == 0` |
| `signature()` | `(neg_count, zero_count, pos_count)` |
| `unstable_manifold_dim()` | Equals the Morse index |
| `stable_manifold_dim()` | Equals the co-index |
| `unstable_basis()` | Matrix of negative-eigenvalue eigenvectors |
| `stable_basis()` | Matrix of positive-eigenvalue eigenvectors |
| `conley_index_homology()` | Homology of the Conley index (Sᵏ for index k) |

### MorseInequalities

```rust
pub struct MorseInequalities {
    pub morse_polynomial: Vec<usize>,
    pub betti_numbers: BettiNumbers,
}
```

The weak and strong Morse inequalities relating critical point counts to homology.

| Method | Description |
|--------|-------------|
| `new(morse_polynomial, betti)` | Create from Mₖ values and Betti numbers |
| `verify_weak_inequalities()` | Check `Mₖ ≥ βₖ` for all k |
| `verify_strong_inequalities()` | Check alternating sums |
| `euler_from_morse()` | `χ = Σ(-1)ᵏMₖ` |
| `verify_euler_characteristic()` | Morse Euler = Betti Euler? |
| `deficiencies()` | `dₖ = Mₖ - βₖ` (non-negative) |
| `is_perfect()` | All deficiencies zero? |

### BettiNumbers

```rust
pub struct BettiNumbers { pub betti: Vec<usize> }
```

| Method | Description |
|--------|-------------|
| `new(betti)` | Create from `β₀, β₁, ...` |
| `euler_characteristic()` | `χ = Σ(-1)ᵏβₖ` |
| `poincare_polynomial_coeffs()` | Same as `betti` |
| `total_rank()` | `Σβₖ` |

### MorseChainComplex

```rust
pub struct MorseChainComplex {
    pub generators: Vec<MorseChainGenerator>,
    pub boundary_maps: Vec<GradientFlowLine>,
    pub max_index: usize,
}
```

The chain complex `Cₖ` generated by critical points of index `k`, with boundary from gradient flow lines.

| Method | Description |
|--------|-------------|
| `new()` | Empty complex |
| `add_generator(gen)` | Add a critical point as a chain generator |
| `add_boundary(line)` | Add a gradient flow line as boundary data |
| `generators_of_index(k)` | Filter generators by Morse index |
| `boundary_matrix(k)` | Integer matrix for `∂ₖ: Cₖ → Cₖ₋₁` |
| `compute_homology()` | Betti numbers via Z/2 rank computation |
| `verify_boundary_squared_zero()` | Check `∂² = 0` |
| `torsion_ranks()` | Placeholder for Z/2 torsion |

### MorseHomology (MorseHomologyGroups)

```rust
pub struct MorseHomology { pub betti: Vec<usize>, pub max_index: usize }
```

| Method | Description |
|--------|-------------|
| `from_chain_complex(cc)` | Compute from chain complex |
| `euler_characteristic()` | `Σ(-1)ᵏβₖ` |
| `total_dimension()` | `Σβₖ` |

### MorseSmaleComplex

```rust
pub struct MorseSmaleComplex {
    pub cells: Vec<MorseSmaleCell>,
    pub adjacency: Vec<(usize, usize, usize)>,
    pub dimension: usize,
    pub critical_counts: Vec<usize>,
}
```

Cell decomposition from transverse intersections of stable/unstable manifolds.

| Method | Description |
|--------|-------------|
| `new(dimension)` | Empty complex |
| `add_cell(cell)` | Add a cell |
| `add_adjacency(from, to, codim)` | Add boundary relation |
| `cells_of_dimension(d)` | Filter cells by dim |
| `check_transversality()` | Verify codim ≤ 1 |
| `num_cells(k)` | Count k-cells |
| `boundary_matrix(k)` | CW boundary matrix |
| `compute_homology()` | Z/2 Betti numbers |
| `verify(expected_betti)` | Transversality + homology match |
| `stable_manifold_cells(cp_id)` | Cells flowing to a critical point |
| `unstable_manifold_cells(cp_id)` | Cells flowing from a critical point |
| `f_vector()` | `[f₀, f₁, ...]` cell counts |

### WittenDeformation

```rust
pub struct WittenDeformation {
    pub critical_values: Vec<f64>,
    pub critical_indices: Vec<usize>,
    pub t: f64,
    pub dimension: usize,
}
```

The Witten-deformed Laplacian `Δₜ = (dₜ + dₜ*)²` that isolates critical points for large `t`.

| Method | Description |
|--------|-------------|
| `new(critical_values, critical_indices, dimension)` | Create deformation |
| `with_t(t)` | Set deformation parameter |
| `deformed_derivative_0(grad_f, dφ)` | `dφ - t·∇f·φ` on 0-forms |
| `deformed_laplacian_0(Δφ, |∇f|², tr(H))` | Witten Laplacian at a point |
| `critical_point_eigenvalue(hess_eigs, index)` | Estimate eigenvalue near a critical point |
| `tunneling_amplitude(p_val, q_val)` | `exp(-t|f(p) - f(q)|)` |
| `boundary_tunneling_amplitudes()` | All amplitudes between consecutive-index critical points |
| `witten_boundary_matrix(k)` | Approximate boundary matrix |
| `shaped_reward(R, ∇f, ∇V)` | Reward shaping via Witten deformation |
| `effective_potential(|∇f|², Δf)` | `½|∇f|² - Δf/(2t)` |
| `is_large_t()` | Check if `t` is big enough for the approximation |
| `num_critical_points()` | Count |

### FitnessLandscape

```rust
pub struct FitnessLandscape {
    pub num_agents: usize,
    pub strategy_dim: usize,
    pub morse_function: MorseFunction,
    pub payoff_matrix: Vec<f64>,
    pub strategy_counts: Vec<usize>,
}
```

A multi-agent fitness landscape modeled as a Morse function on joint strategy space.

| Method | Description |
|--------|-------------|
| `new(num_agents, strategy_dim)` | Create landscape |
| `from_two_player_game(A, B)` | From 2-player payoff matrices |
| `evaluate_fitness(strategy)` | Evaluate fitness at a joint strategy |
| `fitness_gradient(strategy)` | Gradient of fitness |
| `find_nash_equilibria_2x2()` | Find all Nash equilibria for 2×2 games (pure + mixed) |
| `classify_equilibrium(pos, hessian)` | Morse-type classification |
| `regret(current, best_response)` | Fitness gap |
| `is_morse()` | All critical points non-degenerate? |
| `total_dimension()` | `num_agents × strategy_dim` |

### AgentState

```rust
pub struct AgentState {
    pub strategy: DVector<f64>,
    pub id: usize,
    pub fitness: f64,
}
```

| Method | Description |
|--------|-------------|
| `new(id, strategy)` | Create |
| `distance_to(other)` | Euclidean distance |
| `dimension()` | Strategy space dimension |

### GradientFlow

```rust
pub struct GradientFlow {
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub momentum: f64,
    pub adaptive: bool,
}
```

Gradient ascent on the fitness landscape, modeling agent learning.

| Method | Description |
|--------|-------------|
| `new()` | Default: lr=0.01, 1000 iters, tol=1e-8 |
| `with_learning_rate(lr)` | Builder pattern |
| `with_momentum(m)` | Builder pattern |
| `flow(start, gradient_fn)` | Run gradient ascent with closure |
| `flow_polynomial_1d(start, coeffs)` | Ascent on 1D polynomial |
| `flow_quadratic(start, A, b)` | Ascent on quadratic form |
| `stable_manifold_estimate(cp, grad_fn, samples, radius)` | Basin of attraction |
| `classify_critical_point(point, grad_fn, directions, perturbation)` | Classify via flow |

### NashEquilibriumCounter

```rust
pub struct NashEquilibriumCounter {
    pub landscape: FitnessLandscape,
    pub equilibria: Vec<EquilibriumInfo>,
}
```

Count and classify Nash equilibria using Morse theory.

| Method | Description |
|--------|-------------|
| `new(landscape)` | Create counter |
| `count_from_topology(betti)` | Morse inequalities from Betti numbers |
| `lower_bound_weak(betti)` | `Σβₖ` |
| `lower_bound_strong_k(betti, k)` | k-th strong inequality bound |
| `find_equilibria_2x2()` | Find and classify all equilibria in a 2×2 game |
| `num_stable()` | Count ESS-like equilibria |
| `num_unstable()` | Count saddle/max equilibria |
| `morse_polynomial()` | Critical point distribution by index |
| `verify_morse_inequalities(betti)` | Check `Mₖ ≥ βₖ` |
| `wilson_oddness(count)` | Nondegenerate games have odd number of equilibria |
| `euler_characteristic_product(simplices)` | Euler char of product of simplices |

### EquilibriumInfo

```rust
pub struct EquilibriumInfo {
    pub position: DVector<f64>,
    pub equilibrium_type: CriticalPointType,
    pub morse_index: usize,
    pub fitness: f64,
    pub is_stable: bool,
}
```

---

## How It Works

The library follows the classical Morse-theoretic pipeline, adapted to multi-agent systems:

1. **Define the fitness function** as a `MorseFunction` on the joint strategy space. This can be a polynomial (1D), a quadratic form (multi-D), or a game payoff matrix.

2. **Find critical points** — where the gradient vanishes. In 1D, Newton's method from multiple starting points. For games, support enumeration or direct computation.

3. **Compute the Morse index** at each critical point by eigendecomposing the Hessian. The index tells you the dimension of the unstable manifold — how many ways agents can "escape" this equilibrium.

4. **Apply the Morse lemma** — near each critical point, there's a canonical coordinate system where the function looks like `f(p) - y₁² - ... - yλ² + yλ₊₁² + ...`. This tells you the local landscape geometry.

5. **Check the Morse inequalities** — the number of critical points of each index `Mₖ` must satisfy `Mₖ ≥ βₖ` (weak) and the alternating-sum versions (strong). This constrains game design: you can't have fewer equilibria than topology demands.

6. **Build the Morse chain complex** — critical points become chain generators, gradient flow lines become boundary maps. The resulting homology `Hₖ(C, ∂)` recovers the manifold's topology.

7. **Construct the Morse–Smale complex** — decompose strategy space into cells from intersecting stable/unstable manifolds. This gives a CW decomposition with the same homology.

8. **Apply Witten deformation** — for large `t`, the Witten-deformed Laplacian isolates critical points, with exponentially small tunneling amplitudes `e^{-t|f(p)-f(q)|}` between them. This provides reward shaping for learning.

9. **Connect to game theory** — critical points of the fitness function are Nash equilibria. Their Morse index classifies stability. The Wilson oddness theorem guarantees an odd number of equilibria for nondegenerate games.

---

## The Math

### Morse Functions and Critical Points

A **Morse function** is a smooth function `f : M → ℝ` on a manifold `M` such that every critical point (∇f = 0) is **non-degenerate** (the Hessian matrix is invertible). Non-degeneracy means the Hessian has no zero eigenvalues.

For an n-dimensional manifold, critical points are classified by the **Morse index** `λ` — the number of negative eigenvalues of the Hessian. This equals the dimension of the **unstable manifold** of the negative gradient flow.

- **λ = 0**: Local minimum. All directions are ascending. Stable equilibrium.
- **λ = n**: Local maximum. All directions are descending. Unstable equilibrium.
- **0 < λ < n**: Saddle point. Mixed stability.

The **Morse polynomial** is `M(t) = Σ Mₖ tᵏ` where `Mₖ` is the number of critical points of index `k`.

### Morse Lemma

**Theorem (Morse Lemma):** Near a non-degenerate critical point `p` with Morse index `λ`, there exist local coordinates `(y₁, ..., yₙ)` such that:

```
f(y) = f(p) - y₁² - y₂² - ... - yλ² + yλ₊₁² + ... + yₙ²
```

This means every critical point looks like a standard quadratic in appropriate coordinates. The coordinate transformation is obtained from the eigendecomposition of the Hessian.

### Morse Index and Handle Decomposition

The Morse index determines how the manifold is built up:

- A critical point of index `λ` corresponds to attaching a **λ-handle** `Dλ × Dⁿ⁻λ`.
- Starting from the empty set, attaching handles in order of increasing index gives a **handle decomposition** of `M`.
- The **stable manifold** of a critical point (points that flow *to* it) has dimension `n - λ`.
- The **unstable manifold** (points that flow *from* it) has dimension `λ`.

The Conley index of a critical point of index `k` is the homotopy type of `Sᵏ` (the k-sphere).

### Morse Inequalities

**Weak Morse Inequalities:** For all `k`:

```
Mₖ ≥ βₖ
```

where `βₖ = rank Hₖ(M)` is the k-th Betti number.

**Strong Morse Inequalities:** For all `k`:

```
Mₖ - Mₖ₋₁ + Mₖ₋₂ - ... + (-1)ᵏM₀ ≥ βₖ - βₖ₋₁ + βₖ₋₂ - ... + (-1)ᵏβ₀
```

**Euler Characteristic Equality:**

```
χ(M) = Σ(-1)ᵏMₖ = Σ(-1)ᵏβₖ
```

A Morse function is **perfect** if `Mₖ = βₖ` for all `k` (all inequalities are tight). The **deficiency** `dₖ = Mₖ - βₖ ≥ 0` measures how far from perfect the function is.

Examples:
- **S²** (2-sphere): Betti `[1, 0, 1]`. Perfect Morse function: 1 minimum + 1 maximum. χ = 2.
- **T²** (torus): Betti `[1, 2, 1]`. Perfect: 1 min + 2 saddles + 1 max. χ = 0.
- **S¹** (circle): Betti `[1, 1]`. χ = 0.

### Morse Homology

The Morse chain complex is:

```
... → Cₖ₊₁ →∂ₖ₊₁ Cₖ →∂ₖ Cₖ₋₁ → ...
```

where:
- `Cₖ` = free abelian group generated by critical points of index `k`
- `∂ₖ : Cₖ → Cₖ₋₁` counts gradient flow lines between critical points of index `k` and `k-1`, with signs from orientations

The key property is **∂² = 0** (broken flow lines cancel in pairs). Then:

```
Hₖᴹᵒʳˢᵉ = ker(∂ₖ) / im(∂ₖ₊₁) ≅ Hₖ(M)
```

Morse homology is isomorphic to singular homology — the critical points and flow lines recover the topology.

This library computes homology over **Z/2** (no sign issues) using Gaussian elimination on the boundary matrices.

### Morse–Smale Complex

When the gradient flow satisfies the **Smale transversality condition** (stable and unstable manifolds of distinct critical points intersect transversally), the intersections of stable and unstable manifolds give a **CW decomposition** of `M`.

Each cell corresponds to a flow line between two critical points. The k-cells correspond to critical points of index `k`. The boundary maps of this CW complex reproduce the Morse boundary operator.

The **f-vector** `[f₀, f₁, ..., fₙ]` counts cells of each dimension. The resulting cell-complex homology matches Morse homology.

### Witten Deformation

Edward Witten's 1982 insight: deform the exterior derivative by `dₜ = e^{-tf} d e^{tf}`, giving a deformed Laplacian `Δₜ = (dₜ + dₜ*)²`.

For large `t`:
- The low-lying eigenvectors of `Δₜ` concentrate near critical points
- Eigenvalues are exponentially small: `~ e^{-t|f(p)-f(q)|}` (tunneling amplitudes)
- The Witten complex boundary operator has entries proportional to these tunneling amplitudes

**For agent learning**, the Witten deformation provides **reward shaping**:

```
Rₜ(x) = R(x) + t·∇f·∇V - t²|∇f|²
```

where `V` is the value function. This reshapes the reward landscape to make critical points more discoverable.

The **effective potential** `Veff = ½|∇f|² - Δf/(2t)` has its critical points coincide with those of `f` for large `t`.

### Fitness Landscapes and Game Theory

The connection to game theory:

1. **Fitness = Morse function.** The joint fitness of agents in a multi-agent system defines a function on the product of strategy spaces. Under generic conditions, this is a Morse function.

2. **Nash equilibria = critical points.** At a Nash equilibrium, no agent can improve by deviating — the gradient of each agent's fitness with respect to their strategy vanishes.

3. **Morse index = stability.** A Nash equilibrium with Morse index 0 (local minimum of regret) is an **evolutionary stable strategy** (ESS). Higher index means instability.

4. **Morse inequalities bound Nash equilibria.** The topology of the strategy space (a product of simplices) forces a minimum number of equilibria.

5. **Wilson's oddness theorem:** For a nondegenerate game, the number of Nash equilibria is odd.

6. **Gradient flow = learning.** Agents performing gradient ascent on fitness follow the Morse–Smale gradient flow. Trajectories connect equilibria along stable/unstable manifolds.

---

## License

MIT
