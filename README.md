# lau-morse-homology-agents

Morse theory applied to agent fitness landscapes — the topology of the landscape determines the agent's learning dynamics.

Critical points of the fitness function (where gradient = 0) reveal the structure of possible agent behaviors. Morse theory connects critical points to topology.

## Core Concepts

- **Morse function**: Smooth function on agent state space with non-degenerate critical points
- **Morse lemma**: Near critical points, `f(x) = f(p) - x₁² - ... - x_λ² + x_{λ+1}² + ... + x_n²`
- **Morse index**: Number of negative eigenvalues of Hessian = dimension of unstable manifold
- **Morse inequalities**: Relate critical points to homology groups
- **Morse homology**: Chain complex from critical points, boundary operator from gradient flow lines
- **Morse-Smale complex**: When stable/unstable manifolds intersect transversally
- **Witten deformation**: Twist the Laplacian by `e^{-tf}` → isolated critical points for large t
- **Agent fitness landscape**: Fitness as Morse function, critical points = Nash equilibria
- **Gradient flow**: Agent learning as gradient descent on fitness landscape
- **Nash equilibrium counting**: Count equilibria from topology via Morse inequalities

## Usage

```rust
use lau_morse_homology_agents::*;

// Create a Morse function on agent state space
let mut mf = morse_function::MorseFunction::from_polynomial(1, vec![0.0, -1.0, 0.0, 1.0/3.0]);
mf.find_critical_points_1d((-5.0, 5.0), 30);

// Compute Morse index from Hessian
let hessian = nalgebra::DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
let mi = morse_index::MorseIndex::from_hessian(&hessian);
assert_eq!(mi.index, 1); // saddle point

// Morse inequalities
let inequalities = morse_inequalities::MorseInequalities::new(vec![1, 0, 1], vec![1, 0, 1]);
assert!(inequalities.is_perfect()); // perfect Morse function on S^2

// Witten deformation for reward shaping
let mut wd = witten::WittenDeformation::new(vec![0.0, 1.0], vec![0, 1], 1);
wd.with_t(10.0);
let reward = wd.shaped_reward(1.0, &grad_f, &grad_v);
```

## License

MIT
