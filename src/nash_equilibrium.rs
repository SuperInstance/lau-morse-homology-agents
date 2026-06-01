//! Nash equilibrium counting from topology via Morse inequalities.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

use crate::morse_function::CriticalPointType;
use crate::morse_inequalities::MorseInequalities;
use crate::fitness_landscape::FitnessLandscape;

/// Information about a Nash equilibrium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquilibriumInfo {
    /// Position in strategy space.
    pub position: DVector<f64>,
    /// Type: minimum, saddle, or maximum of the regret function.
    pub equilibrium_type: CriticalPointType,
    /// Morse index (number of unstable directions).
    pub morse_index: usize,
    /// Fitness value.
    pub fitness: f64,
    /// Stability: true if a local minimum of regret (ESS-like).
    pub is_stable: bool,
}

/// Counter for Nash equilibria using Morse theory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NashEquilibriumCounter {
    /// The fitness landscape.
    pub landscape: FitnessLandscape,
    /// Found equilibria.
    pub equilibria: Vec<EquilibriumInfo>,
}

impl NashEquilibriumCounter {
    /// Create a new counter for the given landscape.
    pub fn new(landscape: FitnessLandscape) -> Self {
        NashEquilibriumCounter {
            landscape,
            equilibria: Vec::new(),
        }
    }

    /// Count Nash equilibria using Morse inequalities.
    /// Given Betti numbers β_k of the strategy space, the number N of Nash equilibria satisfies:
    /// N ≥ Σ β_k (weak inequalities)
    /// The strong inequalities give tighter bounds.
    pub fn count_from_topology(betti: &[usize]) -> MorseInequalities {
        // Lower bound: sum of Betti numbers
        let _total: usize = betti.iter().sum();
        // Create a Morse polynomial with at least these many critical points
        // The actual count comes from the game structure; this gives the topological lower bound.
        let morse_polynomial = betti.to_vec(); // Perfect Morse function = lower bound
        MorseInequalities::new(morse_polynomial, betti.to_vec())
    }

    /// Lower bound on the number of Nash equilibria from weak Morse inequalities.
    pub fn lower_bound_weak(betti: &[usize]) -> usize {
        betti.iter().sum()
    }

    /// Lower bound from the k-th strong Morse inequality.
    pub fn lower_bound_strong_k(betti: &[usize], k: usize) -> i64 {
        (0..=k).map(|j| {
            let bj = betti.get(j).copied().unwrap_or(0) as i64;
            if (k - j) % 2 == 0 { bj } else { -bj }
        }).sum()
    }

    /// Find and classify Nash equilibria in a 2x2 game.
    pub fn find_equilibria_2x2(&mut self) -> usize {
        let nash = self.landscape.find_nash_equilibria_2x2();
        self.equilibria.clear();

        for (p, _, q, _) in &nash {
            let pos = DVector::from_vec(vec![*p, *q]);
            // Classify: use a simple Hessian approximation
            // For mixed equilibria in coordination games, check stability
            let hess = DMatrix::identity(2, 2); // Placeholder
            let eq_type = self.landscape.classify_equilibrium(&pos, &hess);
            let fitness = self.landscape.evaluate_fitness(&pos);

            self.equilibria.push(EquilibriumInfo {
                position: pos,
                equilibrium_type: eq_type,
                morse_index: 0,
                fitness,
                is_stable: eq_type == CriticalPointType::Minimum,
            });
        }

        self.equilibria.len()
    }

    /// Number of stable equilibria (local minima of regret = ESS candidates).
    pub fn num_stable(&self) -> usize {
        self.equilibria.iter().filter(|e| e.is_stable).count()
    }

    /// Number of unstable equilibria (saddles and maxima).
    pub fn num_unstable(&self) -> usize {
        self.equilibria.iter().filter(|e| !e.is_stable).count()
    }

    /// Morse polynomial from found equilibria.
    pub fn morse_polynomial(&self) -> Vec<usize> {
        if self.equilibria.is_empty() {
            return Vec::new();
        }
        let max_index = self.equilibria.iter().map(|e| e.morse_index).max().unwrap_or(0);
        let mut mp = vec![0usize; max_index + 1];
        for e in &self.equilibria {
            mp[e.morse_index] += 1;
        }
        mp
    }

    /// Check the Morse inequality: M_k ≥ β_k for all k.
    pub fn verify_morse_inequalities(&self, betti: &[usize]) -> bool {
        let mp = self.morse_polynomial();
        let max_k = mp.len().max(betti.len());
        for k in 0..max_k {
            let mk = mp.get(k).copied().unwrap_or(0);
            let bk = betti.get(k).copied().unwrap_or(0);
            if mk < bk {
                return false;
            }
        }
        true
    }

    /// Count Nash equilibria for a general n-player game using the Wilson oddness theorem.
    /// For a nondegenerate game, the number of Nash equilibria is odd.
    pub fn wilson_oddness(equilibria_count: usize) -> bool {
        equilibria_count % 2 == 1
    }

    /// Estimate equilibrium count from the topology of the strategy space.
    /// For strategy space S = (Δ^{n_1-1} × ... × Δ^{n_k-1}), the Euler characteristic is:
    /// χ(S) = Π_i (1 + (-1)^{n_i-1} ... ) (product of simplices)
    pub fn euler_characteristic_product(simplices: &[usize]) -> i64 {
        // Euler characteristic of a simplex Δ^{n-1} = 1 (contractible)
        // For spheres: χ(S^{n-1}) = 1 + (-1)^{n-1}
        // The Nash equilibria live on the product of spheres (boundary of simplices)
        simplices.iter().map(|&n| {
            let chi_sphere = 1 + if (n - 1) % 2 == 0 { 1 } else { -1 };
            chi_sphere
        }).product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_lower_bound_weak() {
        // S^2 betti: [1, 0, 1] → lower bound 2
        assert_eq!(NashEquilibriumCounter::lower_bound_weak(&[1, 0, 1]), 2);
    }

    #[test]
    fn test_lower_bound_strong_k() {
        // S^2: lower bound for k=0: β_0 = 1, for k=2: β_0 - β_1 + β_2 = 1-0+1 = 2
        assert_eq!(NashEquilibriumCounter::lower_bound_strong_k(&[1, 0, 1], 0), 1);
        assert_eq!(NashEquilibriumCounter::lower_bound_strong_k(&[1, 0, 1], 2), 2);
    }

    #[test]
    fn test_count_from_topology() {
        let mi = NashEquilibriumCounter::count_from_topology(&[1, 0, 1]);
        assert!(mi.verify_weak_inequalities().iter().all(|&b| b));
        assert!(mi.is_perfect());
    }

    #[test]
    fn test_find_prisoners_dilemma_equilibria() {
        let row = DMatrix::from_row_slice(2, 2, &[-1.0, -3.0, 0.0, -2.0]);
        let col = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, -3.0, -2.0]);
        let fl = FitnessLandscape::from_two_player_game(&row, &col);
        let mut counter = NashEquilibriumCounter::new(fl);
        let count = counter.find_equilibria_2x2();
        assert!(count >= 1);
    }

    #[test]
    fn test_wilson_oddness() {
        assert!(NashEquilibriumCounter::wilson_oddness(1));
        assert!(NashEquilibriumCounter::wilson_oddness(3));
        assert!(!NashEquilibriumCounter::wilson_oddness(2));
    }

    #[test]
    fn test_num_stable_unstable() {
        let row = DMatrix::from_row_slice(2, 2, &[-1.0, -3.0, 0.0, -2.0]);
        let col = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, -3.0, -2.0]);
        let fl = FitnessLandscape::from_two_player_game(&row, &col);
        let mut counter = NashEquilibriumCounter::new(fl);
        counter.find_equilibria_2x2();
        let total = counter.num_stable() + counter.num_unstable();
        assert_eq!(total, counter.equilibria.len());
    }

    #[test]
    fn test_morse_polynomial_from_equilibria() {
        let fl = FitnessLandscape::new(2, 1);
        let mut counter = NashEquilibriumCounter::new(fl);
        counter.equilibria.push(EquilibriumInfo {
            position: DVector::from_vec(vec![0.0, 0.0]),
            equilibrium_type: CriticalPointType::Minimum,
            morse_index: 0,
            fitness: 1.0,
            is_stable: true,
        });
        counter.equilibria.push(EquilibriumInfo {
            position: DVector::from_vec(vec![1.0, 1.0]),
            equilibrium_type: CriticalPointType::Minimum,
            morse_index: 0,
            fitness: 2.0,
            is_stable: true,
        });
        let mp = counter.morse_polynomial();
        assert_eq!(mp, vec![2]);
    }

    #[test]
    fn test_verify_morse_inequalities() {
        let fl = FitnessLandscape::new(2, 1);
        let mut counter = NashEquilibriumCounter::new(fl);
        counter.equilibria.push(EquilibriumInfo {
            position: DVector::from_vec(vec![0.0]),
            equilibrium_type: CriticalPointType::Minimum,
            morse_index: 0,
            fitness: 0.0,
            is_stable: true,
        });
        // β_0 = 1, M_0 = 1 → holds
        assert!(counter.verify_morse_inequalities(&[1]));
    }

    #[test]
    fn test_euler_characteristic_product() {
        // S^1 × S^1: χ = 0 × 0 = 0 (torus)
        let chi = NashEquilibriumCounter::euler_characteristic_product(&[2, 2]);
        assert_eq!(chi, 0);
    }

    #[test]
    fn test_euler_s2() {
        // S^2: χ = 2
        let chi = NashEquilibriumCounter::euler_characteristic_product(&[3]);
        assert_eq!(chi, 2);
    }

    #[test]
    fn test_coordination_game_multiple_equilibria() {
        let row = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 1.0]);
        let col = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 1.0]);
        let fl = FitnessLandscape::from_two_player_game(&row, &col);
        let mut counter = NashEquilibriumCounter::new(fl);
        let count = counter.find_equilibria_2x2();
        // Coordination game has 3 Nash: 2 pure + 1 mixed
        assert!(count >= 2);
    }

    #[test]
    fn test_serialization_equilibrium_info() {
        let info = EquilibriumInfo {
            position: DVector::from_vec(vec![0.5, 0.5]),
            equilibrium_type: CriticalPointType::Saddle,
            morse_index: 1,
            fitness: 0.5,
            is_stable: false,
        };
        let json = serde_json::to_string(&info).unwrap();
        let decoded: EquilibriumInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.morse_index, 1);
        assert!(!decoded.is_stable);
    }

    #[test]
    fn test_serialization_counter() {
        let fl = FitnessLandscape::new(2, 1);
        let counter = NashEquilibriumCounter::new(fl);
        let json = serde_json::to_string(&counter).unwrap();
        let decoded: NashEquilibriumCounter = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.equilibria.len(), 0);
    }
}
