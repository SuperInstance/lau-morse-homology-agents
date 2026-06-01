//! Morse inequalities: relate critical points to homology groups.

use serde::{Serialize, Deserialize};

/// Betti numbers of a manifold (dimensions of homology groups).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettiNumbers {
    /// β_k = rank(H_k(M))
    pub betti: Vec<usize>,
}

impl BettiNumbers {
    pub fn new(betti: Vec<usize>) -> Self {
        BettiNumbers { betti }
    }

    /// Euler characteristic: χ = Σ (-1)^k β_k
    pub fn euler_characteristic(&self) -> i64 {
        self.betti.iter().enumerate()
            .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
            .sum()
    }

    /// Poincaré polynomial: Σ β_k t^k
    pub fn poincare_polynomial_coeffs(&self) -> Vec<usize> {
        self.betti.clone()
    }

    /// Total rank: Σ β_k
    pub fn total_rank(&self) -> usize {
        self.betti.iter().sum()
    }
}

/// Results of the Morse inequalities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorseInequalities {
    /// Morse polynomial coefficients: M_k = number of critical points of index k.
    pub morse_polynomial: Vec<usize>,
    /// Betti numbers.
    pub betti_numbers: BettiNumbers,
}

impl MorseInequalities {
    /// Compute Morse inequalities from Morse polynomial and Betti numbers.
    pub fn new(morse_polynomial: Vec<usize>, betti: Vec<usize>) -> Self {
        MorseInequalities {
            morse_polynomial,
            betti_numbers: BettiNumbers::new(betti),
        }
    }

    /// Weak Morse inequalities: M_k ≥ β_k for each k.
    pub fn verify_weak_inequalities(&self) -> Vec<bool> {
        let max_dim = self.morse_polynomial.len().max(self.betti_numbers.betti.len());
        (0..max_dim).map(|k| {
            let mk = self.morse_polynomial.get(k).copied().unwrap_or(0);
            let bk = self.betti_numbers.betti.get(k).copied().unwrap_or(0);
            mk >= bk
        }).collect()
    }

    /// Strong Morse inequalities: for all k,
    /// M_k - M_{k-1} + ... + (-1)^k M_0 ≥ β_k - β_{k-1} + ... + (-1)^k β_0
    pub fn verify_strong_inequalities(&self) -> Vec<bool> {
        let max_dim = self.morse_polynomial.len().max(self.betti_numbers.betti.len());
        (0..max_dim).map(|k| {
            let lhs: i64 = (0..=k).map(|j| {
                let mj = self.morse_polynomial.get(j).copied().unwrap_or(0) as i64;
                if (k - j) % 2 == 0 { mj } else { -mj }
            }).sum();
            let rhs: i64 = (0..=k).map(|j| {
                let bj = self.betti_numbers.betti.get(j).copied().unwrap_or(0) as i64;
                if (k - j) % 2 == 0 { bj } else { -bj }
            }).sum();
            lhs >= rhs
        }).collect()
    }

    /// Euler characteristic from Morse polynomial:
    /// χ = Σ (-1)^k M_k (equals the Euler characteristic from Betti numbers)
    pub fn euler_from_morse(&self) -> i64 {
        self.morse_polynomial.iter().enumerate()
            .map(|(k, &m)| if k % 2 == 0 { m as i64 } else { -(m as i64) })
            .sum()
    }

    /// Verify that the two Euler characteristic computations agree.
    pub fn verify_euler_characteristic(&self) -> bool {
        self.euler_from_morse() == self.betti_numbers.euler_characteristic()
    }

    /// Deficiency: d_k = M_k - β_k (non-negative by weak inequality).
    pub fn deficiencies(&self) -> Vec<i64> {
        let max_dim = self.morse_polynomial.len().max(self.betti_numbers.betti.len());
        (0..max_dim).map(|k| {
            let mk = self.morse_polynomial.get(k).copied().unwrap_or(0) as i64;
            let bk = self.betti_numbers.betti.get(k).copied().unwrap_or(0) as i64;
            mk - bk
        }).collect()
    }

    /// Is the Morse function perfect? (All deficiencies = 0, i.e., M_k = β_k for all k)
    pub fn is_perfect(&self) -> bool {
        self.deficiencies().iter().all(|&d| d == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_betti(dim: usize) -> Vec<usize> {
        // S^n: β_0 = 1, β_n = 1, rest = 0
        let mut betti = vec![0; dim + 1];
        betti[0] = 1;
        betti[dim] = 1;
        betti
    }

    fn torus_betti() -> Vec<usize> {
        // T^2: β_0 = 1, β_1 = 2, β_2 = 1
        vec![1, 2, 1]
    }

    #[test]
    fn test_euler_characteristic_sphere() {
        let betti = BettiNumbers::new(sphere_betti(2));
        // χ(S^2) = 1 - 0 + 1 = 2
        assert_eq!(betti.euler_characteristic(), 2);
    }

    #[test]
    fn test_euler_characteristic_torus() {
        let betti = BettiNumbers::new(torus_betti());
        // χ(T^2) = 1 - 2 + 1 = 0
        assert_eq!(betti.euler_characteristic(), 0);
    }

    #[test]
    fn test_euler_characteristic_circle() {
        let betti = BettiNumbers::new(sphere_betti(1));
        // χ(S^1) = 1 - 1 = 0
        assert_eq!(betti.euler_characteristic(), 0);
    }

    #[test]
    fn test_weak_morse_inequalities_hold() {
        // Perfect Morse function on S^2: M = [1, 0, 1]
        let mi = MorseInequalities::new(vec![1, 0, 1], sphere_betti(2));
        assert!(mi.verify_weak_inequalities().iter().all(|&b| b));
    }

    #[test]
    fn test_weak_morse_inequalities_non_perfect() {
        // Non-perfect on S^2: M = [2, 1, 1]
        let mi = MorseInequalities::new(vec![2, 1, 1], sphere_betti(2));
        assert!(mi.verify_weak_inequalities().iter().all(|&b| b));
    }

    #[test]
    fn test_strong_morse_inequalities() {
        let mi = MorseInequalities::new(vec![1, 0, 1], sphere_betti(2));
        assert!(mi.verify_strong_inequalities().iter().all(|&b| b));
    }

    #[test]
    fn test_euler_from_morse_matches_betti() {
        let mi = MorseInequalities::new(vec![1, 0, 1], sphere_betti(2));
        assert!(mi.verify_euler_characteristic());
    }

    #[test]
    fn test_euler_from_morse_torus() {
        // Torus: M = [1, 2, 1] (perfect)
        let mi = MorseInequalities::new(vec![1, 2, 1], torus_betti());
        assert_eq!(mi.euler_from_morse(), 0);
        assert!(mi.verify_euler_characteristic());
    }

    #[test]
    fn test_perfect_morse_function() {
        let mi = MorseInequalities::new(vec![1, 0, 1], sphere_betti(2));
        assert!(mi.is_perfect());
    }

    #[test]
    fn test_non_perfect_morse_function() {
        let mi = MorseInequalities::new(vec![2, 1, 1], sphere_betti(2));
        assert!(!mi.is_perfect());
    }

    #[test]
    fn test_deficiencies() {
        let mi = MorseInequalities::new(vec![2, 1, 1], sphere_betti(2));
        let def = mi.deficiencies();
        assert_eq!(def, vec![1, 1, 0]);
    }

    #[test]
    fn test_total_rank() {
        let betti = BettiNumbers::new(torus_betti());
        assert_eq!(betti.total_rank(), 4);
    }

    #[test]
    fn test_poincare_polynomial() {
        let betti = BettiNumbers::new(torus_betti());
        assert_eq!(betti.poincare_polynomial_coeffs(), vec![1, 2, 1]);
    }

    #[test]
    fn test_serialization() {
        let mi = MorseInequalities::new(vec![1, 2, 1], torus_betti());
        let json = serde_json::to_string(&mi).unwrap();
        let decoded: MorseInequalities = serde_json::from_str(&json).unwrap();
        assert!(decoded.verify_euler_characteristic());
    }

    #[test]
    fn test_inequalities_violated() {
        // Impossible: fewer critical points than Betti numbers
        let mi = MorseInequalities::new(vec![0, 0, 0], sphere_betti(2));
        assert!(!mi.verify_weak_inequalities()[0]);
    }
}
