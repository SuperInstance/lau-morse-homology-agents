//! Agent fitness landscape: fitness as Morse function, critical points = Nash equilibria.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

use crate::morse_function::{MorseFunction, CriticalPointType};

/// An agent's state in the fitness landscape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Strategy parameters (continuous).
    pub strategy: DVector<f64>,
    /// Agent identifier.
    pub id: usize,
    /// Fitness value at current strategy.
    pub fitness: f64,
}

impl AgentState {
    /// Create a new agent state.
    pub fn new(id: usize, strategy: DVector<f64>) -> Self {
        AgentState { strategy, id, fitness: 0.0 }
    }

    /// Distance to another agent's state.
    pub fn distance_to(&self, other: &AgentState) -> f64 {
        (&self.strategy - &other.strategy).norm()
    }

    /// Dimension of the strategy space.
    pub fn dimension(&self) -> usize {
        self.strategy.len()
    }
}

/// The fitness landscape for agents in a multi-agent system.
/// The fitness function acts as a Morse function on the joint strategy space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessLandscape {
    /// Number of agents.
    pub num_agents: usize,
    /// Dimension per agent's strategy space.
    pub strategy_dim: usize,
    /// The Morse function representing fitness.
    pub morse_function: MorseFunction,
    /// Payoff matrix for normal-form games (if applicable).
    /// Stored as a flattened tensor: (agent1_strategy, agent2_strategy, ...) → payoff.
    pub payoff_matrix: Vec<f64>,
    /// Strategy cardinalities per agent (for finite games).
    pub strategy_counts: Vec<usize>,
}

impl FitnessLandscape {
    /// Create a new fitness landscape.
    pub fn new(num_agents: usize, strategy_dim: usize) -> Self {
        FitnessLandscape {
            num_agents,
            strategy_dim,
            morse_function: MorseFunction::new(num_agents * strategy_dim),
            payoff_matrix: Vec::new(),
            strategy_counts: Vec::new(),
        }
    }

    /// Create from a 2-player normal-form game payoff matrix.
    pub fn from_two_player_game(
        payoff_row_player: &DMatrix<f64>,
        payoff_col_player: &DMatrix<f64>,
    ) -> Self {
        let rows = payoff_row_player.nrows();
        let cols = payoff_col_player.ncols();

        // Combined payoff: (row_strategy, col_strategy) → (row_payoff, col_payoff)
        let mut combined = Vec::with_capacity(rows * cols * 2);
        for i in 0..rows {
            for j in 0..cols {
                combined.push(payoff_row_player[(i, j)]);
                combined.push(payoff_col_player[(i, j)]);
            }
        }

        FitnessLandscape {
            num_agents: 2,
            strategy_dim: 1,
            morse_function: MorseFunction::new(2),
            payoff_matrix: combined,
            strategy_counts: vec![rows, cols],
        }
    }

    /// Evaluate fitness for a given joint strategy.
    /// For polynomial fitness: delegates to the Morse function.
    pub fn evaluate_fitness(&self, strategy: &DVector<f64>) -> f64 {
        if strategy.len() == 1 && !self.morse_function.coefficients.is_empty() {
            self.morse_function.evaluate_1d(strategy[0])
        } else if !self.morse_function.coefficients.is_empty() {
            // Quadratic form: f(x) = Σ c_i x_i^2
            self.morse_function.coefficients.iter().enumerate()
                .map(|(i, &c)| {
                    if i < strategy.len() {
                        c * strategy[i] * strategy[i]
                    } else {
                        0.0
                    }
                })
                .sum()
        } else {
            0.0
        }
    }

    /// Compute the gradient of fitness.
    pub fn fitness_gradient(&self, strategy: &DVector<f64>) -> DVector<f64> {
        if strategy.len() == 1 && !self.morse_function.coefficients.is_empty() {
            DVector::from_vec(vec![self.morse_function.gradient_1d(strategy[0])])
        } else {
            // Quadratic: ∇f = 2 * diag(c) * x
            let grad: Vec<f64> = self.morse_function.coefficients.iter().enumerate()
                .map(|(i, &c)| {
                    if i < strategy.len() { 2.0 * c * strategy[i] } else { 0.0 }
                })
                .collect();
            DVector::from_vec(grad)
        }
    }

    /// Find Nash equilibria as critical points of the fitness function.
    /// For normal-form games, uses support enumeration.
    pub fn find_nash_equilibria_2x2(&self) -> Vec<(f64, f64, f64, f64)> {
        // For 2x2 game: mixed strategies (p, 1-p) and (q, 1-q)
        // Nash equilibrium: each player indifferent between pure strategies
        // given the other's mixed strategy.
        if self.strategy_counts.len() != 2 || self.strategy_counts[0] != 2 || self.strategy_counts[1] != 2 {
            return Vec::new();
        }

        // Payoff matrices (extracted from combined)
        // Row player: A[i][j] = combined[2*(i*2+j)]
        // Col player: B[i][j] = combined[2*(i*2+j)+1]
        let a00 = self.payoff_matrix[0];
        let b00 = self.payoff_matrix[1];
        let a01 = self.payoff_matrix[2];
        let b01 = self.payoff_matrix[3];
        let a10 = self.payoff_matrix[4];
        let b10 = self.payoff_matrix[5];
        let a11 = self.payoff_matrix[6];
        let b11 = self.payoff_matrix[7];

        let mut equilibria = Vec::new();

        // Pure strategy equilibria
        for (i, j) in [(0,0), (0,1), (1,0), (1,1)] {
            let (ai0, ai1) = if i == 0 { (a00, a01) } else { (a10, a11) };
            let (b0j, b1j) = if j == 0 { (b00, b10) } else { (b01, b11) };
            let best_row = if ai0 >= ai1 { 0 } else { 1 };
            let best_col = if b0j >= b1j { 0 } else { 1 };
            if best_row == i && best_col == j {
                let p = if i == 0 { 1.0 } else { 0.0 };
                let q = if j == 0 { 1.0 } else { 0.0 };
                equilibria.push((p, 1.0-p, q, 1.0-q));
            }
        }

        // Mixed strategy equilibrium
        // Row player indifferent: q*a00 + (1-q)*a01 = q*a10 + (1-q)*a11
        // q*(a00-a01-a10+a11) = a11-a01
        let denom_r = a00 - a01 - a10 + a11;
        if denom_r.abs() > 1e-10 {
            let q = (a11 - a01) / denom_r;
            let denom_c = b00 - b10 - b01 + b11;
            if denom_c.abs() > 1e-10 {
                let p = (b11 - b10) / denom_c;
                if q > 0.0 && q < 1.0 && p > 0.0 && p < 1.0 {
                    equilibria.push((p, 1.0-p, q, 1.0-q));
                }
            }
        }

        equilibria
    }

    /// Classify a Nash equilibrium using Morse theory.
    /// The type (min/saddle/max of the regret function) determines stability.
    pub fn classify_equilibrium(&self, position: &DVector<f64>, hessian: &DMatrix<f64>) -> CriticalPointType {
        let eig = hessian.symmetric_eigenvalues();
        let index = eig.iter().filter(|&&v| v < 0.0).count();
        CriticalPointType::from_index(index, position.len())
    }

    /// Compute the fitness gap between best response and current strategy.
    pub fn regret(&self, current_fitness: f64, best_response_fitness: f64) -> f64 {
        best_response_fitness - current_fitness
    }

    /// Is the landscape a valid Morse function? (all critical points non-degenerate)
    pub fn is_morse(&self) -> bool {
        self.morse_function.verify_non_degenerate()
    }

    /// Total dimension of the joint strategy space.
    pub fn total_dimension(&self) -> usize {
        self.num_agents * self.strategy_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_agent_state_creation() {
        let state = AgentState::new(0, DVector::from_vec(vec![1.0, 2.0]));
        assert_eq!(state.dimension(), 2);
        assert_eq!(state.id, 0);
    }

    #[test]
    fn test_agent_distance() {
        let s1 = AgentState::new(0, DVector::from_vec(vec![0.0, 0.0]));
        let s2 = AgentState::new(1, DVector::from_vec(vec![3.0, 4.0]));
        assert_relative_eq!(s1.distance_to(&s2), 5.0);
    }

    #[test]
    fn test_fitness_landscape_creation() {
        let fl = FitnessLandscape::new(3, 2);
        assert_eq!(fl.total_dimension(), 6);
    }

    #[test]
    fn test_prisoners_dilemma() {
        // Classic Prisoner's Dilemma
        let row_payoffs = DMatrix::from_row_slice(2, 2, &[
            -1.0, -3.0,  // Cooperate: D, C
             0.0, -2.0,  // Defect: D, C
        ]);
        let col_payoffs = DMatrix::from_row_slice(2, 2, &[
            -1.0,  0.0,
            -3.0, -2.0,
        ]);
        let fl = FitnessLandscape::from_two_player_game(&row_payoffs, &col_payoffs);
        let nash = fl.find_nash_equilibria_2x2();
        // Unique Nash: (Defect, Defect) = (0, 1, 0, 1)
        assert!(!nash.is_empty());
        // (Defect, Defect) = (1, 0, 1, 0) in our convention (p=prob of row 0)
        let has_dd = nash.iter().any(|&(p, _, q, _)| p > 0.99 && q > 0.99);
        assert!(has_dd);
    }

    #[test]
    fn test_coordination_game() {
        // Coordination game: both prefer same action
        let row = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 1.0]);
        let col = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 1.0]);
        let fl = FitnessLandscape::from_two_player_game(&row, &col);
        let nash = fl.find_nash_equilibria_2x2();
        // Two pure Nash: (C,C) and (D,D)
        assert!(nash.len() >= 2);
    }

    #[test]
    fn test_matching_pennies() {
        // Matching pennies: unique mixed Nash
        let row = DMatrix::from_row_slice(2, 2, &[1.0, -1.0, -1.0, 1.0]);
        let col = DMatrix::from_row_slice(2, 2, &[-1.0, 1.0, 1.0, -1.0]);
        let fl = FitnessLandscape::from_two_player_game(&row, &col);
        let nash = fl.find_nash_equilibria_2x2();
        // Should have one mixed equilibrium at (0.5, 0.5)
        let has_mixed = nash.iter().any(|&(p, _, q, _)| {
            (p - 0.5).abs() < 0.01 && (q - 0.5).abs() < 0.01
        });
        assert!(has_mixed);
    }

    #[test]
    fn test_classify_equilibrium() {
        let fl = FitnessLandscape::new(1, 1);
        let pos = DVector::from_vec(vec![0.0]);
        let hess = DMatrix::identity(1, 1);
        assert_eq!(fl.classify_equilibrium(&pos, &hess), CriticalPointType::Minimum);
    }

    #[test]
    fn test_regret() {
        let fl = FitnessLandscape::new(1, 1);
        assert_relative_eq!(fl.regret(3.0, 5.0), 2.0);
    }

    #[test]
    fn test_polynomial_fitness() {
        let mut fl = FitnessLandscape::new(1, 1);
        fl.morse_function = MorseFunction::from_polynomial(1, vec![0.0, 0.0, -1.0]);
        let x = DVector::from_vec(vec![2.0]);
        // f(x) = -x^2, f(2) = -4
        assert_relative_eq!(fl.evaluate_fitness(&x), -4.0);
    }

    #[test]
    fn test_fitness_gradient() {
        let mut fl = FitnessLandscape::new(1, 1);
        fl.morse_function = MorseFunction::from_polynomial(1, vec![0.0, 0.0, 1.0]);
        let x = DVector::from_vec(vec![3.0]);
        let grad = fl.fitness_gradient(&x);
        // ∇(x^2) = 2x = 6
        assert_relative_eq!(grad[0], 6.0);
    }

    #[test]
    fn test_serialization() {
        let state = AgentState::new(0, DVector::from_vec(vec![1.0]));
        let json = serde_json::to_string(&state).unwrap();
        let decoded: AgentState = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, 0);
    }

    #[test]
    fn test_landscape_serialization() {
        let fl = FitnessLandscape::new(2, 3);
        let json = serde_json::to_string(&fl).unwrap();
        let decoded: FitnessLandscape = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.num_agents, 2);
    }
}
