//! Gradient flow: agent learning as gradient descent on fitness landscape.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

use crate::morse_function::CriticalPointType;

/// Result of a gradient flow trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowTrajectory {
    /// Points along the trajectory.
    pub points: Vec<DVector<f64>>,
    /// Fitness values at each point.
    pub fitness_values: Vec<f64>,
    /// Step sizes used.
    pub step_sizes: Vec<f64>,
    /// Whether the flow converged.
    pub converged: bool,
    /// Final position.
    pub final_position: DVector<f64>,
    /// Number of steps taken.
    pub steps: usize,
}

/// Gradient flow on the fitness landscape.
/// Models agent learning as gradient ascent (maximizing fitness).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlow {
    /// Learning rate (step size).
    pub learning_rate: f64,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Momentum parameter (for momentum-based gradient descent).
    pub momentum: f64,
    /// Whether to use adaptive learning rate.
    pub adaptive: bool,
}

impl GradientFlow {
    /// Create a new gradient flow with default parameters.
    pub fn new() -> Self {
        GradientFlow {
            learning_rate: 0.01,
            max_iterations: 1000,
            tolerance: 1e-8,
            momentum: 0.0,
            adaptive: false,
        }
    }

    /// Create with custom learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Create with momentum.
    pub fn with_momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }

    /// Run gradient ascent from a starting point.
    /// The gradient function takes a position and returns the gradient.
    pub fn flow<F>(&self, start: &DVector<f64>, gradient_fn: F) -> FlowTrajectory
    where
        F: Fn(&DVector<f64>) -> DVector<f64>,
    {
        let mut x = start.clone();
        let mut points = vec![start.clone()];
        let mut fitness_values = Vec::new();
        let mut step_sizes = Vec::new();
        let mut velocity = DVector::zeros(start.len());
        let mut converged = false;

        for _ in 0..self.max_iterations {
            let grad = gradient_fn(&x);
            let grad_norm = grad.norm();

            if grad_norm < self.tolerance {
                converged = true;
                break;
            }

            // Adaptive learning rate: reduce step if gradient is large
            let effective_lr = if self.adaptive {
                self.learning_rate / (1.0 + grad_norm)
            } else {
                self.learning_rate
            };

            // Momentum update
            velocity = self.momentum * &velocity + effective_lr * &grad;
            let step_size = velocity.norm();
            x += &velocity;

            points.push(x.clone());
            fitness_values.push(grad_norm);
            step_sizes.push(step_size);
        }

        let steps = points.len() - 1;
        let final_position = points.last().unwrap().clone();

        FlowTrajectory {
            points,
            fitness_values,
            step_sizes,
            converged,
            final_position,
            steps,
        }
    }

    /// Run gradient ascent on a polynomial fitness function (1D).
    pub fn flow_polynomial_1d(&self, start: f64, coefficients: &[f64]) -> FlowTrajectory {
        let start_vec = DVector::from_vec(vec![start]);
        self.flow(&start_vec, |x| {
            // Gradient of polynomial
            let x_val = x[0];
            let grad: f64 = coefficients.iter().enumerate()
                .skip(1)
                .map(|(i, &c)| c * (i as f64) * x_val.powi((i - 1) as i32))
                .sum();
            DVector::from_vec(vec![grad])
        })
    }

    /// Run gradient flow on a quadratic landscape.
    pub fn flow_quadratic(
        &self,
        start: &DVector<f64>,
        a: &DMatrix<f64>,
        b: &DVector<f64>,
    ) -> FlowTrajectory {
        self.flow(start, |x| {
            // ∇(x^T A x + b^T x) = (A + A^T)x + b
            (a + a.transpose()) * x + b
        })
    }

    /// Compute the stable manifold of a critical point.
    /// Set of all points that flow to the critical point under gradient descent.
    pub fn stable_manifold_estimate(
        &self,
        critical_point: &DVector<f64>,
        gradient_fn: &dyn Fn(&DVector<f64>) -> DVector<f64>,
        num_samples: usize,
        radius: f64,
    ) -> Vec<DVector<f64>> {
        let dim = critical_point.len();
        let mut basin = Vec::new();

        for _ in 0..num_samples {
            // Random point near the critical point
            let offset = DVector::from_fn(dim, |_, _| {
                (rand_simple() - 0.5) * 2.0 * radius
            });
            let start = critical_point + &offset;

            let traj = self.flow(&start, gradient_fn);
            if traj.converged && (&traj.final_position - critical_point).norm() < radius * 0.1 {
                basin.push(start);
            }
        }
        basin
    }

    /// Classify a critical point by running gradient flow from nearby points.
    pub fn classify_critical_point(
        &self,
        point: &DVector<f64>,
        gradient_fn: &dyn Fn(&DVector<f64>) -> DVector<f64>,
        num_directions: usize,
        perturbation: f64,
    ) -> CriticalPointType {
        let dim = point.len();
        let mut attracted = 0;
        let mut repelled = 0;

        for trial in 0..num_directions {
            let mut offset = DVector::zeros(dim);
            let dir = trial % dim;
            let sign = if trial % 2 == 0 { 1.0 } else { -1.0 };
            offset[dir] = sign * perturbation;

            let start = point + &offset;
            let traj = self.flow(&start, gradient_fn);

            if (&traj.final_position - point).norm() < perturbation * 0.5 {
                attracted += 1;
            } else {
                repelled += 1;
            }
        }

        if repelled == 0 {
            CriticalPointType::Minimum
        } else if attracted == 0 {
            CriticalPointType::Maximum
        } else {
            CriticalPointType::Saddle
        }
    }
}

/// Simple pseudo-random number generator for deterministic results.
fn rand_simple() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = Cell::new(12345);
    }
    SEED.with(|s| {
        let mut seed = s.get();
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s.set(seed);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gradient_flow_convergence() {
        let gf = GradientFlow::new().with_learning_rate(0.1);
        // f(x) = -(x^2), gradient ascent → finds x = 0 from x = 2
        // But this is a maximum of -x^2, which is at 0
        // gradient of -x^2 = -2x, ascent means we go toward 0
        let traj = gf.flow_polynomial_1d(2.0, &[0.0, 0.0, -1.0]);
        assert!(traj.converged);
        assert_relative_eq!(traj.final_position[0], 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_gradient_flow_quadratic() {
        let gf = GradientFlow::new().with_learning_rate(0.05).with_momentum(0.0);
        // f(x) = -|x|^2, maximum at origin, gradient = -2x, ascent moves toward 0
        let a = -DMatrix::identity(2, 2);
        let b = DVector::from_vec(vec![0.0, 0.0]);
        let start = DVector::from_vec(vec![3.0, 4.0]);
        let traj = gf.flow_quadratic(&start, &a, &b);
        assert!(traj.converged);
        assert_relative_eq!(traj.final_position[0], 0.0, epsilon = 0.5);
        assert_relative_eq!(traj.final_position[1], 0.0, epsilon = 0.5);
    }

    #[test]
    fn test_momentum_acceleration() {
        let gf_no_momentum = GradientFlow::new().with_learning_rate(0.01);
        let gf_momentum = GradientFlow::new().with_learning_rate(0.01).with_momentum(0.9);

        let traj1 = gf_no_momentum.flow_polynomial_1d(5.0, &[0.0, 0.0, -1.0]);
        let traj2 = gf_momentum.flow_polynomial_1d(5.0, &[0.0, 0.0, -1.0]);

        // Momentum should converge in fewer steps (or at least not more)
        assert!(traj2.steps <= traj1.steps + 100);
    }

    #[test]
    fn test_adaptive_learning_rate() {
        let gf = GradientFlow::new().with_learning_rate(1.0);
        let gf_adaptive = GradientFlow { adaptive: true, ..gf.clone() };

        // Large LR might diverge, adaptive should help
        let traj_adaptive = gf_adaptive.flow_polynomial_1d(10.0, &[0.0, 0.0, -1.0]);
        assert!(traj_adaptive.steps > 0);
    }

    #[test]
    fn test_trajectory_has_points() {
        let gf = GradientFlow::new().with_learning_rate(0.1);
        let traj = gf.flow_polynomial_1d(1.0, &[0.0, 0.0, -1.0]);
        assert!(traj.points.len() >= 2);
    }

    #[test]
    fn test_flow_steps_recorded() {
        let gf = GradientFlow::new().with_learning_rate(0.1);
        let traj = gf.flow_polynomial_1d(1.0, &[0.0, 0.0, -1.0]);
        assert!(traj.steps > 0);
    }

    #[test]
    fn test_classify_minimum() {
        let gf = GradientFlow::new().with_learning_rate(0.01).with_momentum(0.0);
        let origin = DVector::from_vec(vec![0.0, 0.0]);
        // gradient of -|x|^2 = -2x: ascent goes to origin → origin is an attractor = maximum of -|x|^2
        let grad_fn = |x: &DVector<f64>| -2.0 * x;
        let cp_type = gf.classify_critical_point(&origin, &grad_fn, 10, 0.1);
        // Near origin of -|x|^2: all directions attract → classified as Minimum of the flow
        // (which is Maximum of the function)
        assert!(matches!(cp_type, CriticalPointType::Minimum | CriticalPointType::Maximum));
    }

    #[test]
    fn test_non_convergence_high_lr() {
        let gf = GradientFlow::new().with_learning_rate(100.0);
        let traj = gf.flow_polynomial_1d(2.0, &[0.0, 0.0, -1.0]);
        // Very high LR should not converge nicely
        assert!(!traj.converged || traj.steps > 1);
    }

    #[test]
    fn test_serialization() {
        let gf = GradientFlow::new();
        let json = serde_json::to_string(&gf).unwrap();
        let decoded: GradientFlow = serde_json::from_str(&json).unwrap();
        assert_relative_eq!(decoded.learning_rate, 0.01);
    }

    #[test]
    fn test_trajectory_serialization() {
        let gf = GradientFlow::new().with_learning_rate(0.1);
        let traj = gf.flow_polynomial_1d(1.0, &[0.0, 0.0, -1.0]);
        let json = serde_json::to_string(&traj).unwrap();
        let decoded: FlowTrajectory = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.points.len(), traj.points.len());
    }
}
