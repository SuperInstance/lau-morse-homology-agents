//! Witten deformation: twist the Laplacian by e^{-tf} → isolated critical points for large t.
//! Connects to lau-witten-reward: reward shaping via Witten deformation.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// Witten deformation of the de Rham complex.
///
/// For a Morse function f on a manifold M, the Witten-deformed Laplacian is:
/// Δ_t = (d_t + d_t*)² where d_t = e^{-tf} d e^{tf}
///
/// For large t, the low-lying eigenvectors of Δ_t concentrate near critical points,
/// and the eigenvalues are exponentially small (proportional to e^{-t|f(p)-f(q)|}).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WittenDeformation {
    /// Morse function values at critical points.
    pub critical_values: Vec<f64>,
    /// Morse indices at critical points.
    pub critical_indices: Vec<usize>,
    /// Deformation parameter t.
    pub t: f64,
    /// Dimension of the manifold.
    pub dimension: usize,
}

impl WittenDeformation {
    /// Create a new Witten deformation.
    pub fn new(critical_values: Vec<f64>, critical_indices: Vec<usize>, dimension: usize) -> Self {
        WittenDeformation {
            critical_values,
            critical_indices,
            t: 1.0,
            dimension,
        }
    }

    /// Set the deformation parameter.
    pub fn with_t(&mut self, t: f64) -> &mut Self {
        self.t = t;
        self
    }

    /// The Witten-deformed exterior derivative on 0-forms:
    /// (d_t φ)(x) = (dφ)(x) - t (df)(x) ∧ φ(x)
    /// For a function φ: d_t φ = dφ - t (∇f) φ
    pub fn deformed_derivative_0(&self, gradient_f: &DVector<f64>, dphi: &DVector<f64>) -> DVector<f64> {
        dphi - self.t * gradient_f
    }

    /// The Witten-deformed Laplacian on functions (0-forms):
    /// Δ_t φ = Δφ - t²|∇f|² + t (Hess f)
    /// Simplified: Δ_t = -Δ + t²|∇f|² - t·tr(Hess f)
    pub fn deformed_laplacian_0(
        &self,
        laplacian_phi: f64,
        grad_f_norm_sq: f64,
        hess_trace: f64,
    ) -> f64 {
        // Witten Laplacian at a point
        -laplacian_phi + self.t * self.t * grad_f_norm_sq + self.t * hess_trace
    }

    /// Estimate the Witten eigenvalue gap for a critical point of index k.
    /// For large t, the eigenvalue near critical point p is approximately:
    /// λ_k(t) ≈ Σ (|μ_i|) where μ_i are the eigenvalues of Hess f at p,
    /// but this grows with t. The instantaneous eigenvalue of the Witten complex
    /// at a critical point of index k is:
    /// λ ≈ t × Σ_{i ≤ k} |μ_i| for the "small" eigenvalue branch.
    pub fn critical_point_eigenvalue(&self, hessian_eigenvalues: &[f64], index: usize) -> f64 {
        let mut sum = 0.0;
        for (i, &ev) in hessian_eigenvalues.iter().enumerate() {
            if i < index {
                sum += ev.abs(); // negative eigenvalues contribute to unstable modes
            }
        }
        self.t * sum
    }

    /// Tunneling amplitude between critical points p and q.
    /// Proportional to e^{-t |f(p) - f(q)|} for large t.
    pub fn tunneling_amplitude(&self, p_value: f64, q_value: f64) -> f64 {
        (-self.t * (p_value - q_value).abs()).exp()
    }

    /// Compute all tunneling amplitudes between critical points of consecutive index.
    pub fn boundary_tunneling_amplitudes(&self) -> Vec<(usize, usize, f64)> {
        let mut result = Vec::new();
        for i in 0..self.critical_values.len() {
            for j in 0..self.critical_values.len() {
                if self.critical_indices[i] == self.critical_indices[j] + 1 {
                    let amp = self.tunneling_amplitude(self.critical_values[i], self.critical_values[j]);
                    result.push((i, j, amp));
                }
            }
        }
        result
    }

    /// Witten complex boundary operator (approximate, for large t).
    /// The boundary ∂_t in the Witten complex has entries proportional to
    /// e^{-t·(f(p)-f(q))} where p has index k and q has index k-1.
    pub fn witten_boundary_matrix(&self, k: usize) -> Vec<Vec<f64>> {
        let source: Vec<_> = (0..self.critical_values.len())
            .filter(|&i| self.critical_indices[i] == k).collect();
        let target: Vec<_> = (0..self.critical_values.len())
            .filter(|&i| self.critical_indices[i] == k - 1).collect();

        let mut mat = vec![vec![0.0; source.len()]; target.len()];
        for (ti, &tidx) in target.iter().enumerate() {
            for (sj, &sidx) in source.iter().enumerate() {
                mat[ti][sj] = self.tunneling_amplitude(self.critical_values[sidx], self.critical_values[tidx]);
            }
        }
        mat
    }

    /// Reward shaping: use Witten deformation to shape the reward signal.
    /// For agent learning, the Witten-deformed reward at state x is:
    /// R_t(x) = R(x) + t · ∇f(x) · (∇V(x) - t·∇f(x))
    /// where V is the value function and f is the potential.
    pub fn shaped_reward(&self, base_reward: f64, grad_f: &DVector<f64>, grad_v: &DVector<f64>) -> f64 {
        let correction = self.t * grad_f.dot(grad_v) - self.t * self.t * grad_f.dot(grad_f);
        base_reward + correction
    }

    /// Effective potential from Witten deformation: V_eff = ½|∇f|² - ½Δf/t
    /// For large t, the critical points of V_eff coincide with critical points of f.
    pub fn effective_potential(&self, grad_f_norm_sq: f64, laplacian_f: f64) -> f64 {
        0.5 * grad_f_norm_sq - 0.5 * laplacian_f / self.t
    }

    /// Check if t is large enough for the Witten approximation to be valid.
    /// Heuristic: t >> 1/min_gap where min_gap is the minimum function value difference
    /// between critical points.
    pub fn is_large_t(&self) -> bool {
        if self.critical_values.len() < 2 {
            return true;
        }
        let mut min_gap = f64::INFINITY;
        for i in 0..self.critical_values.len() {
            for j in (i+1)..self.critical_values.len() {
                let gap = (self.critical_values[i] - self.critical_values[j]).abs();
                if gap > 1e-10 {
                    min_gap = min_gap.min(gap);
                }
            }
        }
        self.t * min_gap > 5.0
    }

    /// Number of critical points.
    pub fn num_critical_points(&self) -> usize {
        self.critical_values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_witten_creation() {
        let wd = WittenDeformation::new(vec![0.0, 1.0], vec![0, 1], 1);
        assert_eq!(wd.num_critical_points(), 2);
    }

    #[test]
    fn test_deformed_derivative() {
        let wd = WittenDeformation::new(vec![], vec![], 1);
        let grad = DVector::from_vec(vec![1.0, 2.0]);
        let dphi = DVector::from_vec(vec![0.5, 0.5]);
        let result = wd.deformed_derivative_0(&grad, &dphi);
        assert_relative_eq!(result[0], 0.5 - 1.0);
        assert_relative_eq!(result[1], 0.5 - 2.0);
    }

    #[test]
    fn test_deformed_laplacian() {
        let wd = WittenDeformation::new(vec![], vec![], 1);
        let result = wd.deformed_laplacian_0(1.0, 4.0, 2.0);
        // -1.0 + 1*4.0 + 1*2.0 = 5.0
        assert_relative_eq!(result, 5.0);
    }

    #[test]
    fn test_tunneling_amplitude() {
        let wd = WittenDeformation::new(vec![0.0, 1.0], vec![0, 1], 1);
        let amp = wd.tunneling_amplitude(1.0, 0.0);
        assert_relative_eq!(amp, (-1.0f64).exp());
    }

    #[test]
    fn test_tunneling_amplitude_zero_gap() {
        let wd = WittenDeformation::new(vec![0.0, 0.0], vec![0, 1], 1);
        let amp = wd.tunneling_amplitude(0.0, 0.0);
        assert_relative_eq!(amp, 1.0);
    }

    #[test]
    fn test_boundary_tunneling_amplitudes() {
        let wd = WittenDeformation::new(vec![0.0, 1.0, 2.0], vec![0, 1, 2], 1);
        let amps = wd.boundary_tunneling_amplitudes();
        // index 1 → index 0: (1, 0), index 2 → index 1: (2, 1)
        assert_eq!(amps.len(), 2);
    }

    #[test]
    fn test_witten_boundary_matrix() {
        let wd = WittenDeformation::new(vec![0.0, 1.0], vec![0, 1], 1);
        let bm = wd.witten_boundary_matrix(1);
        assert_eq!(bm.len(), 1); // 1 target (index 0)
        assert_eq!(bm[0].len(), 1); // 1 source (index 1)
    }

    #[test]
    fn test_shaped_reward() {
        let wd = WittenDeformation::new(vec![], vec![], 1);
        let grad_f = DVector::from_vec(vec![1.0]);
        let grad_v = DVector::from_vec(vec![2.0]);
        // R = 10 + 1*(1*2) - 1*(1*1) = 10 + 2 - 1 = 11
        let r = wd.shaped_reward(10.0, &grad_f, &grad_v);
        assert_relative_eq!(r, 11.0);
    }

    #[test]
    fn test_effective_potential() {
        let wd = WittenDeformation::new(vec![], vec![], 10);
        // V_eff = 0.5 * 4.0 - 0.5 * 2.0 / 10.0 = 2.0 - 0.1 = 1.9
        let v = wd.effective_potential(4.0, 2.0);
        assert_relative_eq!(v, 1.9);
    }

    #[test]
    fn test_large_t_detection() {
        let wd = WittenDeformation::new(vec![0.0, 1.0], vec![0, 1], 100);
        assert!(wd.is_large_t());
    }

    #[test]
    fn test_small_t_detection() {
        let wd = WittenDeformation::new(vec![0.0, 1.0], vec![0, 1], 0);
        assert!(!wd.is_large_t());
    }

    #[test]
    fn test_critical_point_eigenvalue() {
        let wd = WittenDeformation::new(vec![], vec![], 2);
        let eigvals = vec![-2.0, 3.0];
        // λ ≈ t × |μ_0| = 2.0 × 2.0 = 4.0
        let ev = wd.critical_point_eigenvalue(&eigvals, 1);
        assert_relative_eq!(ev, 4.0);
    }

    #[test]
    fn test_with_t_builder() {
        let mut wd = WittenDeformation::new(vec![], vec![], 1);
        wd.with_t(5.0);
        assert_relative_eq!(wd.t, 5.0);
    }

    #[test]
    fn test_serialization() {
        let wd = WittenDeformation::new(vec![0.0, 1.0], vec![0, 1], 1);
        let json = serde_json::to_string(&wd).unwrap();
        let decoded: WittenDeformation = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.num_critical_points(), 2);
    }

    #[test]
    fn test_single_critical_point_large_t() {
        let wd = WittenDeformation::new(vec![0.0], vec![0], 1);
        assert!(wd.is_large_t()); // Only 1 point, vacuously true
    }
}
