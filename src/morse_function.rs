//! Morse function: smooth function on agent state space with non-degenerate critical points.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// Type classification of a critical point based on its Morse index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CriticalPointType {
    Minimum,
    Maximum,
    Saddle,
}

impl CriticalPointType {
    pub fn from_index(index: usize, dim: usize) -> Self {
        if index == 0 {
            CriticalPointType::Minimum
        } else if index == dim {
            CriticalPointType::Maximum
        } else {
            CriticalPointType::Saddle
        }
    }
}

/// A critical point of a Morse function on the agent state space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPoint {
    /// Coordinates of the critical point.
    pub position: DVector<f64>,
    /// Function value at the critical point.
    pub value: f64,
    /// Morse index (number of negative eigenvalues of the Hessian).
    pub index: usize,
    /// Hessian matrix at the critical point.
    pub hessian: DMatrix<f64>,
}

impl CriticalPoint {
    /// Classify this critical point.
    pub fn point_type(&self) -> CriticalPointType {
        CriticalPointType::from_index(self.index, self.position.len())
    }

    /// Check that the Hessian is non-degenerate (all eigenvalues nonzero).
    pub fn is_non_degenerate(&self) -> bool {
        if self.hessian.nrows() == 0 {
            return false;
        }
        let eigenvalues = self.hessian.symmetric_eigenvalues();
        eigenvalues.iter().all(|&v| v.abs() > 1e-10)
    }

    /// Stability parameter: ratio of negative to positive eigenvalues.
    pub fn stability(&self) -> f64 {
        let eigenvalues = self.hessian.symmetric_eigenvalues();
        let neg: f64 = eigenvalues.iter().filter(|&&v| v < 0.0).map(|&v| v.abs()).sum();
        let pos: f64 = eigenvalues.iter().filter(|&&v| v > 0.0).map(|v| *v).sum();
        if pos.abs() < 1e-14 {
            return f64::INFINITY;
        }
        neg / pos
    }
}

/// A Morse function defined on agent state space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorseFunction {
    /// Dimension of the agent state space.
    pub dimension: usize,
    /// Known critical points.
    pub critical_points: Vec<CriticalPoint>,
    /// The function evaluation closure is not serializable; use coefficients for polynomial.
    /// Stored as monomial coefficients for a polynomial Morse function.
    /// coefficient[i] multiplies the monomial x^i (in 1D) or encodes multi-index in higher dim.
    pub coefficients: Vec<f64>,
}

impl MorseFunction {
    /// Create a new Morse function with the given dimension.
    pub fn new(dimension: usize) -> Self {
        MorseFunction {
            dimension,
            critical_points: Vec::new(),
            coefficients: Vec::new(),
        }
    }

    /// Create a polynomial Morse function from coefficients.
    pub fn from_polynomial(dimension: usize, coefficients: Vec<f64>) -> Self {
        MorseFunction {
            dimension,
            critical_points: Vec::new(),
            coefficients,
        }
    }

    /// Evaluate the polynomial Morse function at a point (1D).
    pub fn evaluate_1d(&self, x: f64) -> f64 {
        self.coefficients.iter().enumerate()
            .map(|(i, &c)| c * x.powi(i as i32))
            .sum()
    }

    /// Gradient (1D polynomial).
    pub fn gradient_1d(&self, x: f64) -> f64 {
        self.coefficients.iter().enumerate()
            .skip(1)
            .map(|(i, &c)| c * (i as f64) * x.powi((i - 1) as i32))
            .sum()
    }

    /// Hessian (1D, second derivative).
    pub fn hessian_1d(&self, x: f64) -> f64 {
        self.coefficients.iter().enumerate()
            .skip(2)
            .map(|(i, &c)| c * (i as f64) * ((i - 1) as f64) * x.powi((i - 2) as i32))
            .sum()
    }

    /// Find critical points of a 1D polynomial using Newton's method from multiple starting points.
    pub fn find_critical_points_1d(&mut self, search_range: (f64, f64), num_starts: usize) -> usize {
        let (lo, hi) = search_range;
        let mut found = Vec::new();
        for i in 0..num_starts {
            let mut x = lo + (hi - lo) * (i as f64) / ((num_starts - 1).max(1) as f64);
            // Newton's method on gradient = 0
            for _ in 0..200 {
                let g = self.gradient_1d(x);
                let h = self.hessian_1d(x);
                if h.abs() < 1e-14 {
                    break;
                }
                x -= g / h;
            }
            // Check convergence
            if self.gradient_1d(x).abs() < 1e-8 {
                // Check not duplicate
                if !found.iter().any(|(px, _): &(f64, CriticalPoint)| (*px - x).abs() < 1e-6) {
                    let val = self.evaluate_1d(x);
                    let hess = self.hessian_1d(x);
                    let index = if hess < 0.0 { 1 } else { 0 };
                    let cp = CriticalPoint {
                        position: DVector::from_vec(vec![x]),
                        value: val,
                        index,
                        hessian: DMatrix::from_vec(1, 1, vec![hess]),
                    };
                    found.push((x, cp));
                }
            }
        }
        let count = found.len();
        self.critical_points = found.into_iter().map(|(_, cp)| cp).collect();
        count
    }

    /// Evaluate a quadratic form f(x) = x^T A x + b^T x + c.
    pub fn evaluate_quadratic(&self, x: &DVector<f64>, a: &DMatrix<f64>, b: &DVector<f64>, c: f64) -> f64 {
        (x.transpose() * a * x)[(0,0)] + (b.transpose() * x)[(0,0)] + c
    }

    /// Gradient of quadratic form.
    pub fn gradient_quadratic(a: &DMatrix<f64>, b: &DVector<f64>, x: &DVector<f64>) -> DVector<f64> {
        (a + a.transpose()) * x + b
    }

    /// Hessian of quadratic form.
    pub fn hessian_quadratic(a: &DMatrix<f64>) -> DMatrix<f64> {
        a + a.transpose()
    }

    /// Verify that all critical points are non-degenerate.
    pub fn verify_non_degenerate(&self) -> bool {
        self.critical_points.iter().all(|cp| cp.is_non_degenerate())
    }

    /// Add a critical point manually.
    pub fn add_critical_point(&mut self, cp: CriticalPoint) {
        self.critical_points.push(cp);
    }

    /// Get critical points by index.
    pub fn critical_points_of_index(&self, index: usize) -> Vec<&CriticalPoint> {
        self.critical_points.iter().filter(|cp| cp.index == index).collect()
    }

    /// Morse polynomial: M_k = number of critical points of index k.
    pub fn morse_polynomial(&self) -> Vec<usize> {
        if self.critical_points.is_empty() {
            return Vec::new();
        }
        let max_index = self.critical_points.iter().map(|cp| cp.index).max().unwrap_or(0);
        let mut mp = vec![0usize; max_index + 1];
        for cp in &self.critical_points {
            mp[cp.index] += 1;
        }
        mp
    }

    /// Total number of critical points.
    pub fn total_critical_points(&self) -> usize {
        self.critical_points.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_critical_point_type_classification() {
        assert_eq!(CriticalPointType::from_index(0, 3), CriticalPointType::Minimum);
        assert_eq!(CriticalPointType::from_index(3, 3), CriticalPointType::Maximum);
        assert_eq!(CriticalPointType::from_index(1, 3), CriticalPointType::Saddle);
        assert_eq!(CriticalPointType::from_index(2, 3), CriticalPointType::Saddle);
    }

    #[test]
    fn test_non_degenerate_check() {
        let cp = CriticalPoint {
            position: DVector::from_vec(vec![0.0]),
            value: 0.0,
            index: 0,
            hessian: DMatrix::from_vec(1, 1, vec![2.0]),
        };
        assert!(cp.is_non_degenerate());
    }

    #[test]
    fn test_degenerate_hessian() {
        let cp = CriticalPoint {
            position: DVector::from_vec(vec![0.0]),
            value: 0.0,
            index: 0,
            hessian: DMatrix::from_vec(1, 1, vec![0.0]),
        };
        assert!(!cp.is_non_degenerate());
    }

    #[test]
    fn test_polynomial_evaluation() {
        // f(x) = x^2 - 2x + 1 = (x-1)^2
        let mf = MorseFunction::from_polynomial(1, vec![1.0, -2.0, 1.0]);
        assert_relative_eq!(mf.evaluate_1d(0.0), 1.0);
        assert_relative_eq!(mf.evaluate_1d(1.0), 0.0);
    }

    #[test]
    fn test_gradient_1d() {
        // f(x) = x^2, f'(x) = 2x
        let mf = MorseFunction::from_polynomial(1, vec![0.0, 0.0, 1.0]);
        assert_relative_eq!(mf.gradient_1d(3.0), 6.0);
    }

    #[test]
    fn test_hessian_1d() {
        // f(x) = x^2, f''(x) = 2
        let mf = MorseFunction::from_polynomial(1, vec![0.0, 0.0, 1.0]);
        assert_relative_eq!(mf.hessian_1d(3.0), 2.0);
    }

    #[test]
    fn test_find_critical_points_parabola() {
        // f(x) = x^2, single critical point at 0
        let mut mf = MorseFunction::from_polynomial(1, vec![0.0, 0.0, 1.0]);
        let count = mf.find_critical_points_1d((-10.0, 10.0), 20);
        assert_eq!(count, 1);
        assert_relative_eq!(mf.critical_points[0].position[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_find_critical_points_cubic() {
        // f(x) = x^3/3 - x, critical points at x = ±1
        // f'(x) = x^2 - 1, f''(x) = 2x
        // Coefficients: [0, -1, 0, 1/3]
        let mut mf = MorseFunction::from_polynomial(1, vec![0.0, -1.0, 0.0, 1.0/3.0]);
        let count = mf.find_critical_points_1d((-5.0, 5.0), 30);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_morse_polynomial() {
        let mut mf = MorseFunction::new(2);
        mf.add_critical_point(CriticalPoint {
            position: DVector::from_vec(vec![0.0, 0.0]),
            value: 0.0,
            index: 0,
            hessian: DMatrix::identity(2, 2),
        });
        mf.add_critical_point(CriticalPoint {
            position: DVector::from_vec(vec![1.0, 0.0]),
            value: 1.0,
            index: 1,
            hessian: DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]),
        });
        mf.add_critical_point(CriticalPoint {
            position: DVector::from_vec(vec![0.0, 1.0]),
            value: 2.0,
            index: 1,
            hessian: DMatrix::from_vec(2, 2, vec![-1.0, 0.0, 0.0, 1.0]),
        });
        mf.add_critical_point(CriticalPoint {
            position: DVector::from_vec(vec![1.0, 1.0]),
            value: 3.0,
            index: 2,
            hessian: -DMatrix::identity(2, 2),
        });
        let mp = mf.morse_polynomial();
        assert_eq!(mp, vec![1, 2, 1]);
    }

    #[test]
    fn test_quadratic_evaluation() {
        let a = DMatrix::identity(2, 2);
        let b = DVector::from_vec(vec![0.0, 0.0]);
        let mf = MorseFunction::new(2);
        let x = DVector::from_vec(vec![1.0, 2.0]);
        // f(x) = x^T I x = 1 + 4 = 5
        assert_relative_eq!(mf.evaluate_quadratic(&x, &a, &b, 0.0), 5.0);
    }

    #[test]
    fn test_verify_non_degenerate() {
        let mut mf = MorseFunction::new(1);
        mf.add_critical_point(CriticalPoint {
            position: DVector::from_vec(vec![0.0]),
            value: 0.0,
            index: 0,
            hessian: DMatrix::from_vec(1, 1, vec![2.0]),
        });
        assert!(mf.verify_non_degenerate());
    }

    #[test]
    fn test_critical_points_of_index() {
        let mut mf = MorseFunction::new(2);
        mf.add_critical_point(CriticalPoint {
            position: DVector::from_vec(vec![0.0, 0.0]),
            value: 0.0,
            index: 0,
            hessian: DMatrix::identity(2, 2),
        });
        mf.add_critical_point(CriticalPoint {
            position: DVector::from_vec(vec![1.0, 0.0]),
            value: 1.0,
            index: 1,
            hessian: DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]),
        });
        assert_eq!(mf.critical_points_of_index(0).len(), 1);
        assert_eq!(mf.critical_points_of_index(1).len(), 1);
        assert_eq!(mf.critical_points_of_index(2).len(), 0);
    }

    #[test]
    fn test_stability() {
        let cp = CriticalPoint {
            position: DVector::from_vec(vec![0.0, 0.0]),
            value: 0.0,
            index: 1,
            hessian: DMatrix::from_vec(2, 2, vec![-2.0, 0.0, 0.0, 3.0]),
        };
        let s = cp.stability();
        assert_relative_eq!(s, 2.0 / 3.0);
    }
}
