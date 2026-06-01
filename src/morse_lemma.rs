//! Morse lemma: near critical points, f(x) = f(p) - x₁² - ... - x_λ² + x_{λ+1}² + ... + x_n²

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// Represents the Morse lemma coordinate system near a critical point.
/// Near a non-degenerate critical point p, there exist coordinates such that
/// f(x) = f(p) - y₁² - ... - y_λ² + y_{λ+1}² + ... + y_n²
/// where λ is the Morse index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorseLemmaCoordinates {
    /// The critical point position.
    pub critical_point: DVector<f64>,
    /// Function value at the critical point.
    pub critical_value: f64,
    /// Morse index (number of descending directions).
    pub morse_index: usize,
    /// Dimension of the ambient space.
    pub dimension: usize,
    /// Coordinate transformation matrix Q (orthogonal).
    pub transformation: DMatrix<f64>,
    /// Eigenvalues of the Hessian at the critical point.
    pub eigenvalues: DVector<f64>,
}

impl MorseLemmaCoordinates {
    /// Construct Morse lemma coordinates from a critical point's Hessian.
    /// Computes the eigendecomposition and orthogonal transformation.
    pub fn from_hessian(
        position: DVector<f64>,
        value: f64,
        hessian: &DMatrix<f64>,
    ) -> Self {
        let dim = position.len();
        let eigendecomp = hessian.clone().symmetric_eigen();
        let eigenvalues = eigendecomp.eigenvalues.clone();
        let transformation = eigendecomp.eigenvectors.clone();
        let morse_index = eigenvalues.iter().filter(|&&v| v < 0.0).count();

        MorseLemmaCoordinates {
            critical_point: position,
            critical_value: value,
            morse_index,
            dimension: dim,
            transformation,
            eigenvalues,
        }
    }

    /// Evaluate the Morse normal form at coordinates y.
    /// f(p) + Σ (-1 for first λ eigenvalues) * yᵢ²
    pub fn evaluate_normal_form(&self, y: &DVector<f64>) -> f64 {
        let mut result = self.critical_value;
        for i in 0..y.len().min(self.dimension) {
            if i < self.morse_index {
                result -= y[i] * y[i];
            } else {
                result += y[i] * y[i];
            }
        }
        result
    }

    /// Convert from original coordinates x to Morse lemma coordinates y.
    /// y = Q^T (x - p)
    pub fn to_morse_coords(&self, x: &DVector<f64>) -> DVector<f64> {
        let delta = x - &self.critical_point;
        &self.transformation.transpose() * delta
    }

    /// Convert from Morse lemma coordinates y back to original coordinates x.
    /// x = Q y + p
    pub fn from_morse_coords(&self, y: &DVector<f64>) -> DVector<f64> {
        &self.transformation * y + &self.critical_point
    }

    /// Verify the Morse lemma: compare the actual function value with the normal form.
    pub fn verify(&self, f_actual: f64, x: &DVector<f64>) -> f64 {
        let y = self.to_morse_coords(x);
        let f_normal = self.evaluate_normal_form(&y);
        (f_actual - f_normal).abs()
    }

    /// The descending directions (negative eigenvalue eigenvectors).
    pub fn descending_directions(&self) -> DMatrix<f64> {
        let neg_cols: Vec<_> = (0..self.dimension)
            .filter(|&i| i < self.morse_index)
            .collect();
        if neg_cols.is_empty() {
            return DMatrix::zeros(self.dimension, 0);
        }
        let mut cols = DMatrix::zeros(self.dimension, neg_cols.len());
        for (j, &i) in neg_cols.iter().enumerate() {
            cols.set_column(j, &self.transformation.column(i));
        }
        cols
    }

    /// The ascending directions (positive eigenvalue eigenvectors).
    pub fn ascending_directions(&self) -> DMatrix<f64> {
        let pos_cols: Vec<_> = (self.morse_index..self.dimension).collect();
        if pos_cols.is_empty() {
            return DMatrix::zeros(self.dimension, 0);
        }
        let mut cols = DMatrix::zeros(self.dimension, pos_cols.len());
        for (j, &i) in pos_cols.iter().enumerate() {
            cols.set_column(j, &self.transformation.column(i));
        }
        cols
    }

    /// Isotropy group dimension at the critical point.
    pub fn isotropy_dimension(&self) -> usize {
        self.dimension - self.morse_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_morse_lemma_simple_minimum() {
        // f(x) = x^2 at origin, Hessian = [[2]], index = 0
        let pos = DVector::from_vec(vec![0.0]);
        let hessian = DMatrix::from_vec(1, 1, vec![2.0]);
        let mlc = MorseLemmaCoordinates::from_hessian(pos, 0.0, &hessian);
        assert_eq!(mlc.morse_index, 0);
    }

    #[test]
    fn test_morse_lemma_2d_saddle() {
        // Saddle at origin: Hessian = [[1, 0], [0, -1]]
        let pos = DVector::from_vec(vec![0.0, 0.0]);
        let hessian = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
        let mlc = MorseLemmaCoordinates::from_hessian(pos, 0.0, &hessian);
        assert_eq!(mlc.morse_index, 1);
    }

    #[test]
    fn test_normal_form_evaluation() {
        let pos = DVector::from_vec(vec![0.0]);
        let hessian = DMatrix::from_vec(1, 1, vec![2.0]);
        let mlc = MorseLemmaCoordinates::from_hessian(pos, 5.0, &hessian);
        // Normal form: 5 + y^2 (index 0, so all positive)
        let y = DVector::from_vec(vec![3.0]);
        assert_relative_eq!(mlc.evaluate_normal_form(&y), 14.0);
    }

    #[test]
    fn test_coordinate_roundtrip() {
        let pos = DVector::from_vec(vec![1.0, 2.0]);
        let hessian = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
        let mlc = MorseLemmaCoordinates::from_hessian(pos, 0.0, &hessian);
        let x = DVector::from_vec(vec![3.0, 4.0]);
        let y = mlc.to_morse_coords(&x);
        let x_back = mlc.from_morse_coords(&y);
        assert_relative_eq!(x_back[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(x_back[1], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_descending_directions() {
        let pos = DVector::from_vec(vec![0.0, 0.0]);
        let hessian = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
        let mlc = MorseLemmaCoordinates::from_hessian(pos, 0.0, &hessian);
        let desc = mlc.descending_directions();
        assert_eq!(desc.ncols(), 1);
    }

    #[test]
    fn test_ascending_directions() {
        let pos = DVector::from_vec(vec![0.0, 0.0]);
        let hessian = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
        let mlc = MorseLemmaCoordinates::from_hessian(pos, 0.0, &hessian);
        let asc = mlc.ascending_directions();
        assert_eq!(asc.ncols(), 1);
    }

    #[test]
    fn test_maximum_all_descending() {
        let pos = DVector::from_vec(vec![0.0, 0.0]);
        let hessian = -DMatrix::identity(2, 2);
        let mlc = MorseLemmaCoordinates::from_hessian(pos, 1.0, &hessian);
        assert_eq!(mlc.morse_index, 2);
        let desc = mlc.descending_directions();
        assert_eq!(desc.ncols(), 2);
        let asc = mlc.ascending_directions();
        assert_eq!(asc.ncols(), 0);
    }

    #[test]
    fn test_isotropy_dimension() {
        let pos = DVector::from_vec(vec![0.0, 0.0]);
        let hessian = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
        let mlc = MorseLemmaCoordinates::from_hessian(pos, 0.0, &hessian);
        assert_eq!(mlc.isotropy_dimension(), 1);
    }

    #[test]
    fn test_serialization() {
        let pos = DVector::from_vec(vec![1.0, 2.0]);
        let hessian = DMatrix::identity(2, 2);
        let mlc = MorseLemmaCoordinates::from_hessian(pos, 3.0, &hessian);
        let json = serde_json::to_string(&mlc).unwrap();
        let decoded: MorseLemmaCoordinates = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.morse_index, 0);
        assert_eq!(decoded.dimension, 2);
    }
}
