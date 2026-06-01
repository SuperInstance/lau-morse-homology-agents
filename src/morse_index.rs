//! Morse index: number of negative eigenvalues of the Hessian at a critical point.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// The Morse index of a critical point.
/// Defined as the number of negative eigenvalues of the Hessian.
/// Equals the dimension of the unstable manifold of the gradient flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorseIndex {
    /// The index value.
    pub index: usize,
    /// Dimension of the manifold.
    pub dimension: usize,
    /// Eigenvalues of the Hessian.
    pub eigenvalues: DVector<f64>,
    /// Eigenvectors of the Hessian (columns).
    pub eigenvectors: DMatrix<f64>,
}

impl MorseIndex {
    /// Compute the Morse index from a Hessian matrix.
    pub fn from_hessian(hessian: &DMatrix<f64>) -> Self {
        let dim = hessian.nrows();
        let decomp = hessian.clone().symmetric_eigen();
        let eigenvalues = decomp.eigenvalues.clone();
        let eigenvectors = decomp.eigenvectors.clone();
        let index = eigenvalues.iter().filter(|&&v| v < 0.0).count();

        MorseIndex {
            index,
            dimension: dim,
            eigenvalues,
            eigenvectors,
        }
    }

    /// The co-index: dimension of the stable manifold.
    pub fn co_index(&self) -> usize {
        self.dimension - self.index
    }

    /// Number of zero eigenvalues (should be 0 for non-degenerate).
    pub fn nullity(&self) -> usize {
        self.eigenvalues.iter().filter(|&&v| v.abs() < 1e-10).count()
    }

    /// Is the critical point non-degenerate?
    pub fn is_non_degenerate(&self) -> bool {
        self.nullity() == 0
    }

    /// Signature: (number of negative, zero, positive eigenvalues).
    pub fn signature(&self) -> (usize, usize, usize) {
        let neg = self.eigenvalues.iter().filter(|&&v| v < 0.0).count();
        let zero = self.nullity();
        let pos = self.dimension - neg - zero;
        (neg, zero, pos)
    }

    /// Unstable manifold dimension (= Morse index).
    pub fn unstable_manifold_dim(&self) -> usize {
        self.index
    }

    /// Stable manifold dimension (= co-index).
    pub fn stable_manifold_dim(&self) -> usize {
        self.co_index()
    }

    /// The negative eigenspace basis (unstable directions).
    pub fn unstable_basis(&self) -> DMatrix<f64> {
        let neg_indices: Vec<_> = (0..self.dimension)
            .filter(|&i| self.eigenvalues[i] < 0.0)
            .collect();
        if neg_indices.is_empty() {
            return DMatrix::zeros(self.dimension, 0);
        }
        let mut basis = DMatrix::zeros(self.dimension, neg_indices.len());
        for (j, &i) in neg_indices.iter().enumerate() {
            basis.set_column(j, &self.eigenvectors.column(i));
        }
        basis
    }

    /// The positive eigenspace basis (stable directions).
    pub fn stable_basis(&self) -> DMatrix<f64> {
        let pos_indices: Vec<_> = (0..self.dimension)
            .filter(|&i| self.eigenvalues[i] > 0.0)
            .collect();
        if pos_indices.is_empty() {
            return DMatrix::zeros(self.dimension, 0);
        }
        let mut basis = DMatrix::zeros(self.dimension, pos_indices.len());
        for (j, &i) in pos_indices.iter().enumerate() {
            basis.set_column(j, &self.eigenvectors.column(i));
        }
        basis
    }

    /// Compute the Conley index (related to the Morse index).
    /// For non-degenerate critical points, the Conley index is that of a k-sphere.
    pub fn conley_index_homology(&self) -> Vec<Option<usize>> {
        // Homology of S^k: H_0 = Z, H_k = Z, all others = 0
        let mut homology = vec![None; self.index + 1];
        if self.index > 0 {
            homology[0] = Some(1);
            homology[self.index] = Some(1);
        } else {
            homology[0] = Some(1);
        }
        homology
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_morse_index_identity() {
        // Identity Hessian → all positive → index 0
        let h = DMatrix::identity(3, 3);
        let mi = MorseIndex::from_hessian(&h);
        assert_eq!(mi.index, 0);
        assert!(mi.is_non_degenerate());
    }

    #[test]
    fn test_morse_index_negative_identity() {
        // -I → all negative → index = dim
        let h = -DMatrix::identity(3, 3);
        let mi = MorseIndex::from_hessian(&h);
        assert_eq!(mi.index, 3);
    }

    #[test]
    fn test_morse_index_saddle_2d() {
        let h = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
        let mi = MorseIndex::from_hessian(&h);
        assert_eq!(mi.index, 1);
    }

    #[test]
    fn test_co_index() {
        let h = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
        let mi = MorseIndex::from_hessian(&h);
        assert_eq!(mi.co_index(), 1);
    }

    #[test]
    fn test_signature() {
        let h = DMatrix::from_vec(3, 3, vec![
            -2.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, -1.0,
        ]);
        let mi = MorseIndex::from_hessian(&h);
        assert_eq!(mi.signature(), (2, 0, 1));
    }

    #[test]
    fn test_stable_unstable_dimensions() {
        let h = DMatrix::from_vec(3, 3, vec![
            -1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, -3.0,
        ]);
        let mi = MorseIndex::from_hessian(&h);
        assert_eq!(mi.unstable_manifold_dim(), 2);
        assert_eq!(mi.stable_manifold_dim(), 1);
    }

    #[test]
    fn test_unstable_basis() {
        let h = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
        let mi = MorseIndex::from_hessian(&h);
        let ub = mi.unstable_basis();
        assert_eq!(ub.ncols(), 1);
        assert_eq!(ub.nrows(), 2);
    }

    #[test]
    fn test_stable_basis() {
        let h = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
        let mi = MorseIndex::from_hessian(&h);
        let sb = mi.stable_basis();
        assert_eq!(sb.ncols(), 1);
    }

    #[test]
    fn test_minimum_stable_basis_is_full() {
        let h = DMatrix::identity(3, 3);
        let mi = MorseIndex::from_hessian(&h);
        assert_eq!(mi.stable_basis().ncols(), 3);
        assert_eq!(mi.unstable_basis().ncols(), 0);
    }

    #[test]
    fn test_conley_index_minimum() {
        let h = DMatrix::identity(2, 2);
        let mi = MorseIndex::from_hessian(&h);
        let ch = mi.conley_index_homology();
        assert_eq!(ch[0], Some(1)); // H_0(S^0) = Z
    }

    #[test]
    fn test_conley_index_saddle() {
        let h = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
        let mi = MorseIndex::from_hessian(&h);
        let ch = mi.conley_index_homology();
        assert_eq!(ch[0], Some(1)); // H_0(S^1) = Z
        assert_eq!(ch[1], Some(1)); // H_1(S^1) = Z
    }

    #[test]
    fn test_serialization() {
        let h = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
        let mi = MorseIndex::from_hessian(&h);
        let json = serde_json::to_string(&mi).unwrap();
        let decoded: MorseIndex = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.index, 1);
    }
}
