//! Morse-Smale complex: when stable/unstable manifolds intersect transversally.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// A cell in the Morse-Smale complex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorseSmaleCell {
    /// Unique identifier.
    pub id: usize,
    /// Index of the critical point generating the unstable manifold.
    pub source_index: usize,
    /// Index of the critical point generating the stable manifold.
    pub target_index: usize,
    /// Dimension of this cell.
    pub dimension: usize,
    /// Position (center of the cell).
    pub center: DVector<f64>,
}

/// The Morse-Smale complex: decomposition of the manifold into cells
/// from intersecting stable/unstable manifolds of critical points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorseSmaleComplex {
    /// Cells of the complex.
    pub cells: Vec<MorseSmaleCell>,
    /// Adjacency: cell id → list of (neighbor_id, codimension).
    pub adjacency: Vec<(usize, usize, usize)>,
    /// Dimension of the manifold.
    pub dimension: usize,
    /// Number of critical points of each index.
    pub critical_counts: Vec<usize>,
}

impl MorseSmaleComplex {
    /// Create a new empty Morse-Smale complex.
    pub fn new(dimension: usize) -> Self {
        MorseSmaleComplex {
            cells: Vec::new(),
            adjacency: Vec::new(),
            dimension,
            critical_counts: Vec::new(),
        }
    }

    /// Add a critical point count.
    pub fn set_critical_counts(&mut self, counts: Vec<usize>) {
        self.critical_counts = counts;
    }

    /// Add a cell.
    pub fn add_cell(&mut self, cell: MorseSmaleCell) {
        self.cells.push(cell);
    }

    /// Add an adjacency relation.
    pub fn add_adjacency(&mut self, from: usize, to: usize, codim: usize) {
        self.adjacency.push((from, to, codim));
    }

    /// Get cells by dimension.
    pub fn cells_of_dimension(&self, d: usize) -> Vec<&MorseSmaleCell> {
        self.cells.iter().filter(|c| c.dimension == d).collect()
    }

    /// Check transversality condition (simplified).
    /// For a proper Morse-Smale system, stable and unstable manifolds
    /// of distinct critical points must intersect transversally.
    pub fn check_transversality(&self) -> bool {
        // In the ideal case, all cells should have well-defined dimensions
        // and the codimension of adjacency should be 1 (boundary relation).
        self.adjacency.iter().all(|&(_, _, codim)| codim <= 1)
    }

    /// Number of k-cells.
    pub fn num_cells(&self, k: usize) -> usize {
        self.cells_of_dimension(k).len()
    }

    /// Compute the boundary matrix of the CW complex.
    /// Returns (target_dim x source_dim) integer matrix for ∂: C_{k} → C_{k-1}.
    pub fn boundary_matrix(&self, k: usize) -> Vec<Vec<i64>> {
        let target: Vec<_> = self.cells_of_dimension(k - 1).iter().map(|c| c.id).collect();
        let source: Vec<_> = self.cells_of_dimension(k).iter().map(|c| c.id).collect();
        let mut mat = vec![vec![0i64; source.len()]; target.len()];

        for &(from, to, codim) in &self.adjacency {
            if codim == 1 {
                if let (Some(ti), Some(sj)) = (
                    target.iter().position(|&id| id == to),
                    source.iter().position(|&id| id == from),
                ) {
                    mat[ti][sj] += 1;
                }
            }
        }
        mat
    }

    /// Compute cell complex homology (Z/2).
    pub fn compute_homology(&self) -> Vec<usize> {
        if self.cells.is_empty() {
            return Vec::new();
        }
        let max_dim = self.cells.iter().map(|c| c.dimension).max().unwrap_or(0);
        let mut betti = vec![0; max_dim + 1];

        for k in 0..=max_dim {
            let ck = self.num_cells(k);
            let rank_dk = if k > 0 { Self::rank_mod2(&self.boundary_matrix(k)) } else { 0 };
            let rank_dk1 = if k < max_dim { Self::rank_mod2(&self.boundary_matrix(k + 1)) } else { 0 };
            betti[k] = (ck as i64 - rank_dk as i64 - rank_dk1 as i64).max(0) as usize;
        }
        betti
    }

    fn rank_mod2(matrix: &[Vec<i64>]) -> usize {
        if matrix.is_empty() || matrix[0].is_empty() { return 0; }
        let m = matrix.len();
        let n = matrix[0].len();
        let mut mat: Vec<Vec<u64>> = matrix.iter()
            .map(|row| row.iter().map(|&v| (v & 1) as u64).collect())
            .collect();
        let mut rank = 0;
        for col in 0..n {
            let pivot = (rank..m).find(|&r| mat[r][col] == 1);
            if let Some(p) = pivot {
                mat.swap(rank, p);
                for row in 0..m {
                    if row != rank && mat[row][col] == 1 {
                        for c in 0..n { mat[row][c] ^= mat[rank][c]; }
                    }
                }
                rank += 1;
            }
        }
        rank
    }

    /// Verify that the complex satisfies the Morse-Smale transversality condition
    /// and computes the same homology as the Morse chain complex.
    pub fn verify(&self, expected_betti: &[usize]) -> bool {
        self.check_transversality() && self.compute_homology() == expected_betti
    }

    /// Stable manifold of a critical point (cells whose target is this critical point).
    pub fn stable_manifold_cells(&self, cp_id: usize) -> Vec<&MorseSmaleCell> {
        self.cells.iter().filter(|c| c.target_index == cp_id || c.id == cp_id).collect()
    }

    /// Unstable manifold of a critical point (cells whose source is this critical point).
    pub fn unstable_manifold_cells(&self, cp_id: usize) -> Vec<&MorseSmaleCell> {
        self.cells.iter().filter(|c| c.source_index == cp_id || c.id == cp_id).collect()
    }

    /// F-vector: (f_0, f_1, ..., f_n) where f_k = number of k-cells.
    pub fn f_vector(&self) -> Vec<usize> {
        if self.cells.is_empty() { return Vec::new(); }
        let max_dim = self.cells.iter().map(|c| c.dimension).max().unwrap_or(0);
        (0..=max_dim).map(|k| self.num_cells(k)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_interval_complex() -> MorseSmaleComplex {
        // Interval [0,1]: 2 vertices, 1 edge
        let mut msc = MorseSmaleComplex::new(1);
        msc.add_cell(MorseSmaleCell { id: 0, source_index: 0, target_index: 0, dimension: 0, center: DVector::from_vec(vec![0.0]) });
        msc.add_cell(MorseSmaleCell { id: 1, source_index: 0, target_index: 0, dimension: 0, center: DVector::from_vec(vec![1.0]) });
        msc.add_cell(MorseSmaleCell { id: 2, source_index: 1, target_index: 0, dimension: 1, center: DVector::from_vec(vec![0.5]) });
        msc.add_adjacency(2, 0, 1);
        msc.add_adjacency(2, 1, 1);
        msc
    }

    fn build_circle_complex() -> MorseSmaleComplex {
        // Circle: 2 vertices, 2 edges
        let mut msc = MorseSmaleComplex::new(1);
        msc.add_cell(MorseSmaleCell { id: 0, source_index: 0, target_index: 0, dimension: 0, center: DVector::from_vec(vec![0.0]) });
        msc.add_cell(MorseSmaleCell { id: 1, source_index: 1, target_index: 1, dimension: 0, center: DVector::from_vec(vec![3.14]) });
        msc.add_cell(MorseSmaleCell { id: 2, source_index: 1, target_index: 0, dimension: 1, center: DVector::from_vec(vec![1.57]) });
        msc.add_cell(MorseSmaleCell { id: 3, source_index: 1, target_index: 0, dimension: 1, center: DVector::from_vec(vec![4.71]) });
        // Each edge connects the two vertices
        msc.add_adjacency(2, 0, 1);
        msc.add_adjacency(2, 1, 1);
        msc.add_adjacency(3, 0, 1);
        msc.add_adjacency(3, 1, 1);
        msc
    }

    #[test]
    fn test_cells_by_dimension() {
        let msc = build_interval_complex();
        assert_eq!(msc.cells_of_dimension(0).len(), 2);
        assert_eq!(msc.cells_of_dimension(1).len(), 1);
    }

    #[test]
    fn test_f_vector_interval() {
        let msc = build_interval_complex();
        assert_eq!(msc.f_vector(), vec![2, 1]);
    }

    #[test]
    fn test_f_vector_circle() {
        let msc = build_circle_complex();
        assert_eq!(msc.f_vector(), vec![2, 2]);
    }

    #[test]
    fn test_transversality() {
        let msc = build_interval_complex();
        assert!(msc.check_transversality());
    }

    #[test]
    fn test_interval_homology() {
        let msc = build_interval_complex();
        let h = msc.compute_homology();
        // Contractible: H_0 = Z, H_1 = 0
        // But our Z/2 computation with boundary [1,1] gives rank 1
        // so H_0 = 2 - 0 - 1 = 1 (if ∂_1 has rank 1)
        // and H_1 = 1 - 1 - 0 = 0
        assert_eq!(h[0], 1);
    }

    #[test]
    fn test_stable_manifold() {
        let msc = build_interval_complex();
        let sm = msc.stable_manifold_cells(0);
        assert_eq!(sm.len(), 3); // Both vertices + the edge
    }

    #[test]
    fn test_unstable_manifold() {
        let msc = build_interval_complex();
        let um = msc.unstable_manifold_cells(0);
        assert_eq!(um.len(), 2); // vertex 0 + vertex 1
    }

    #[test]
    fn test_num_cells() {
        let msc = build_circle_complex();
        assert_eq!(msc.num_cells(0), 2);
        assert_eq!(msc.num_cells(1), 2);
    }

    #[test]
    fn test_verify() {
        let msc = build_interval_complex();
        assert!(msc.verify(&[1, 0]));
    }

    #[test]
    fn test_serialization() {
        let msc = build_interval_complex();
        let json = serde_json::to_string(&msc).unwrap();
        let decoded: MorseSmaleComplex = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.cells.len(), 3);
    }

    #[test]
    fn test_empty_complex() {
        let msc = MorseSmaleComplex::new(2);
        assert_eq!(msc.compute_homology(), Vec::<usize>::new());
    }

    #[test]
    fn test_boundary_matrix_interval() {
        let msc = build_interval_complex();
        let bm = msc.boundary_matrix(1);
        // ∂: edge → (v0, v1), matrix [[1], [1]]
        assert_eq!(bm, vec![vec![1], vec![1]]);
    }
}
