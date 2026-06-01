//! # lau-morse-homology-agents
//!
//! Morse theory applied to agent fitness landscapes.
//! The topology of the landscape determines the agent's learning dynamics.
//!
//! Critical points of the fitness function (where gradient = 0) reveal the
//! structure of possible agent behaviors. Morse theory connects critical
//! points to the topology of the underlying manifold.

pub mod morse_function;
pub mod morse_lemma;
pub mod morse_index;
pub mod morse_inequalities;
pub mod morse_homology;
pub mod morse_smale;
pub mod witten;
pub mod fitness_landscape;
pub mod gradient_flow;
pub mod nash_equilibrium;

pub use morse_function::{MorseFunction, CriticalPoint, CriticalPointType};
pub use morse_lemma::MorseLemmaCoordinates;
pub use morse_index::MorseIndex;
pub use morse_inequalities::MorseInequalities;
pub use morse_homology::{MorseChainComplex, MorseHomology as MorseHomologyGroups};
pub use morse_smale::MorseSmaleComplex;
pub use witten::WittenDeformation;
pub use fitness_landscape::{FitnessLandscape, AgentState};
pub use gradient_flow::GradientFlow;
pub use nash_equilibrium::{NashEquilibriumCounter, EquilibriumInfo};
