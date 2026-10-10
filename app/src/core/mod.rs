//! Domain logic. Nothing here touches the network directly; external
//! services arrive through the traits in [`crate::api`].

pub mod dataset;
pub mod domain;
pub mod evaluation;
pub mod experts;
pub mod pipeline;
pub mod report;
pub mod router;
pub mod segmenter;
pub mod underwriting;

#[cfg(test)]
pub(crate) mod testing;
