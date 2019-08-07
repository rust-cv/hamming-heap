//! This is a special heap specifically for hamming space searches.
//!
//! This queue works by having n-bits + 1 vectors, one for each hamming distance. When we find that any item
//! achieves a distance of `n` at the least, we place the index of that node into the vector associated
//! with that distance. Any time we take an item off, we place all of its children into the appropriate
//! distance priorities.
//!
//! We maintain the lowest weight vector at any given time in the queue. When a vector runs out,
//! we iterate until we find the next-best non-empty distance vector.

use generic_array::{ArrayLength, GenericArray};
use std::fmt;

#[derive(Clone)]
pub struct HammingHeap<W, T>
where
    W: ArrayLength<Vec<T>>,
{
    distances: GenericArray<Vec<T>, W>,
    best: u32,
}

impl<W, T> HammingHeap<W, T>
where
    W: ArrayLength<Vec<T>>,
{
    /// This allows the queue to be cleared so that we don't need to reallocate memory.
    pub fn clear(&mut self) {
        for v in self.distances[self.best as usize..].iter_mut() {
            v.clear();
        }
        self.best = 0;
    }

    /// This removes the nearest candidate from the queue.
    #[inline]
    pub fn pop(&mut self) -> Option<(u32, T)> {
        loop {
            if let Some(node) = self.distances[self.best as usize].pop() {
                return Some((self.best, node));
            } else if self.best == W::to_u32() - 1 {
                return None;
            } else {
                self.best += 1;
            }
        }
    }

    /// Inserts a node.
    #[inline]
    pub fn push(&mut self, distance: u32, node: T) {
        if distance < self.best {
            self.best = distance;
        }
        self.distances[distance as usize].push(node);
    }

    /// Returns the best distance if not empty.
    pub fn best(&self) -> Option<u32> {
        self.distances[self.best as usize..]
            .iter()
            .position(|v| !v.is_empty())
            .map(|n| n as u32 + self.best)
    }

    /// Iterate over the entire queue in best-to-worse order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.distances[self.best as usize..]
            .iter()
            .enumerate()
            .flat_map(|(distance, v)| v.iter().map(move |item| (distance as u32, item)))
    }
    /// Iterate over the entire queue in best-to-worse order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u32, &mut T)> {
        self.distances[self.best as usize..]
            .iter_mut()
            .enumerate()
            .flat_map(|(distance, v)| v.iter_mut().map(move |item| (distance as u32, item)))
    }
}

impl<W, T> fmt::Debug for HammingHeap<W, T>
where
    W: ArrayLength<Vec<T>>,
    T: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.distances[..].fmt(formatter)
    }
}

impl<W, T> Default for HammingHeap<W, T>
where
    W: ArrayLength<Vec<T>>,
{
    fn default() -> Self {
        Self {
            distances: std::iter::repeat_with(|| vec![]).collect(),
            best: 0,
        }
    }
}
