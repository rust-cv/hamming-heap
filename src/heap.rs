//! This is a special heap specifically for 128-bit hamming space searches.
//!
//! This queue works by having 129 vectors, one for each distance. When we find that any item
//! achieves a distance of `n` at the least, we place the index of that node into the vector associated
//! with that distance. Any time we take an item off, we place all of its children into the appropriate
//! distance priorities.
//!
//! We maintain the lowest weight vector at any given time in the queue. When a vector runs out,
//! because of the greedy nature of the search algorithm, we are guaranteed that nothing will ever have a distance
//! lower than the previous candidates. This means we only have to move the lowest weight vector forwards.
//! Also, typically every removal will be constant time since we are incredibly likely to find all the nearest
//! neighbors required before we reach a distance of 64, which is the lowest possible max distance in the root node
//! (distances of the hamming weights 0-64 and 64-128) and the average distance between two random bit strings.
//! The more things in the search, the less likely this becomes. Assuming randomly distributed features, we expect
//! half of the features to have a distance below 64, so it is incredibly likely that all removals are constant time
//! since we will always encounter a removal below or equal to 64.

use std::fmt;

type Distances<T> = [Vec<T>; 129];

#[derive(Clone)]
pub struct HammingHeap128<T> {
    distances: Distances<T>,
    best: u32,
}

impl<T> HammingHeap128<T> {
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
            } else if self.best == 128 {
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
    pub fn iter(&self) -> impl Iterator<Item = (&T, u32)> {
        self.distances[self.best as usize..]
            .iter()
            .enumerate()
            .flat_map(|(distance, v)| v.iter().map(move |item| (item, distance as u32)))
    }
    /// Iterate over the entire queue in best-to-worse order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&mut T, u32)> {
        self.distances[self.best as usize..]
            .iter_mut()
            .enumerate()
            .flat_map(|(distance, v)| v.iter_mut().map(move |item| (item, distance as u32)))
    }
}

impl<T> fmt::Debug for HammingHeap128<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.distances[..].fmt(formatter)
    }
}

impl<T> Default for HammingHeap128<T> {
    fn default() -> Self {
        Self {
            distances: [
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            ],
            best: 0,
        }
    }
}
