use generic_array::{ArrayLength, GenericArray};
use std::fmt;

/// This keeps the nearest `cap` items at all times.
///
/// This heap is not intended to be popped. Instead, this maintains the best `cap` items, and then when you are
/// done adding items, you may fill a slice or iterate over the results. Theoretically, this could also allow
/// popping elements in constant time, but that would incur a performance penalty for the highly specialized
/// purpose this serves. This is specifically tailored for doing hamming space nearest neighbor searches.
#[derive(Clone)]
pub struct FixedHammingHeap<W, T>
where
    W: ArrayLength<Vec<T>>,
{
    cap: usize,
    size: usize,
    worst: u32,
    distances: GenericArray<Vec<T>, W>,
}

impl<W, T> FixedHammingHeap<W, T>
where
    W: ArrayLength<Vec<T>>,
{
    /// This sets the capacity of the queue to `cap`, meaning that adding items to the queue will eject the worst ones
    /// if they are better once `cap` is reached. If the capacity is lowered, this removes the worst elements to
    /// keep `size == cap`.
    pub fn set_capacity(&mut self, cap: usize) {
        assert_ne!(cap, 0);
        self.set_len(cap);
        self.cap = cap;
        // After the capacity is changed, if the size now equals the capacity we need to update the worst because it must
        // actually be set to the worst item.
        self.worst = W::to_u32() - 1;
        if self.size == self.cap {
            self.update_worst();
        }
    }

    /// This removes elements until it reaches `len`. If `len` is higher than the current
    /// number of elements, this does nothing. If the len is lowered, this will unconditionally allow insertions
    /// until `cap` is reached.
    pub fn set_len(&mut self, len: usize) {
        if len == 0 {
            let end = self.end();
            for v in &mut self.distances[..=end] {
                v.clear();
            }
            self.size = 0;
            self.worst = W::to_u32() - 1;
        } else if len < self.size {
            // Remove the difference between them.
            let end = self.end();
            let mut remaining = self.size - len;
            for vec in &mut self.distances[..=end] {
                if vec.len() >= remaining {
                    // This has enough, remove them then break.
                    vec.drain(vec.len() - remaining..);
                    break;
                } else {
                    // There werent enough, so remove everything and move on.
                    remaining -= vec.len();
                    vec.clear();
                }
            }
            // When len is less than the cap, worst must be set to max.
            self.worst = W::to_u32() - 1;
            self.size = len;
        }
    }

    /// Gets the `len` or `size` of the heap.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Checks if the heap is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Clear the queue while maintaining the allocated memory.
    pub fn clear(&mut self) {
        let end = self.end();
        for v in self.distances[..=end].iter_mut() {
            v.clear();
        }
        self.size = 0;
        self.worst = W::to_u32() - 1;
    }

    /// Add a feature to the search.
    ///
    /// Returns true if it was added.
    pub fn push(&mut self, distance: u32, item: T) -> bool {
        if self.size != self.cap {
            self.distances[distance as usize].push(item);
            self.size += 1;
            // Set the worst feature appropriately.
            if self.size == self.cap {
                self.update_worst();
            }
            true
        } else {
            unsafe { self.push_at_cap(distance, item) }
        }
    }

    /// Fill a slice with the `top` elements and return the part of the slice written.
    pub fn fill_slice<'a>(&self, s: &'a mut [T]) -> &'a mut [T]
    where
        T: Clone,
    {
        let total_fill = std::cmp::min(s.len(), self.size);
        for (ix, f) in self.distances[..=self.end()]
            .iter()
            .flat_map(|v| v.iter())
            .take(total_fill)
            .enumerate()
        {
            s[ix] = f.clone();
        }
        &mut s[0..total_fill]
    }

    /// Gets the worst distance in the queue currently.
    ///
    /// This is initialized to max (which is the worst possible distance) until `cap` elements have been inserted.
    pub fn worst(&self) -> u32 {
        self.worst
    }

    /// Returns true if the cap has been reached.
    pub fn at_cap(&self) -> bool {
        self.size == self.cap
    }

    /// Iterate over the entire queue in best-to-worse order.
    pub fn iter(&mut self) -> impl Iterator<Item = (u32, &T)> {
        self.distances[..=self.end()]
            .iter()
            .enumerate()
            .flat_map(|(distance, v)| v.iter().map(move |item| (distance as u32, item)))
    }
    /// Iterate over the entire queue in best-to-worse order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u32, &mut T)> {
        let end = self.end();
        self.distances[..=end]
            .iter_mut()
            .enumerate()
            .flat_map(|(distance, v)| v.iter_mut().map(move |item| (distance as u32, item)))
    }

    /// Add a feature to the search with the precondition we are already at the cap.
    ///
    /// Warning: This function cannot cause undefined behavior, but it can be used incorrectly.
    /// This should only be called after `at_cap()` can been called and returns true.
    /// This shouldn't be used unless you profile and actually find that the branch predictor is having
    /// issues with the if statement in `push()`.
    pub unsafe fn push_at_cap(&mut self, distance: u32, item: T) -> bool {
        // We stop searching once we have enough features under the search distance,
        // so if this is true it will always get added to the FeatureHeap.
        if distance < self.worst {
            self.distances[distance as usize].push(item);
            self.remove_worst();
            true
        } else {
            false
        }
    }

    /// Gets the smallest known inclusive end of the datastructure.
    fn end(&self) -> usize {
        if self.at_cap() {
            self.worst as usize
        } else {
            W::to_usize() - 1
        }
    }

    /// Updates the worst when it has been set.
    fn update_worst(&mut self) {
        // If there is nothing left, it gets reset to max.
        self.worst = self.distances[0..=self.worst as usize]
            .iter()
            .rev()
            .position(|v| !v.is_empty())
            .map(|n| self.worst - n as u32)
            .unwrap_or(W::to_u32() - 1);
    }

    /// Remove the worst item and update the worst distance.
    fn remove_worst(&mut self) {
        self.distances[self.worst as usize].pop();
        self.update_worst();
    }
}

impl<W, T> fmt::Debug for FixedHammingHeap<W, T>
where
    W: ArrayLength<Vec<T>>,
    T: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.distances[..].fmt(formatter)
    }
}

impl<W, T> Default for FixedHammingHeap<W, T>
where
    W: ArrayLength<Vec<T>>,
{
    fn default() -> Self {
        Self {
            cap: 0,
            size: 0,
            worst: W::to_u32() - 1,
            distances: std::iter::repeat_with(|| vec![]).collect(),
        }
    }
}

#[cfg(test)]
#[test]
fn test_fixed_heap() {
    let mut candidates: FixedHammingHeap<generic_array::typenum::U129, u32> =
        FixedHammingHeap::default();
    candidates.set_capacity(3);
    assert!(candidates.push(5, 0));
    assert!(candidates.push(4, 1));
    assert!(candidates.push(3, 2));
    assert!(!candidates.push(6, 3));
    assert!(!candidates.push(7, 4));
    assert!(candidates.push(2, 5));
    assert!(candidates.push(3, 6));
    assert!(!candidates.push(10, 7));
    assert!(!candidates.push(6, 8));
    assert!(!candidates.push(4, 9));
    assert!(candidates.push(1, 10));
    assert!(candidates.push(2, 11));
    let mut arr = [0; 3];
    candidates.fill_slice(&mut arr);
    arr[1..3].sort_unstable();
    assert_eq!(arr, [10, 5, 11]);
}
