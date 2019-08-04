use std::fmt;

/// This keeps the nearest `cap` items at all times.
///
/// This heap is not intended to be popped. Instead, this maintains the best `cap` items, and then when you are
/// done adding items, you may fill a slice or iterate over the results. Theoretically, this could also allow
/// popping elements in constant time, but that would incur a performance penalty for the highly specialized
/// purpose this serves. This is specifically tailored for doing hamming space nearest neighbor searches.
#[derive(Clone)]
pub struct FixedHammingHeap128<T> {
    cap: usize,
    size: usize,
    worst: u32,
    distances: [Vec<T>; 129],
}

impl<T> FixedHammingHeap128<T> {
    /// This sets the capacity of the queue to `cap`, meaning that adding items to the queue will eject the worst ones
    /// if they are better once `cap` is reached. If the capacity is lowered, this removes the worst elements to
    /// keep `size == cap`.
    pub fn set_capacity(&mut self, cap: usize) {
        assert_ne!(cap, 0);
        self.set_size(cap);
        self.cap = cap;
    }

    /// This removes elements until it reaches `size`. If `size` is lower than the current
    /// number of elements, this does nothing. If the size is lowered, this will unconditionally allow insertions
    /// until `cap` is reached.
    pub fn set_size(&mut self, size: usize) {
        if size == 0 {
            for v in &mut self.distances[..] {
                v.clear();
            }
            self.size = 0;
            self.worst = 128;
        } else if size < self.size {
            // Remove the difference between them.
            for _ in size..self.size {
                self.remove_worst();
            }
            self.size = size;
            self.worst = 128;
        }
    }

    /// Clear the queue while maintaining the allocated memory.
    pub fn clear(&mut self) {
        self.size = 0;
        self.worst = 128;
        for v in self.distances.iter_mut() {
            v.clear();
        }
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
        for (ix, f) in self
            .distances
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
    /// This is initialized to 128 (which is the worst possible distance) until `cap` elements have been inserted.
    pub fn worst(&self) -> u32 {
        self.worst
    }

    /// Returns true if the cap has been reached.
    pub fn at_cap(&self) -> bool {
        self.size == self.cap
    }

    /// Iterate over the entire queue in best-to-worse order.
    pub fn iter(&mut self) -> impl Iterator<Item = (u32, &T)> {
        self.distances[..=self.worst as usize]
            .iter()
            .enumerate()
            .flat_map(|(distance, v)| v.iter().map(move |item| (distance as u32, item)))
    }
    /// Iterate over the entire queue in best-to-worse order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u32, &mut T)> {
        self.distances[..=self.worst as usize]
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

    /// Updates the worst when it has been set.
    fn update_worst(&mut self) {
        self.worst -= self.distances[0..=self.worst as usize]
            .iter()
            .rev()
            .position(|v| !v.is_empty())
            .unwrap() as u32;
    }

    /// Remove the worst item and update the worst distance.
    fn remove_worst(&mut self) {
        self.distances[self.worst as usize].pop();
        self.update_worst();
    }
}

impl<T> fmt::Debug for FixedHammingHeap128<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.distances[..].fmt(formatter)
    }
}

impl<T> Default for FixedHammingHeap128<T> {
    fn default() -> Self {
        Self {
            cap: 0,
            size: 0,
            worst: 128,
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
        }
    }
}
