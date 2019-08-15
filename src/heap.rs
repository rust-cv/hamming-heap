/// This is a special heap specifically for hamming space searches.
///
/// This queue works by having n-bits + 1 vectors, one for each hamming distance. When we find that any item
/// achieves a distance of `n` at the least, we place the index of that node into the vector associated
/// with that distance. Any time we take an item off, we place all of its children into the appropriate
/// distance priorities.
///
/// We maintain the lowest weight vector at any given time in the queue. When a vector runs out,
/// we iterate until we find the next-best non-empty distance vector.
///
/// To use this you will need to call `set_distances` before use. This should be passed the maximum number of
/// distances. Please keep in mind that the maximum number of hamming distances between an `n` bit number
/// is `n + 1`. An example would be:
///
/// ```
/// assert_eq!((0u128 ^ !0).count_ones(), 128);
/// ```
///
/// So make sure you use `n + 1` as your `distances` or else you may encounter a runtime panic.
///
/// ```
/// use hamming_heap::HammingHeap;
/// let mut candidates = HammingHeap::new_distances(129);
/// candidates.push((0u128 ^ !0u128).count_ones(), ());
/// ```
#[derive(Clone, Debug)]
pub struct HammingHeap<T> {
    distances: Vec<Vec<T>>,
    best: u32,
}

impl<T> HammingHeap<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Automatically initializes self with `distances` distances.
    pub fn new_distances(distances: usize) -> Self {
        let mut s = Self::new();
        s.set_distances(distances);
        s
    }

    /// This allows the queue to be cleared so that we don't need to reallocate memory.
    pub fn clear(&mut self) {
        for v in self.distances[self.best as usize..].iter_mut() {
            v.clear();
        }
        self.best = 0;
    }

    /// Set number of distances. Also clears the heap.
    ///
    /// This does not preserve the allocated memory, so don't call this on each search.
    ///
    /// If you have a 128-bit number, keep in mind that it has `129` distances because
    /// `128` is one of the possible distances.
    pub fn set_distances(&mut self, distances: usize) {
        self.distances.clear();
        self.distances.resize_with(distances, || vec![]);
        self.best = 0;
    }

    /// This removes the nearest candidate from the queue.
    #[inline]
    pub fn pop(&mut self) -> Option<(u32, T)> {
        loop {
            if let Some(node) = self.distances[self.best as usize].pop() {
                return Some((self.best, node));
            } else if self.best == self.distances.len() as u32 - 1 {
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

impl<T> Default for HammingHeap<T> {
    fn default() -> Self {
        Self {
            distances: vec![],
            best: 0,
        }
    }
}
