use core::panic;
use std::{
    borrow::Borrow,
    fmt::{Debug, Display},
};

use crate::{
    array::ArrayLike,
    track_size::{TrackSize, WithSize},
};

use super::iter::BTreeIter;

#[derive(Debug, Clone)]
/// The BTree data structure. It wraps a BTreeNode which contains most of the method's implementation.
/// It will shorten or merge the top node when needed in order to maintain the BTree property.
pub struct BTree<T> {
    bucket_size: usize,
    root: Box<BTreeNode<T>>,
}

#[derive(PartialEq, Debug, Clone)]
/// A node inside the BTree. The node will be a leaves if it does not have any children
/// When a node is NOT a leaf, `data.size() == children.size() - 1`.
pub struct BTreeNode<T> {
    data: Vec<T>,
    children: TrackSize<Vec<Box<BTreeNode<T>>>, Box<BTreeNode<T>>>,
}

impl<T> WithSize for Box<BTreeNode<T>> {
    fn size(&self) -> usize {
        self.data.len() + self.children.size()
    }
}

type InsertResult<T> = Result<Option<(T, Box<BTreeNode<T>>)>, ()>;

impl<T: Display + Debug + Ord> Display for BTreeNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("\n")?;
        self.print_impl(0, f)
    }
}

impl<T: Display + Debug + Ord> BTreeNode<T> {
    fn print_impl(&self, depth: usize, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.size() == 0 {
            return f.write_fmt(format_args!("{}<e>\n", " ".repeat(depth * 2)));
        }

        if self.is_leaf() {
            f.write_fmt(format_args!(
                "{}{}\n",
                " ".repeat(depth * 2),
                self.data
                    .iter()
                    .map(|v| format!("{v}"))
                    .collect::<Vec<_>>()
                    .join(",")
            ))?;
        } else {
            for i in 0..self.children.as_ref().len() {
                self.children.as_ref()[i].print_impl(depth + 1, f)?;
                if i < self.data.len() {
                    f.write_fmt(format_args!("{}{}\n", " ".repeat(depth * 2), self.data[i]))?;
                }
            }
        }
        Ok(())
    }
}

impl<T> BTreeNode<T> {
    pub fn size(&self) -> usize {
        self.data.len() + self.children.size()
    }
}

impl<T> BTreeNode<T>
where
    T: Ord + Debug,
{
    #[inline]
    pub fn data(&self) -> &[T] {
        self.data.as_ref()
    }

    #[inline]
    pub fn children(&self) -> &[Box<BTreeNode<T>>] {
        self.children.as_ref()
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        let result = self.children.as_ref().is_empty();
        return result;
    }

    #[inline]
    fn boxed(bucket_size: usize) -> Box<Self> {
        Box::new(Self::new(bucket_size))
    }

    #[inline]
    fn new(bucket_size: usize) -> Self {
        debug_assert!(bucket_size % 2 == 0);
        Self {
            data: Vec::with_capacity(bucket_size),
            children: TrackSize::from_inner(Vec::with_capacity(bucket_size + 1)),
        }
    }

    #[inline]
    fn contains<I>(&self, value: &I) -> bool
    where
        T: Borrow<I>,
        I: Ord,
    {
        match self.data.binary_search_by(|v| v.borrow().cmp(value)) {
            Ok(_) => true,
            Err(idx) => {
                if !self.is_leaf() {
                    self.children.as_ref()[idx].contains(value)
                } else {
                    false
                }
            }
        }
    }

    /// Insert the new element into the tree if it doesn't already exist.
    /// Split the node into two if it is already full.
    fn insert_and_split(&mut self, value: T, bucket_size: usize) -> InsertResult<T> {
        if self.is_leaf() {
            let Err(idx) = self.data.binary_search(&value) else {
                return Err(());
            };

            if self.data.len() < bucket_size {
                self.data.insert(idx, value);
                return Ok(None);
            } else {
                let mut new_node = Self::boxed(bucket_size);
                let mid = bucket_size / 2;

                if idx == mid {
                    new_node.data.extend(self.data.drain(mid..));
                    return Ok(Some((value, new_node)));
                } else if idx < mid {
                    new_node.data.extend(self.data.drain(mid..));
                    let pivot = self.data.pop().unwrap();
                    self.data.insert(idx, value);
                    return Ok(Some((pivot, new_node)));
                } else {
                    new_node.data.extend(self.data.drain(mid + 1..));
                    let pivot = self.data.pop().unwrap();
                    new_node.data.insert(idx - mid - 1, value);
                    return Ok(Some((pivot, new_node)));
                }
            }
        } else {
            let Err(idx) = self.data.binary_search(&value) else {
                return Err(());
            };

            let insert_result = self
                .children
                .modify(idx, |child| child.insert_and_split(value, bucket_size));

            match insert_result {
                Err(()) => return Err(()),
                Ok(None) => {
                    return Ok(None);
                }
                Ok(Some((new_pivot, new_child))) => {
                    self.data.insert(idx, new_pivot);
                    self.children.insert(idx + 1, new_child);

                    if self.children.as_ref().len() <= bucket_size {
                        return Ok(None);
                    } else {
                        let mut new_node = Self::boxed(bucket_size);
                        let mid = bucket_size / 2;

                        new_node.data.extend(self.data.drain(mid..));
                        new_node.children.extend(self.children.drain(mid..));
                        let pivot = self.data.pop().unwrap();

                        return Ok(Some((pivot, new_node)));
                    }
                }
            };
        }
    }

    #[inline]
    fn pop_last(&mut self, bucket_size: usize) -> Option<T> {
        if self.size() == 0 {
            return None;
        }

        Some(self.pop_at(self.size() - 1, bucket_size))
    }

    fn pop_at(&mut self, position: usize, bucket_size: usize) -> T {
        if position >= self.size() {
            panic!("Invalid position: {}", position);
        }

        if self.is_leaf() {
            return self.data.remove(position);
        }

        let mut is_recurse = false;
        let mut idx = 0;
        let mut new_position = 0;

        let size_at = |idx: usize| {
            self.children.as_ref()[idx].size() as i64 + self.data.get(idx).is_some() as i64
        };

        let position = position as i64;
        if position <= self.size() as i64 / 2 {
            let mut curr = 0;
            for i in 0..self.children.as_ref().len() {
                curr += size_at(i);
                if curr - 1 > position {
                    idx = i;
                    new_position = position - curr + size_at(i);
                    is_recurse = true;
                    break;
                } else if curr - 1 == position {
                    idx = i;
                    new_position = 0;
                    is_recurse = false;
                    break;
                }
            }
        } else {
            let mut curr = self.size() as i64 - 1;
            for i in (0..self.children.as_ref().len()).rev() {
                curr -= size_at(i);
                if curr < position {
                    idx = i;
                    new_position = position - curr - 1;
                    is_recurse = true;
                    break;
                } else if curr == position {
                    idx = i - 1;
                    new_position = 0;
                    is_recurse = false;
                    break;
                }
            }
        }

        let removed_value;
        if !is_recurse {
            let new_pivot = self
                .children
                .modify(idx, |child| child.pop_last(bucket_size).unwrap());
            removed_value = std::mem::replace(&mut self.data[idx], new_pivot);
        } else {
            removed_value = self.children.modify(idx, |child| {
                child.pop_at(new_position as usize, bucket_size)
            });
        }

        self.rebalance_child(idx, bucket_size);
        removed_value
    }

    /// Remove the value from the node. After removing, all child nodes are balanced, while the root
    /// node might not be.
    fn remove<I>(&mut self, value: &I, bucket_size: usize) -> Option<T>
    where
        I: Ord,
        T: Borrow<I>,
    {
        if self.is_leaf() {
            if let Ok(idx) = self.data.binary_search_by(|v| v.borrow().cmp(value)) {
                return Some(self.data.remove(idx));
            } else {
                return None;
            }
        }

        let removed_value;
        let removed_idx;
        match self.data.binary_search_by(|v| v.borrow().cmp(value)) {
            Ok(idx) => {
                removed_idx = idx;
                let new_pivot = self
                    .children
                    .modify(idx, |child| child.pop_last(bucket_size).unwrap());
                removed_value = std::mem::replace(&mut self.data[idx], new_pivot);
            }
            Err(idx) => {
                removed_idx = idx;
                if let Some(value) = self
                    .children
                    .modify(idx, |child| child.remove(value, bucket_size))
                {
                    removed_value = value;
                } else {
                    return None;
                }
            }
        };

        self.rebalance_child(removed_idx, bucket_size);
        Some(removed_value)
    }

    #[inline]
    fn iter(&self) -> BTreeIter<'_, T> {
        BTreeIter::new(self)
    }

    #[inline]
    fn degree(&self) -> usize {
        self.data.len().max(self.children.as_ref().len())
    }

    /// ## Returns
    /// - `Ok(depthOfTheNode)` if the node is valid
    fn verify(&self, is_root: bool, bucket_size: usize) -> anyhow::Result<usize> {
        let mut depth = vec![];

        if !is_root && (self.degree() < bucket_size / 2 || self.degree() > bucket_size) {
            anyhow::bail!(
                "degree is not correct, bucket_size: {}, degree: {}",
                bucket_size,
                self.degree()
            );
        }

        for c in self.children.as_ref() {
            depth.push(c.verify(false, bucket_size)?);
        }

        depth.dedup();
        if depth.len() > 1 {
            anyhow::bail!("depth is not consistent");
        }

        if !self
            .data
            .iter()
            .zip(self.data.iter().skip(1))
            .all(|(v1, v2)| v1 < v2)
        {
            anyhow::bail!("data is not sorted");
        }

        if !self
            .data
            .iter()
            .zip(self.children.as_ref().iter())
            .all(|(v, child)| child.data.last().unwrap() < v)
        {
            anyhow::bail!("data is not sorted");
        }

        if !self.children.as_ref().is_empty() && self.children.as_ref().len() != self.data.len() + 1
        {
            anyhow::bail!("children size is not correct");
        }

        if self.children.as_ref().is_empty() {
            Ok(1)
        } else {
            Ok(depth[0] + 1)
        }
    }

    fn rebalance_child(&mut self, idx: usize, bucket_size: usize) {
        // No need to rebalance or merge
        if self.children.as_ref()[idx].degree() >= bucket_size / 2 {
            return;
        }

        if self.children.as_ref().len() == 1 {
            return;
        }

        if self.children.as_ref()[idx].is_leaf() {
            let pivot = if idx < self.data.len() {
                self.data.remove(idx)
            } else {
                self.data.pop().unwrap()
            };

            let mut child = self.children.remove(idx);

            let insert_result = self.insert_and_split(pivot, bucket_size).unwrap();
            debug_assert!(insert_result.is_none());

            for v in child.data.drain(..) {
                let insert_result = self.insert_and_split(v, bucket_size).unwrap();
                debug_assert!(insert_result.is_none());
            }

            return;
        }

        debug_assert!(self.children.as_ref()[idx].degree() > 0);

        // rebalance between child `idx`` and `idx + 1`
        let mut idx = idx;
        if idx == self.children.as_ref().len() - 1 {
            idx -= 1;
        }

        if self.children.as_ref()[idx].degree() + self.children.as_ref()[idx + 1].degree() + 1
            <= bucket_size
        {
            // merge
            let pivot = self.data.remove(idx);
            let mut extra_node = self.children.remove(idx + 1);
            self.children.modify(idx, |child| {
                child.data.push(pivot);
                child.data.extend(extra_node.data.drain(..));
                child.children.extend(extra_node.children.drain(..));
            });

            if self.data.len() == 0 {
                let child1 = self.children.remove(idx);
                *self = *child1;
            }
        } else {
            if self.children.as_ref()[idx].degree() < bucket_size / 2 {
                // rebalance right to left
                let new_pivot_idx = bucket_size / 2 - self.children.as_ref()[idx].degree() - 1;
                self.children.modify(idx + 1, |child| {
                    std::mem::swap(&mut self.data[idx], &mut child.data[new_pivot_idx])
                });

                let mut child1 = self.children.replace(idx, Self::boxed(0));
                let mut child2 = self.children.replace(idx + 1, Self::boxed(0));

                child1.data.extend(child2.data.drain(..=new_pivot_idx));
                if !child2.is_leaf() {
                    child1
                        .children
                        .extend(child2.children.drain(..=new_pivot_idx));
                }

                self.children.replace(idx, child1);
                self.children.replace(idx + 1, child2);

                debug_assert!(self.children.as_ref()[idx].degree() == bucket_size / 2);
                debug_assert!(self.children.as_ref()[idx + 1].degree() >= bucket_size / 2);
            } else {
                // balance left to right
                let new_pivot_idx = bucket_size / 2 - 1;
                self.children.modify(idx, |child| {
                    std::mem::swap(&mut self.data[idx], &mut child.data[new_pivot_idx])
                });

                let mut child1 = self.children.replace(idx, Self::boxed(0));
                let child2 = self.children.replace(idx + 1, Self::boxed(0));
                let mut child2_data = child2.data;
                let mut child2_children = child2.children;

                self.children.modify(idx + 1, |child| {
                    child.data.extend(child1.data.drain(new_pivot_idx + 1..));
                    child.data.extend(child1.data.drain(new_pivot_idx..));
                    child.data.extend(child2_data.drain(..));
                });

                debug_assert!(!child1.is_leaf());
                self.children.modify(idx + 1, |child| {
                    child
                        .children
                        .extend(child1.children.drain(new_pivot_idx + 1..))
                });
                self.children.modify(idx + 1, |child| {
                    child.children.extend(child2_children.drain(..))
                });

                self.children.replace(idx, child1);
                debug_assert!(
                    self.children.as_ref()[idx].degree() >= bucket_size / 2,
                    "Left tree: {:?}",
                    self.children.as_ref()[idx]
                );
                debug_assert!(
                    self.children.as_ref()[idx + 1].degree() >= bucket_size / 2,
                    "Right tree: {:?}",
                    self.children.as_ref()[idx + 1]
                );
            }
        }
    }
}

impl<T: Display + Debug + Ord> Display for BTree<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <BTreeNode<T> as Display>::fmt(&self.root, f)
    }
}

impl<T: Ord + Debug + Display> BTree<T> {
    pub fn new(bucket_size: usize) -> Self {
        assert!(bucket_size >= 2, "Bucket size must be larger than 1");
        assert!(bucket_size % 2 == 0, "Bucket size must be multiple of 2");
        Self {
            root: BTreeNode::boxed(bucket_size),
            bucket_size,
        }
    }

    fn shorten_root(&mut self) {
        if self.root.data.is_empty() && !self.root.children.as_ref().is_empty() {
            debug_assert_eq!(self.root.children.as_ref().len(), 1);
            self.root = Box::new(*self.root.children.remove(0));
        }
    }

    #[inline]
    pub fn contains<I>(&self, value: &I) -> bool
    where
        I: Ord,
        T: Borrow<I>,
    {
        self.root.contains(value)
    }

    pub fn insert(&mut self, value: T) {
        if let Ok(Some((pivot, new_node))) = self.root.insert_and_split(value, self.bucket_size) {
            let old_root = std::mem::replace(&mut self.root, BTreeNode::boxed(self.bucket_size));

            let new_root = &mut self.root;
            new_root.data.push(pivot);
            new_root.children.push(old_root);
            new_root.children.push(new_node);

            debug_assert!(
                self.root.verify(true, self.bucket_size).is_ok(),
                "Invalid tree: {}",
                self
            );
            debug_assert!(!self.root.is_leaf());
        }
    }

    pub fn remove<I>(&mut self, value: &I) -> Option<T>
    where
        T: Borrow<I>,
        I: Ord,
    {
        let removed = self.root.remove(value, self.bucket_size);
        self.shorten_root();
        debug_assert!(
            self.root.verify(true, self.bucket_size).is_ok(),
            "Invalid tree: {}",
            self
        );
        removed
    }

    pub fn pop_last(&mut self) -> Option<T> {
        let removed = self.root.pop_last(self.bucket_size);
        self.shorten_root();
        debug_assert!(
            self.root.verify(true, self.bucket_size).is_ok(),
            "Invalid tree: {}",
            self
        );
        removed
    }

    pub fn len(&self) -> usize {
        self.root.size()
    }

    pub fn iter(&self) -> BTreeIter<'_, T> {
        self.root.iter()
    }
}

#[cfg(test)]
mod test {
    use std::collections::BTreeSet;

    extern crate test;
    use super::*;
    use rand::seq::SliceRandom;
    use test::Bencher;

    #[test]
    fn test_leaf_node() {
        let mut node = BTreeNode::<i32>::new(4);
        assert!(node.is_leaf());

        assert_eq!(node.insert_and_split(1, 4), Ok(None));
        assert_eq!(node.insert_and_split(2, 4), Ok(None));
        assert_eq!(node.insert_and_split(3, 4), Ok(None));
        assert_eq!(node.insert_and_split(4, 4), Ok(None));

        let (new_pivot, new_node) = node.insert_and_split(5, 4).unwrap().unwrap();
        assert_eq!(new_pivot, 3);
        assert_eq!(node.data, vec![1, 2]);
        assert_eq!(new_node.data, vec![4, 5]);
    }

    #[test]
    fn test_tree_split_root_node() {
        let mut tree = BTree::new(4);
        tree.insert(1);
        tree.insert(2);
        tree.insert(3);
        tree.insert(4);
        tree.insert(5);

        assert_eq!(
            vec![1, 2, 3, 4, 5],
            tree.iter().copied().collect::<Vec<_>>()
        );

        assert_eq!(tree.len(), 5);
    }

    #[test]
    fn test_tree_split_internal_node() {
        let mut tree = BTree::new(4);

        const MIN: i32 = 0;
        const MAX: i32 = 64;

        for i in MIN..MAX {
            tree.insert(i);
        }

        assert_eq!(tree.len(), (MAX - MIN) as usize);

        for (v, idx) in tree.iter().zip(MIN..) {
            assert_eq!(*v, idx);
        }

        for (v, idx) in tree.iter().rev().zip((MIN..MAX).rev()) {
            assert_eq!(*v, idx);
        }

        for i in MIN..MAX {
            assert!(tree.contains(&i));
        }

        for i in MAX..2 * MAX {
            assert!(!tree.contains(&i));
        }
    }

    #[test]
    fn test_insert_randomized() {
        let mut tree = BTree::new(8);
        let mut set = BTreeSet::new();

        for _ in 0..1024 {
            let v: i32 = rand::random::<i32>().abs() % 1024;
            tree.insert(v);
            set.insert(v);
        }

        let vec1 = tree.iter().copied().collect::<Vec<_>>();
        let vec2 = set.iter().copied().collect::<Vec<_>>();
        assert_eq!(vec1, vec2);
        assert_eq!(tree.len(), set.len());
    }

    #[test]
    fn test_remove_randomized() {
        let mut elements = (0..1024).collect::<Vec<_>>();
        elements.shuffle(&mut rand::thread_rng());

        let mut tree = BTree::new(8);
        let mut set = BTreeSet::new();

        for v in &elements {
            tree.insert(*v);
            set.insert(*v);
        }

        let vec1 = tree.iter().copied().collect::<Vec<_>>();
        let vec2 = set.iter().copied().collect::<Vec<_>>();
        assert_eq!(vec1, vec2);
        assert_eq!(tree.len(), set.len());

        elements.shuffle(&mut rand::thread_rng());
        for v in &elements {
            let removed = tree.remove(v);
            assert_eq!(removed, Some(*v));
            set.remove(v);
        }

        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_remove_merge_left_to_right() {
        let mut tree = BTree::new(4);
        for i in (0..20).rev() {
            tree.insert(i);
        }

        for i in (0..20).rev() {
            tree.remove(&i);
            println!("Tree: {}", tree);
            assert_eq!(tree.len(), i);
        }
    }

    #[test]
    fn test_remove_node_not_leaf() {
        let mut tree = BTree::new(4);
        for i in 0..20 {
            tree.insert(i);
        }

        let remove_seq = [14, 8, 2, 9, 13];
        for v in remove_seq {
            println!("Tree: {}", tree);
            let removed = tree.remove(&v);
            assert_eq!(removed, Some(v));
        }
    }

    #[test]
    fn test_pop_last() {
        let mut tree = BTree::new(4);
        for i in 0..20 {
            tree.insert(i);
        }

        for i in (0..20).rev() {
            assert_eq!(tree.pop_last(), Some(i));
            assert_eq!(tree.len(), i);
        }

        assert_eq!(tree.pop_last(), None);
    }

    #[test]
    fn test_remove_node() {
        for i in 0..10 {
            let mut tree = BTree::new(4);

            for i in 0..10 {
                tree.insert(i);
            }

            let v = tree.remove(&i);
            assert_eq!(v.unwrap(), i);
            assert_eq!(tree.len(), 9);

            let mut values = (0..10).collect::<Vec<_>>();
            values.remove(i);

            assert_eq!(tree.iter().copied().collect::<Vec<_>>(), values);
        }

        let mut tree = BTree::new(4);
        for i in 0..20 {
            tree.insert(i);
        }

        for i in 0..20 {
            tree.remove(&i);
            assert_eq!(tree.len(), 19 - i);
        }

        let mut tree = BTree::new(4);
        for i in 0..20 {
            tree.insert(i);
        }

        for i in (0..20).rev() {
            tree.remove(&i);
            assert_eq!(tree.len(), i);
        }
    }

    #[test]
    fn test_pop_at() {
        let mut tree = BTree::new(4);
        let values: Vec<usize> = (0..20).collect::<Vec<_>>();

        for i in 0..20 {
            tree.insert(i);
        }

        for v in 0..20 {
            let mut tree = tree.clone().root;
            let mut values = values.clone();
            assert_eq!(tree.pop_at(v, 4), v);
            assert_eq!(tree.size(), 19);
            values.remove(v);
            assert_eq!(tree.iter().copied().collect::<Vec<_>>(), values);
        }
    }

    #[bench]
    fn bench_insertion(b: &mut Bencher) {
        b.iter(|| {
            let mut tree = BTree::new(32);
            for i in 0..8192 {
                tree.insert(i);
            }

            return test::black_box(tree);
        });
    }

    #[bench]
    fn bench_insertion_baseline(b: &mut Bencher) {
        b.iter(|| {
            let mut tree = BTreeSet::new();
            for i in 0..8192 {
                tree.insert(i);
            }

            return test::black_box(tree);
        });
    }

    #[test]
    fn test_large_bucket_size() {
        let mut tree = BTree::new(32);
        for i in 0..8192 {
            tree.insert(i);
        }
    }

    #[bench]
    fn bench_contains(b: &mut Bencher) {
        let mut tree = BTree::new(32);
        for i in 0..8192 {
            tree.insert(i);
        }

        b.iter(|| {
            for i in 0..8192 {
                test::black_box(tree.contains(&i));
            }
        });
    }

    #[bench]
    fn bench_contains_baseline(b: &mut Bencher) {
        let mut tree = BTreeSet::new();
        for i in 0..8192 {
            tree.insert(i);
        }

        b.iter(|| {
            for i in 0..8192 {
                test::black_box(tree.contains(&i));
            }
        });
    }

    #[bench]
    fn bench_remove(b: &mut Bencher) {
        let mut elements = (0..8192).collect::<Vec<_>>();
        elements.shuffle(&mut rand::thread_rng());

        let mut tree = BTree::new(32);
        for i in &elements {
            tree.insert(*i);
        }

        b.iter(|| {
            let mut tree = tree.clone();
            elements.shuffle(&mut rand::thread_rng());
            for i in &elements {
                let v = tree.remove(i).is_some();
                test::black_box(v);
            }
        });
    }

    #[bench]
    fn bench_remove_baseline(b: &mut Bencher) {
        let mut elements = (0..8192).collect::<Vec<_>>();
        elements.shuffle(&mut rand::thread_rng());

        let mut tree = BTreeSet::new();
        for i in &elements {
            tree.insert(*i);
        }

        b.iter(|| {
            let mut tree = tree.clone();
            elements.shuffle(&mut rand::thread_rng());
            for i in &elements {
                let v = tree.remove(i);
                test::black_box(v);
            }
        });
    }

    #[bench]
    fn bench_iteration(b: &mut Bencher) {
        let mut tree = BTree::new(32);
        for i in 0..8192 {
            tree.insert(i);
        }

        b.iter(|| {
            for v in tree.iter() {
                test::black_box(v);
            }
        });
    }

    #[bench]
    fn bench_iteration_baseline(b: &mut Bencher) {
        let mut tree = BTreeSet::new();
        for i in 0..8192 {
            tree.insert(i);
        }

        b.iter(|| {
            for v in tree.iter() {
                test::black_box(v);
            }
        });
    }
}
