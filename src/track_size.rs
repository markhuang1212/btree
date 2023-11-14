use std::marker::PhantomData;

use crate::{
    array::{ArrayLike, ArrayLikeMut},
    utils::get_range,
};

/// Any container that has a size. In particular BTreeNode implements this trait.
pub trait WithSize {
    fn size(&self) -> usize;
}

#[derive(PartialEq, Debug, Clone)]
/// A data structure that track the sum of the size of its elements.
pub struct TrackSize<Inner, T> {
    size: usize,
    data: Inner,
    _phantom: PhantomData<T>,
}

impl<Inner, T> Extend<T> for TrackSize<Inner, T>
where
    T: WithSize,
    Inner: ArrayLikeMut<T>,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let mut iter = iter.into_iter();
        while let Some(element) = iter.next() {
            self.push(element);
        }
    }
}

impl<Inner, T> AsRef<[T]> for TrackSize<Inner, T>
where
    T: WithSize,
    Inner: ArrayLikeMut<T>,
{
    fn as_ref(&self) -> &[T] {
        self.data.as_ref()
    }
}

impl<Inner, T> Default for TrackSize<Inner, T>
where
    Inner: ArrayLikeMut<T> + Default,
    T: WithSize,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Inner, T> ArrayLike<T> for TrackSize<Inner, T>
where
    T: WithSize,
    Inner: ArrayLikeMut<T>,
{
    fn len(&self) -> usize {
        self.data.len()
    }

    fn insert(&mut self, index: usize, element: T) {
        let size = element.size();
        self.size += size;
        self.data.insert(index, element);
    }

    fn drain(&mut self, range: impl std::ops::RangeBounds<usize>) -> impl Iterator<Item = T> + '_ {
        for i in get_range(&range, self.len()) {
            self.size -= self.data.as_ref()[i].size();
        }

        self.data.drain(range)
    }
}

impl<Inner, T> WithSize for TrackSize<Inner, T> {
    fn size(&self) -> usize {
        self.size
    }
}

impl<Inner, T> TrackSize<Inner, T>
where
    Inner: ArrayLikeMut<T>,
    T: WithSize,
{
    pub fn new() -> Self
    where
        Inner: Default,
    {
        Self {
            size: 0,
            data: Inner::default(),
            _phantom: PhantomData,
        }
    }

    pub fn from_inner(inner: Inner) -> Self {
        let size = inner.as_ref().iter().map(|x| x.size()).sum();
        Self {
            size,
            data: inner,
            _phantom: PhantomData,
        }
    }

    pub fn modify<R>(&mut self, idx: usize, f: impl FnOnce(&mut T) -> R) -> R {
        self.size -= self.data.as_ref()[idx].size();
        let r = f(&mut self.data.as_mut()[idx]);
        self.size += self.data.as_ref()[idx].size();
        r
    }

    pub fn replace(&mut self, idx: usize, element: T) -> T {
        self.modify(idx, |x| std::mem::replace(x, element))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    struct ObjectWithSize(usize);
    impl WithSize for ObjectWithSize {
        fn size(&self) -> usize {
            self.0
        }
    }

    #[test]
    fn test_insert() {
        let v1 = ObjectWithSize(1);
        let v2 = ObjectWithSize(2);

        let mut array = TrackSize::from_inner(Vec::new());
        array.push(v1);
        array.push(v2);

        assert_eq!(array.size(), 3);
        assert_eq!(array.as_ref().len(), 2);

        array.insert(0, ObjectWithSize(3));
        assert_eq!(array.size(), 6);
    }

    #[test]
    fn test_remove() {
        let v1 = ObjectWithSize(1);
        let v2 = ObjectWithSize(2);

        let mut array = TrackSize::from_inner(vec![v1, v2]);
        assert_eq!(array.size(), 3);

        array.remove(0);
        assert_eq!(array.size(), 2);

        array.remove(0);
        assert_eq!(array.size(), 0);
    }

    #[test]
    fn test_modify() {
        let v1 = ObjectWithSize(1);
        let v2 = ObjectWithSize(2);

        let mut array = TrackSize::from_inner(vec![v1, v2]);
        assert_eq!(array.size(), 3);

        array.modify(0, |x| x.0 = 3);
        assert_eq!(array.size(), 5);
    }
}
