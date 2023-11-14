use std::ops::RangeBounds;

pub trait ArrayLikeMut<T>: ArrayLike<T> + AsMut<[T]> {
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T>
    where
        T: 'a,
    {
        self.as_mut().iter_mut()
    }
}

pub trait ArrayLike<T>: AsRef<[T]> + Extend<T> {
    fn len(&self) -> usize;
    fn insert(&mut self, index: usize, element: T);
    fn drain(&mut self, range: impl RangeBounds<usize>) -> impl Iterator<Item = T> + '_;

    #[inline]
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        self.as_ref().iter()
    }

    #[inline]
    fn remove(&mut self, index: usize) -> T {
        let mut drain = self.drain(index..index + 1);
        let value = drain.next().unwrap();
        drop(drain);
        value
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(self.remove(self.len() - 1))
        }
    }

    #[inline]
    fn push(&mut self, element: T) {
        self.insert(self.len(), element);
    }
}

impl<T> ArrayLike<T> for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }

    fn insert(&mut self, index: usize, element: T) {
        self.insert(index, element);
    }

    fn drain(&mut self, range: impl RangeBounds<usize>) -> impl Iterator<Item = T> + '_ {
        self.drain(range)
    }
}

impl<T> ArrayLikeMut<T> for Vec<T> {}
