use std::{fmt::Debug, ops::AddAssign};

use super::btree::BTreeNode;

struct BTreeIterStack<'a, T> {
    node: &'a BTreeNode<T>,
    curr_idx: usize,
    /// ignored when node is a leaf
    is_children_done: bool,
}

pub enum IterDirection {
    Forward,
    Backward,
}

pub struct BTreeIter<'a, T> {
    direction: IterDirection,
    stack: Vec<BTreeIterStack<'a, T>>,
}

impl<'a, T: Ord + Debug> BTreeIter<'a, T> {
    pub fn new(root: &'a BTreeNode<T>) -> Self {
        let mut stack = Vec::new();
        stack.push(BTreeIterStack {
            node: root,
            curr_idx: 0,
            is_children_done: false,
        });

        Self {
            direction: IterDirection::Forward,
            stack,
        }
    }

    pub fn rev(mut self) -> Self {
        self.direction = match self.direction {
            IterDirection::Forward => IterDirection::Backward,
            IterDirection::Backward => IterDirection::Forward,
        };

        for frame in &mut self.stack {
            frame.curr_idx = match self.direction {
                IterDirection::Forward => 0,
                IterDirection::Backward => {
                    frame.node.data().len().max(frame.node.children().len()) - 1
                }
            };
        }

        self
    }
}

impl<'a, T: Ord + Debug> Iterator for BTreeIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let increment_fn = match self.direction {
            IterDirection::Forward => |x: &mut usize| x.add_assign(1),
            IterDirection::Backward => |x: &mut usize| *x = x.wrapping_sub(1),
        };

        let make_stack_fn = |node: &'a BTreeNode<T>| BTreeIterStack {
            node,
            curr_idx: match self.direction {
                IterDirection::Forward => 0,
                IterDirection::Backward => node.data().len().max(node.children().len()) - 1,
            },
            is_children_done: false,
        };

        let frame = self.stack.last_mut()?;
        if frame.node.is_leaf() {
            let value = frame.node.data().get(frame.curr_idx);
            if let Some(value) = value {
                increment_fn(&mut frame.curr_idx);
                return Some(value);
            } else {
                self.stack.pop();
                return self.next();
            }
        } else {
            if !frame.is_children_done {
                frame.is_children_done = true;
                let child_node = frame.node.children()[frame.curr_idx].as_ref();
                self.stack.push(make_stack_fn(child_node));
                return self.next();
            } else {
                let value = frame.node.data().get(frame.curr_idx);
                if let Some(value) = value {
                    frame.is_children_done = false;
                    increment_fn(&mut frame.curr_idx);
                    return Some(value);
                } else {
                    self.stack.pop();
                    return self.next();
                }
            }
        }
    }
}
