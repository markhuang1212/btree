# btree

Btree written in rust. For learning and experimenting.

### Status

This project is feature complete. However, many optimization can be done, especially for the rebalancing during removal, and some avoidable dynamic allocation.

### Benchmark

`BTree` is written in completely safe rust. It delivers compariable performance for inserting and searching, but relatively slow for removal and iteration.

### Future Plans

- [ ] Improve rebalancing logic
- [ ] Use unsafe Rust to improve critical path
