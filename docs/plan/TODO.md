# TODO

## Completed simplification pass
- [x] reduced the public API to the four model classes
- [x] removed the package-level synthetic helper
- [x] removed the package-level recommendation helpers
- [x] rebuilt `AutoTimeBase` and `AutoTimeBaseTrend` on top of Nixtla's native auto pattern
- [x] deleted compatibility script aliases
- [x] removed synthetic benchmark and plotting tooling
- [x] simplified benchmark outputs to CSV and markdown
- [x] updated unit and integration tests around the new contracts

## Follow-up ideas
- [ ] keep the auto search spaces compact and predictable
- [ ] revisit dependency weight if a lighter auto backend becomes practical later
