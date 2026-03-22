# TODO

## Completed maintenance pass
- [x] removed the custom `predict_single_series` helper from the public API
- [x] moved the synthetic generator into `timebaseula/synthetic.py`
- [x] updated scripts to import package utilities instead of `tests/`
- [x] unified shared training logic for `TimeBase` and `TimeBaseTrend`
- [x] standardized `scripts/generate_datasets.py` with Typer, Rich, and rotating logs
- [x] added an agent-friendly markdown digest for the TimeBase paper
- [x] aligned README and MkDocs pages with the shipped behavior
- [x] documented that the package has been vibecoded
- [x] kept the fast unit suite separate from heavier integration and benchmark suites
- [x] added coverage for the NeuralForecast single-series-after-multi-series workflow
- [x] regenerated the synthetic scenario images

## Follow-up ideas
- [ ] add benchmark-marked tests only when a stable, mockable benchmark contract is needed
- [ ] consider factoring shared logging helpers for scripts if more scripts are added
