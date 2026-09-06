"""
Scalers: target standardization with train-only fitting.

Scalers must only be fit on training/dev data, never on test/OOD/external.
Exposure 20/40/60/80/100% must respect corresponding data boundaries.
"""
