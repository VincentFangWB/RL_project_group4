import numpy as np
from src import metrics as M


def test_metrics_basic():
    r = np.array([0.01, -0.005, 0.002, 0.0, 0.003])
    assert M.cumulative_returns(r).iloc[-1] > 0
    assert M.annualized_vol(r) >= 0
    assert -1.0 <= M.max_drawdown(r) <= 0.0