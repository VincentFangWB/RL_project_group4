import numpy as np
import pandas as pd
from src.envs.single_asset_env import SingleAssetTradingEnv, EnvConfig


def test_env_shapes():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = pd.Series(np.linspace(100, 110, 100), index=idx)

    feats = pd.DataFrame(
        {
            "ret": pd.Series(np.random.normal(0, 0.001, size=100), index=idx),
            "f1": 0.0,
            "f2": 1.0,
        },
        index=idx,
    )

    cfg = EnvConfig(window=10)
    env = SingleAssetTradingEnv(prices, feats, cfg)
    obs, info = env.reset()
    assert obs.shape[0] == 10 * 3  # window * feat_dim

    for _ in range(5):
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        if terminated or truncated:
            break