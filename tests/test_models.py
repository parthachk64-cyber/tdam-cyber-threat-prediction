import numpy as np
import torch

from tdam.models.ensemble import blend, learn_weight
from tdam.models.lstm import LSTMClassifier, SeqLSTM, seed_everything


def test_padded_sequence_equals_unpadded():
    seed_everything(0)
    net = SeqLSTM(6, 8).eval()
    x = torch.randn(1, 5, 6)
    full = net(x, torch.tensor([5]))
    padded = torch.cat([torch.zeros(1, 4, 6), x], dim=1)  # 4 leading pad bins, right aligned data
    out = net(padded, torch.tensor([5]))
    assert torch.allclose(full, out, atol=1e-6)


def test_final_state_is_newest_valid_bin_and_shape():
    seed_everything(1)
    net = SeqLSTM(6, 8).eval()
    x = torch.randn(3, 10, 6)
    out = net(x, torch.tensor([10, 6, 3]))
    assert out.shape == (3,)
    # changing a PAD position must not change the logit
    y = x.clone()
    y[1, :4] = 99.0
    assert torch.allclose(net(y, torch.tensor([10, 6, 3]))[1], out[1], atol=1e-6)


def test_training_is_deterministic_given_seed():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(64, 8, 6)).astype(np.float32)
    y = (Z[:, -1, 0] > 0.3).astype(np.int8)
    ln = np.full(64, 8, dtype=np.int64)
    outs = []
    for _ in range(2):
        seed_everything(3, 1)
        m = LSTMClassifier(6, 8)
        m.fit(Z, ln, y, epochs=2, seed=3)
        outs.append(m.predict_proba(Z, ln))
    assert np.allclose(outs[0], outs[1])


def test_blend_is_convex_and_validated():
    a, b = np.array([0.2, 0.8]), np.array([0.6, 0.0])
    assert np.allclose(blend(a, b, 1.0), a) and np.allclose(blend(a, b, 0.0), b)
    assert np.allclose(blend(a, b, 0.25), 0.25 * a + 0.75 * b)
    try:
        blend(a, b, 1.2)
    except ValueError:
        return
    raise AssertionError("weight outside [0, 1] must be rejected")


def test_learn_weight_prefers_better_branch_and_handles_single_class():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.2).astype(int)
    good = y + rng.normal(0, 0.3, 400)
    bad = rng.random(400)
    w, _ = learn_weight(good, bad, y)
    assert w >= 0.8
    w1, detail = learn_weight(good, bad, np.zeros(400, int))
    assert w1 == 0.5 and "single-class" in detail["reason"]
