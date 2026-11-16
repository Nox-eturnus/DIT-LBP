import numpy as np

from ditlbp.features import POPCOUNT8, dit_hist, dit_lbp, flip_count, lbp_codes, lbp_hist, ldb_hist, diff_feat


def test_xor_example_from_proposal():
    left, right = np.uint8(0b11001010), np.uint8(0b10001110)
    assert np.bitwise_xor(left, right) == 0b01000100
    assert flip_count(np.array([left]), np.array([right]))[0] == 2


def test_popcount_table():
    assert POPCOUNT8[0] == 0 and POPCOUNT8[255] == 8 and POPCOUNT8[0b1010] == 2


def test_dit_hist_shape_and_identical_stack():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64))
    stack = {t: img for t in (0, 90, 180, 270)}
    h = dit_lbp(stack)
    assert h.shape == (81,) and np.isclose(h.sum(), 1)
    assert np.isclose(h[0], 1.0)  # no light change -> every pixel in bin (0 flips, 0 flips)


def test_dit_invariant_to_monotonic_exposure():
    # a flat print seen under brighter/dimmer, gamma-changed exposure has no flips
    rng = np.random.default_rng(1)
    img = rng.random((64, 64)) + 0.1
    stack = {0: img, 180: 1.7 * img ** 0.8, 90: 0.5 * img, 270: img ** 1.2}
    assert np.isclose(dit_lbp(stack)[0], 1.0)


def test_dit_detects_shading_reversal():
    # a sinusoidal ridge lit from opposite sides: shading gradient reverses sign
    x = np.linspace(0, 8 * np.pi, 64)
    slope = np.cos(x)[None, :].repeat(64, 0)
    stack = {0: 1 + 0.5 * slope, 180: 1 - 0.5 * slope, 90: np.ones((64, 64)), 270: np.ones((64, 64))}
    h = dit_lbp(stack).reshape(9, 9)
    assert h[0, :].sum() < 0.5  # most pixels flip bits between opposite lights


def test_other_descriptor_shapes():
    rng = np.random.default_rng(2)
    img = rng.random((128, 128))
    assert lbp_hist(img).shape == (59,)
    assert ldb_hist(img).shape == ((6 + 36 + 120 + 300) * 3,)
    assert diff_feat({t: img + 1 for t in (0, 90, 180, 270)}).shape == (38,)
    assert lbp_codes(img).dtype == np.uint8


def test_lbp_codes_match_manual_3x3():
    img = np.array([[5, 9, 1], [4, 6, 7], [2, 6, 3]], float)
    # neighbours (circular): right 7, up-right 1, up 9, up-left 5, left 4, down-left 2, down 6, down-right 3
    expected = sum(b << k for k, b in enumerate([7 >= 6, 1 >= 6, 9 >= 6, 5 >= 6, 4 >= 6, 2 >= 6, 6 >= 6, 3 >= 6]))
    assert lbp_codes(img)[1, 1] == expected


def test_gated_dit_suppresses_pure_noise_flips():
    from ditlbp.features import dit_gated_maps, dit_maps, noise_sigma
    rng = np.random.default_rng(3)
    flat = {t: 1 + 0.01 * rng.standard_normal((128, 128)) for t in (0, 90, 180, 270)}
    plain = np.mean(dit_maps({t: lbp_codes(a) for t, a in flat.items()})[0])
    gated = np.mean(dit_gated_maps(flat)[0])
    assert plain > 3.0 and gated < 0.2
    assert abs(noise_sigma(flat[0]) - 0.01) < 0.002
