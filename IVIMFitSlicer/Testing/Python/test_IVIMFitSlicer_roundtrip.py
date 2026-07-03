"""
Self-test for the IVIMFitSlicer extension (additive — does NOT touch runtime logic).

Proves the full pipeline (pixel extraction -> normalization -> b-value handling ->
fitting -> result packaging) recovers KNOWN IVIM parameters from a synthetic phantom,
across methods and features. Doubles as 'verified' evidence for the SoftwareX paper.

HOW TO RUN (inside 3D Slicer Python console):
    import sys; sys.path.insert(0, r"<path-to-repo>/IVIMFitSlicer/Testing/Python")
    import test_IVIMFitSlicer_roundtrip as t
    t.run()
Re-run after edits:
    import importlib; importlib.reload(t); t.run()
"""
import numpy as np
import slicer

B_VALUES = [0, 10, 20, 30, 50, 100, 200, 400, 600, 800]
TRUTH = {"f": 0.10, "D": 1.2e-3, "Ds": 15.0e-3}
S0 = 1000.0

BOUNDS = {
    "f":   {"init": 0.2,   "low": 0.0,    "high": 0.5},
    "D":   {"init": 0.001, "low": 0.0001, "high": 0.005},
    "Ds":  {"init": 0.02,  "low": 0.001,  "high": 0.1},
    "ADC": {"init": 0.001, "low": 0.0,    "high": 0.03},
}


def _biexp(b, f, D, Ds):
    b = np.asarray(b, float)
    return f * np.exp(-b * Ds) + (1.0 - f) * np.exp(-b * D)


def _make_vector_volume(name, signal_1d, Z=2, Y=6, X=6, snr=None, seed=None):
    """Homogeneous 4D (Z,Y,X,B) phantom as a vector volume (B = components).
    If snr is given, add per-voxel Rician noise (sigma = S0/snr)."""
    base = (S0 * np.asarray(signal_1d, float)).astype(np.float32)
    arr = np.empty((Z, Y, X, len(signal_1d)), dtype=np.float32)
    arr[...] = base[None, None, None, :]
    if snr is not None:
        rng = np.random.default_rng(seed)
        sigma = S0 / snr
        n1 = rng.normal(0, sigma, arr.shape)
        n2 = rng.normal(0, sigma, arr.shape)
        arr = np.sqrt((arr + n1) ** 2 + n2 ** 2).astype(np.float32)
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", name)
    slicer.util.updateVolumeFromArray(node, arr)
    return node, (Z, Y, X)


def _make_full_mask(refNode, shape, empty=False):
    Z, Y, X = shape
    mask = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TestMask")
    mask.CopyOrientation(refNode)
    fill = np.zeros if empty else np.ones
    slicer.util.updateVolumeFromArray(mask, fill((Z, Y, X), dtype=np.int16))
    return mask


def _logic():
    from IVIMFitSlicer import IVIMFitSlicerLogic
    return IVIMFitSlicerLogic()


def _run(vol, mask, method, b=None, excl=None):
    return _logic().process(vol, mask, list(b or B_VALUES), list(excl or []),
                            method, 200, False, False, BOUNDS, None)


# ---------------------------------------------------------------- recovery
def test_biexp_round_trip():
    vol, shape = _make_vector_volume("TestIVIM_biexp",
                                     _biexp(B_VALUES, TRUTH["f"], TRUTH["D"], TRUTH["Ds"]))
    res = _run(vol, _make_full_mask(vol, shape), "biexp")
    p = res["params"]
    assert abs(p["f"]  - TRUTH["f"])  < 0.03, f"f off: {p['f']}"
    assert abs(p["D"]  - TRUTH["D"])  < 2e-4, f"D off: {p['D']}"
    assert abs(p["Ds"] - TRUTH["Ds"]) < 5e-3, f"D* off: {p['Ds']}"
    assert res["r2"] > 0.99, f"R2 low: {res['r2']}"
    print(f"[PASS] biexp round-trip  f={p['f']:.3f} D={p['D']:.2e} D*={p['Ds']:.2e} R2={res['r2']:.4f}")


def test_segmented_round_trip():
    vol, shape = _make_vector_volume("TestIVIM_seg",
                                     _biexp(B_VALUES, TRUTH["f"], TRUTH["D"], TRUTH["Ds"]))
    res = _run(vol, _make_full_mask(vol, shape), "segmented")
    p = res["params"]
    assert abs(p["f"]  - TRUTH["f"])  < 0.04, f"f off: {p['f']}"
    assert abs(p["D"]  - TRUTH["D"])  < 2e-4, f"D off: {p['D']}"
    assert abs(p["Ds"] - TRUTH["Ds"]) < 6e-3, f"D* off: {p['Ds']}"
    print(f"[PASS] segmented round-trip  f={p['f']:.3f} D={p['D']:.2e} D*={p['Ds']:.2e}")


def test_adc_round_trip():
    D_true = 1.5e-3
    sig = np.exp(-np.asarray(B_VALUES, float) * D_true)
    vol, shape = _make_vector_volume("TestIVIM_adc", sig)
    res = _run(vol, _make_full_mask(vol, shape), "adc")
    assert abs(res["params"]["D"] - D_true) < 1e-4, f"ADC off: {res['params']['D']}"
    print(f"[PASS] adc round-trip    ADC={res['params']['D']:.2e} (truth {D_true:.2e})")


# ---------------------------------------------------------------- features / guards
def test_excluded_bvalues():
    vol, shape = _make_vector_volume("TestIVIM_excl",
                                     _biexp(B_VALUES, TRUTH["f"], TRUTH["D"], TRUTH["Ds"]))
    mask = _make_full_mask(vol, shape)
    res = _run(vol, mask, "biexp", excl=[800.0])
    assert 800.0 not in list(res["b"]), f"excluded b still present: {res['b']}"
    assert len(res["b"]) == len(B_VALUES) - 1
    print("[PASS] excluded b-value removed (800 dropped)")
    try:                                   # excluding all but b=0 -> <2 left -> must raise
        _run(vol, mask, "biexp", excl=list(B_VALUES[1:]))
    except ValueError:
        print("[PASS] excluding to <2 b-values raises ValueError")
        return
    raise AssertionError("excluding to <2 did not raise")


def test_empty_mask_raises():
    vol, shape = _make_vector_volume("TestIVIM_empty",
                                     _biexp(B_VALUES, TRUTH["f"], TRUTH["D"], TRUTH["Ds"]))
    mask = _make_full_mask(vol, shape, empty=True)
    try:
        _run(vol, mask, "biexp")
    except ValueError as e:
        print(f"[PASS] empty mask raises ValueError ('{e}')")
        return
    raise AssertionError("empty mask did NOT raise")


def test_bvalue_autoinsert():
    vol, shape = _make_vector_volume("TestIVIM_autoins",
                                     _biexp(B_VALUES, TRUTH["f"], TRUTH["D"], TRUTH["Ds"]))
    res = _run(vol, _make_full_mask(vol, shape), "biexp", b=B_VALUES[1:])  # drop the 0
    assert res["b"][0] == 0.0, f"b=0 not auto-inserted: {res['b'][:2]}"
    print(f"[PASS] b-value auto-insert  b[0]={res['b'][0]}")


# ---------------------------------------------------------------- robustness
def test_biexp_noise_robustness():
    sig = _biexp(B_VALUES, TRUTH["f"], TRUTH["D"], TRUTH["Ds"])
    vol, shape = _make_vector_volume("TestIVIM_noise", sig, Z=4, Y=10, X=10, snr=30, seed=42)
    res = _run(vol, _make_full_mask(vol, shape), "biexp")
    p = res["params"]
    assert abs(p["f"] - TRUTH["f"]) < 0.05, f"f off under noise: {p['f']}"
    assert abs(p["D"] - TRUTH["D"]) < 5e-4, f"D off under noise: {p['D']}"
    print(f"[PASS] biexp noise robustness (SNR30, ROI-avg)  f={p['f']:.3f} D={p['D']:.2e} D*={p['Ds']:.2e}")


# ---------------------------------------------------------------- tri-exponential
# Ground truth via the library's OWN forward model -> immune to convention mismatch.
# Compartments well separated so the (degenerate) tri-exp fit is identifiable.
TRI_TRUTH = {"f_fast": 0.20, "f_inter": 0.20, "D_slow": 1.0e-3, "D_inter": 10.0e-3, "D_fast": 80.0e-3}
TRI_BOUNDS = {
    "f_fast":      {"init": 0.2,   "low": 0.0,    "high": 0.5},
    "f_inter":     {"init": 0.3,   "low": 0.0,    "high": 0.5},
    "D_slow_tri":  {"init": 0.001, "low": 0.0001, "high": 0.003},
    "D_inter_tri": {"init": 0.01,  "low": 0.003,  "high": 0.015},
    "D_fast_tri":  {"init": 0.1,   "low": 0.015,  "high": 0.15},
}


def test_triexp_fit_quality():
    from ivimfit import triexp
    b = np.asarray(B_VALUES, float)
    # same positional order the extension uses: (b, f_fast, f_inter, D_slow, D_fast, D_inter)
    sig = triexp.triexp_model(b, TRI_TRUTH["f_fast"], TRI_TRUTH["f_inter"],
                              TRI_TRUTH["D_slow"], TRI_TRUTH["D_fast"], TRI_TRUTH["D_inter"])
    vol, shape = _make_vector_volume("TestIVIM_tri", sig)
    res = _logic().process(vol, _make_full_mask(vol, shape), list(B_VALUES), [],
                           "triexp", 200, False, False, TRI_BOUNDS, None)
    p = res["params"]
    f_fast, f_inter = p["f"], p["f2"]
    f_slow = 1.0 - f_fast - f_inter
    assert res["r2"] > 0.99, f"triexp R2 low: {res['r2']}"
    assert 0.0 <= f_fast <= 1.0 and 0.0 <= f_inter <= 1.0 and -0.05 <= f_slow <= 1.05, \
        f"invalid fractions: f_fast={f_fast} f_inter={f_inter} f_slow={f_slow}"
    assert p["D"] > 0 and p["Ds"] > 0 and p["Ds2"] > 0, "non-positive diffusivity"
    print(f"[PASS] triexp fit quality  R2={res['r2']:.4f}  f_fast={f_fast:.2f} f_inter={f_inter:.2f} "
          f"D_slow={p['D']:.2e} D_inter={p['Ds2']:.2e} D_fast={p['Ds']:.2e}")


# ---------------------------------------------------------------- bayesian (opt-in: pulls in PyMC, slow)
def test_bayesian_fast_smoke():
    sig = _biexp(B_VALUES, TRUTH["f"], TRUTH["D"], TRUTH["Ds"])
    vol, shape = _make_vector_volume("TestIVIM_bayes", sig)
    res = _logic().process(vol, _make_full_mask(vol, shape), list(B_VALUES), [],
                           "bayesian_fast", 200, False, False, BOUNDS, None)
    p = res["params"]
    assert "f" in p, "bayesian returned no params (PyMC/ivimfit.bayesian unavailable?)"
    assert abs(p["f"] - TRUTH["f"]) < 0.06, f"bayes f off: {p['f']}"
    assert abs(p["D"] - TRUTH["D"]) < 3e-4, f"bayes D off: {p['D']}"
    print(f"[PASS] bayesian_fast smoke  f={p['f']:.3f} D={p['D']:.2e} D*={p['Ds']:.2e} "
          f"r_hat={p.get('r_hat', 'n/a')}")


def run(include_bayesian=False):
    slicer.mrmlScene.Clear(0)
    tests = [test_biexp_round_trip, test_segmented_round_trip, test_adc_round_trip,
             test_excluded_bvalues, test_empty_mask_raises, test_bvalue_autoinsert,
             test_biexp_noise_robustness, test_triexp_fit_quality]
    if include_bayesian:
        tests.append(test_bayesian_fast_smoke)   # needs PyMC; first run installs it (slow)
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
        finally:
            slicer.mrmlScene.Clear(0)
    print(f"\n{passed}/{len(tests)} passed")


if __name__ == "__main__":
    run()
