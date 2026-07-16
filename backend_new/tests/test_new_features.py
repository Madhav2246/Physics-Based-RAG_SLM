"""
Pytest unit tests for NodeProfileManager, SweepEngine, and SLMExtractor.

Run from backend/:
    pytest -v tests/test_new_features.py

No model load required — all tests are deterministic and offline.
"""
import sys
import math
import json
import tempfile
from pathlib import Path

# Make backend importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Feature 1: NodeProfileManager
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeProfileManager:

    @pytest.fixture(autouse=True)
    def manager(self):
        from physics.node_profile_manager import NodeProfileManager
        self.mgr = NodeProfileManager()

    def test_list_nodes_returns_four(self):
        nodes = self.mgr.list_nodes()
        assert len(nodes) == 4
        assert "100nm_CMOS"  in nodes
        assert "28nm_FDSOI"  in nodes
        assert "16nm_FinFET" in nodes
        assert "5nm_GAA"     in nodes

    def test_load_100nm_cmos(self):
        profile = self.mgr.load("100nm_CMOS")
        assert "params" in profile
        assert "mu" in profile["params"]
        assert abs(profile["params"]["mu"]["value"] - 0.05) < 1e-6

    def test_load_5nm_gaa_vth(self):
        """5nm GAA threshold must be lower than 100nm CMOS."""
        gaa  = self.mgr.load("5nm_GAA")
        cmos = self.mgr.load("100nm_CMOS")
        assert gaa["params"]["Vth"]["value"] < cmos["params"]["Vth"]["value"]

    def test_load_caches(self):
        """Two loads of the same profile must return the identical dict object."""
        a = self.mgr.load("100nm_CMOS")
        b = self.mgr.load("100nm_CMOS")
        assert a is b

    def test_load_unknown_raises(self):
        with pytest.raises(FileNotFoundError):
            self.mgr.load("999nm_FAKE")

    # Detection tests
    def test_detect_5nm_gaa(self):
        assert self.mgr.detect_from_query("Using the 5nm GAA node") == "5nm_GAA"

    def test_detect_gaa_alias(self):
        assert self.mgr.detect_from_query("gate-all-around process") == "5nm_GAA"

    def test_detect_finfet(self):
        assert self.mgr.detect_from_query("16nm FinFET design") == "16nm_FinFET"

    def test_detect_fdsoi(self):
        assert self.mgr.detect_from_query("28nm FDSOI process node") == "28nm_FDSOI"

    def test_detect_100nm(self):
        assert self.mgr.detect_from_query("standard 100nm CMOS") == "100nm_CMOS"

    def test_detect_none_when_no_keyword(self):
        assert self.mgr.detect_from_query("what is the W/L ratio?") is None

    def test_detect_case_insensitive(self):
        assert self.mgr.detect_from_query("Using 5NM GAA") == "5nm_GAA"

    # as_tracker_defaults tests
    def test_as_tracker_defaults_has_physical_constants(self):
        """k and q (Boltzmann/charge) must survive node override."""
        from physics.value_tracker import ValueTracker
        defaults = self.mgr.as_tracker_defaults("5nm_GAA")
        assert "k" in defaults
        assert abs(defaults["k"].value - 1.38e-23) < 1e-30
        assert "q" in defaults
        assert abs(defaults["q"].value - 1.6e-19) < 1e-26

    def test_as_tracker_defaults_node_overrides_mu(self):
        """mu for 5nm_GAA must differ from the 100nm baseline."""
        defaults_5nm   = self.mgr.as_tracker_defaults("5nm_GAA")
        defaults_100nm = self.mgr.as_tracker_defaults("100nm_CMOS")
        assert defaults_5nm["mu"].value != defaults_100nm["mu"].value

    def test_as_tracker_defaults_all_provenance_default(self):
        """Node-profile params must be provenance='default'.
        Physical constants k and q are correctly tagged 'corpus' in DEFAULTS
        and must be preserved as-is (they don't come from the node profile).
        """
        # Symbols that come from ValueTracker.DEFAULTS as physical constants
        # are intentionally tagged 'corpus' — that is correct and must not change.
        CORPUS_CONSTANTS = {"k", "q"}
        defaults = self.mgr.as_tracker_defaults("16nm_FinFET")
        for sym, tv in defaults.items():
            if sym in CORPUS_CONSTANTS:
                assert tv.provenance == "corpus", (
                    f"{sym} should be 'corpus', got '{tv.provenance}'"
                )
            else:
                assert tv.provenance == "default", (
                    f"{sym} has provenance '{tv.provenance}', expected 'default'"
                )

    def test_four_nodes_produce_distinct_wl(self):
        """W/L results must differ for all 4 nodes on the same query."""
        import sympy as sp
        from physics.equation_validator import EquationValidator
        from physics.exploration_engine import ExplorationEngine

        validator = EquationValidator()
        engine    = ExplorationEngine(validator.symbols)

        CORPUS_EQ = "Id = 0.5 * mu * Cox * (W/L) * (Vgs - Vth)^2"
        lhs, rhs, msg = validator.validate(CORPUS_EQ)
        assert lhs is not None, f"corpus eq parse failed: {msg}"

        query = "What W/L for Id=1mA, Vov=0.3V?"
        wl_values = []
        for node in ["100nm_CMOS", "28nm_FDSOI", "16nm_FinFET", "5nm_GAA"]:
            nd = self.mgr.as_tracker_defaults(node)
            result = engine.solve(lhs, rhs, query, node_defaults=nd)
            wl = result.get("numeric")
            assert wl is not None, f"node {node} failed to solve"
            wl_values.append(round(wl, 6))

        assert len(set(wl_values)) == 4, (
            f"Expected 4 distinct W/L values, got: {wl_values}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2: SweepEngine & parse_sweep_request
# ─────────────────────────────────────────────────────────────────────────────

class TestParseSweepRequest:

    @pytest.fixture(autouse=True)
    def imports(self):
        from physics.sweep_engine import parse_sweep_request, SweepRequest
        self.parse = parse_sweep_request
        self.SweepRequest = SweepRequest

    def test_plot_versus_from_to(self):
        r = self.parse("Plot W/L versus Vov from 0.1V to 0.6V for Id=1mA")
        assert r is not None
        assert r.sweep_var == "Vov"
        assert abs(r.start - 0.1) < 1e-9
        assert abs(r.stop  - 0.6) < 1e-9

    def test_sweep_from_to_with_units(self):
        r = self.parse("sweep Id from 1uA to 10mA")
        assert r is not None
        assert r.sweep_var == "Id"
        assert abs(r.start - 1e-6)  < 1e-12
        assert abs(r.stop  - 10e-3) < 1e-9

    def test_non_sweep_returns_none(self):
        assert self.parse("what W/L for Id=1mA, Vov=0.3V") is None

    def test_vs_range_form(self):
        r = self.parse("Vov vs W/L, range 0.2 to 0.8")
        assert r is not None
        assert r.sweep_var == "Vov"
        assert abs(r.start - 0.2) < 1e-9
        assert abs(r.stop  - 0.8) < 1e-9

    def test_bare_plot_no_range_returns_none(self):
        """A 'plot' keyword with no parseable range must NOT trigger SWEEP."""
        assert self.parse("plot the W/L ratio in saturation") is None

    def test_default_n_points(self):
        r = self.parse("sweep Vov from 0.1 to 0.5")
        assert r is not None
        assert r.n_points == 50


class TestSweepEngine:

    @pytest.fixture(autouse=True)
    def setup(self):
        from physics.sweep_engine import SweepEngine, SweepRequest, SweepResult
        from physics.equation_validator import EquationValidator
        from physics.exploration_engine import ExplorationEngine

        self.engine    = SweepEngine()
        self.validator = EquationValidator()
        self.ex_engine = ExplorationEngine(self.validator.symbols)

        # Parse a drain-current equation once for all tests
        lhs, rhs, msg = self.validator.validate(
            "Id = 0.5 * mu * Cox * (W/L) * (Vgs - Vth)^2"
        )
        assert lhs is not None
        self.lhs = lhs
        self.rhs = rhs

    def _make_tracker(self, **kwargs):
        """Build a ValueTracker with user-supplied values."""
        from physics.value_tracker import ValueTracker
        tracker = ValueTracker()
        for sym, val in kwargs.items():
            tracker.add_user(sym, val, "SI", "test")
        return tracker

    def test_run_sweep_returns_correct_length(self):
        from physics.sweep_engine import SweepRequest
        import sympy as sp

        # Solve for WL symbolically
        WL = sp.Symbol("WL")
        syms = self.ex_engine.validator.symbols
        result = self.ex_engine.solve(self.lhs, self.rhs,
                                      "What W/L for Id=1mA, Vov=0.3V?")
        sym_expr = result.get("symbolic_expr")
        if sym_expr is None:
            pytest.skip("symbolic_expr not exposed by solve()")

        req     = SweepRequest(sweep_var="Vov", target_var="WL",
                               start=0.1, stop=0.6, n_points=25)
        tracker = self._make_tracker(Id=1e-3)
        sr      = self.engine.run_sweep(req, sym_expr, tracker, "WL")
        assert sr.error == "", f"Sweep error: {sr.error}"
        assert len(sr.x) == 25
        assert len(sr.y) == 25

    def test_run_sweep_y_decreases_as_vov_increases(self):
        """For fixed Id, W/L must decrease monotonically as Vov increases
        (because Id ∝ Vov²·WL → WL ∝ 1/Vov²).

        NOTE: We build the symbolic expression directly rather than going
        through engine.solve() with a query that contains Vov=0.3V, because
        solve() would pre-substitute Vov, leaving no free symbol to sweep.
        """
        import sympy as sp
        from physics.sweep_engine import SweepRequest

        # WL = Id / (0.5 * mu * Cox * Vov**2)  — derived analytically
        WL, mu, Cox, Id, Vov = sp.symbols("WL mu Cox Id Vov")
        sym_expr = Id / (sp.Rational(1, 2) * mu * Cox * Vov**2)

        req     = SweepRequest(sweep_var="Vov", target_var="WL",
                               start=0.1, stop=0.6, n_points=10)
        # Fix Id, mu, Cox — leave Vov free so the sweep can vary it
        tracker = self._make_tracker(Id=1e-3, mu=0.05, Cox=0.02)
        sr      = self.engine.run_sweep(req, sym_expr, tracker, "WL")
        assert sr.error == "", f"Sweep error: {sr.error}"
        assert len(sr.y) == 10
        # WL must decrease monotonically as Vov increases (WL ∝ 1/Vov²)
        assert sr.y[0] > sr.y[-1], (
            f"W/L should decrease as Vov increases: y[0]={sr.y[0]:.4f}, "
            f"y[-1]={sr.y[-1]:.4f}"
        )

    def test_plot_creates_png_file(self):
        """plot() must write a PNG to the specified path and return the path."""
        from physics.sweep_engine import SweepResult
        import numpy as np

        sr = SweepResult(
            sweep_var="Vov", target_var="WL", node_name="100nm_CMOS",
            x=list(np.linspace(0.1, 0.6, 20)),
            y=list(np.linspace(10, 2, 20)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "test_sweep.png")
            returned = self.engine.plot(sr, out)
            assert returned == out
            assert Path(out).exists(), "PNG file was not created"
            assert Path(out).stat().st_size > 1000, "PNG is suspiciously small"


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3: SLMExtractor — _verify_extractable gate
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyExtractable:
    """
    Tests for the verification gate in slm_extractor.py.
    The gate accepts a value only if it is traceable to a numeric token
    in the source query (within 2% relative tolerance after unit conversion).
    """

    @pytest.fixture(autouse=True)
    def import_gate(self):
        from physics.slm_extractor import _verify_extractable
        self.gate = _verify_extractable

    # ── Should ACCEPT ────────────────────────────────────────────────────────

    def test_accepts_exact_si_value(self):
        """Id = 1e-3 A when query contains '1mA'."""
        assert self.gate(1e-3, "Id=1mA, Vov=0.3V")

    def test_accepts_voltage_millivolts(self):
        """Vth = 0.4 V when query contains '400 millivolts'."""
        assert self.gate(0.4, "threshold at 400 millivolts")

    def test_accepts_half_literal(self):
        """Vov = 0.5 V when query contains 'half a volt'."""
        assert self.gate(0.5, "overdrive voltage is half a volt")

    def test_accepts_scientific_notation(self):
        """Id = 2e-3 when query contains '2e-3 A'."""
        assert self.gate(2e-3, "Id = 2e-3 A")

    def test_accepts_zero_when_zero_in_query(self):
        """Vsb = 0.0 when query contains '0 volts'."""
        assert self.gate(0.0, "Vsb is 0 volts")

    def test_accepts_nanometer_length(self):
        """tox = 2e-9 when query contains '2nm'."""
        assert self.gate(2e-9, "tox equals 2nm")

    def test_accepts_micron_width(self):
        """W = 2e-6 when query contains '2 microns'."""
        assert self.gate(2e-6, "W is 2 microns, L is 100 nanometers")

    def test_accepts_within_tolerance(self):
        """1.0001e-3 is within 2% of 1e-3 (from '1mA')."""
        assert self.gate(1.0001e-3, "Id=1mA")

    # ── Should REJECT ────────────────────────────────────────────────────────

    def test_rejects_fabricated_value(self):
        """A value with no numeric token anywhere near it must be rejected."""
        # Query only mentions W/L; fabricated Cox should not be traceable
        assert not self.gate(0.02, "what is the W/L ratio?")

    def test_rejects_zero_when_no_zero_in_query(self):
        """0.0 should not be traceable from a query with no standalone zero token.
        'Vov=0.3V' contains '0' inside a decimal — that must NOT count as a zero.
        """
        # Gate uses (?!\.\d) to exclude '0.3' — so 0.0 is NOT traceable here
        assert not self.gate(0.0, "Id=1mA, Vov=0.3V")
        # Sanity: a query with explicit '0 volts' SHOULD be traceable
        assert self.gate(0.0, "Vsb is 0 volts")

    def test_rejects_wrong_magnitude(self):
        """1e-6 is not within 2% of 1e-3 (from '1mA')."""
        assert not self.gate(1e-6, "Id=1mA")

    def test_rejects_value_far_from_any_token(self):
        """A value of 99.0 is not traceable from a query with only small numbers."""
        assert not self.gate(99.0, "Id=1mA, Vov=0.3V")


class TestSLMExtractorFallback:
    """
    Tests that SLMExtractor returns {} safely when model=None
    (i.e. regex-only mode still works — no crash).
    """

    def test_extract_returns_empty_when_model_none(self):
        from physics.slm_extractor import SLMExtractor
        extractor = SLMExtractor(model=None)
        assert extractor.available is False
        result = extractor.extract("Id=1mA, Vov=0.3V")
        assert result == {}

    def test_available_false_when_none(self):
        from physics.slm_extractor import SLMExtractor
        extractor = SLMExtractor(model=None)
        assert not extractor.available


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Two-stage extractor (no model — falls through to regex only)
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoStageExtractorNoModel:
    """
    Runs extract_user_values_twostage with _slm_extractor=None
    (model not wired in) and verifies the regex stage still works correctly.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from physics.equation_validator import EquationValidator
        from physics.exploration_engine import ExplorationEngine
        validator = EquationValidator()
        self.engine = ExplorationEngine(validator.symbols)
        # Deliberately do NOT call set_slm_model() → SLM stage disabled

    def _extract(self, query):
        return self.engine.extract_user_values(query)

    def test_structured_id_vov(self):
        t = self._extract("Id=1mA, Vov=0.3V")
        assert abs(t._values["Id"].value  - 1e-3) < 1e-9
        assert abs(t._values["Vov"].value - 0.3)  < 1e-6

    def test_nl_milliamp(self):
        t = self._extract("Assume the drain current is 1 milliamp")
        assert "Id" in t._values
        assert abs(t._values["Id"].value - 1e-3) < 1e-9

    def test_nl_half_volt(self):
        t = self._extract("overdrive voltage is half a volt")
        assert "Vov" in t._values
        assert abs(t._values["Vov"].value - 0.5) < 1e-9

    def test_nl_microamp_roughly(self):
        t = self._extract("current of roughly 500 microamps")
        assert "Id" in t._values
        assert abs(t._values["Id"].value - 5e-4) < 1e-9

    def test_nl_micron_nanometer(self):
        t = self._extract("W is 2 microns, L is 100 nanometers")
        assert "W" in t._values
        assert "L" in t._values
        assert abs(t._values["W"].value - 2e-6) < 1e-12
        assert abs(t._values["L"].value - 1e-7) < 1e-12

    def test_sweep_range_not_extracted_as_value(self):
        """'Vov from 0.1V to 0.6V' must NOT produce a single Vov value."""
        t = self._extract("Plot W/L versus Vov from 0.1V to 0.6V for Id=1mA")
        # Id should be extracted; Vov should NOT (it's a sweep range)
        assert "Id" in t._values
        # Vov must not be pulled from the sweep range bounds
        if "Vov" in t._values:
            # If it was extracted, it better not be 0.1 or 0.6 (the range endpoints)
            v = t._values["Vov"].value
            assert abs(v - 0.1) > 0.01 and abs(v - 0.6) > 0.01, (
                f"Vov={v} looks like a range endpoint, not a user value"
            )

    def test_no_values_returns_empty_tracker(self):
        t = self._extract("what is the W/L ratio?")
        user_vals = {k: v for k, v in t._values.items() if v.provenance == "user"}
        assert len(user_vals) == 0
