from carbontwin_r.data.generate import make_synthetic_facility
from carbontwin_r.simulation.faults import inject_fault

def test_fault_injection_preserves_known_ground_truth():
    h = make_synthetic_facility(n_steps=500)
    f = inject_fault(h, fault_type="sudden_cooling", start_fraction=0.5, severity=0.2)
    assert f["is_fault"].sum() > 0
    assert f["true_wasted_energy"].sum() > 0
    assert (f["total_power"] >= h["total_power"]).all()
