import pytest
from physics.node_profile_manager import NodeProfileManager
from physics.value_tracker import TrackedValue

def test_node_profile_manager():
    manager = NodeProfileManager()
    
    # test list_nodes()
    nodes = manager.list_nodes()
    assert len(nodes) == 4
    assert "5nm_GAA" in nodes
    assert "16nm_FinFET" in nodes

    # test detect_from_query()
    assert manager.detect_from_query("what is the gm for 5nm GAA?") == "5nm_GAA"
    assert manager.detect_from_query("using the fd-soi node") == "28nm_FDSOI"
    assert manager.detect_from_query("gate width of 100nm") == "100nm_CMOS"
    assert manager.detect_from_query("just a regular question") is None

    # test as_tracker_defaults()
    defaults_5nm = manager.as_tracker_defaults("5nm_GAA")
    assert defaults_5nm["tox"].value == 0.8e-9
    assert defaults_5nm["Cox"].value == 0.06
    assert defaults_5nm["Vth"].value == 0.18

    defaults_100nm = manager.as_tracker_defaults("100nm_CMOS")
    assert defaults_100nm["tox"].value == 2e-9
    assert defaults_100nm["Vth"].value == 0.4

    # Test that missing node raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        manager.load("nonexistent_node")
