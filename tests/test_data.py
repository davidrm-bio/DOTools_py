import dotools_py as do


def test_heart_markers():
    data = do.dt.heart_markers()
    assert isinstance(data, dict)
    expected_cts = [
        "Art_EC",
        "CapEC",
        "VeinEC",
        "LymphEC",
        "EndoEC",
        "SMC",
        "PC",
        "FB",
        "FBa",
        "Neurons",
        "CM",
        "B_cells",
        "Myeloid",
        "MP_recruit",
        "MP_resident",
        "ImmuneCells",
        "Epicardial",
        "Adip",
        "Mast",
    ]
    for ct in expected_cts:
        assert ct in data.keys()


def test_standard_ct_labels():
    data = do.dt.standard_ct_labels_heart()
    assert isinstance(data, dict)
    labels = ["PC1_vent", "B", "vCM1", "DC"]  # A few labels to test

    for lb in labels:
        assert lb in data.keys()
