import dotools_py as do
from dotools_py._custom_class import InputError

def test_heart_markers():
    for species in ["mouse", "human"]:
        markers = do.dt.heart_markers(species)
        cts = ['Art_EC', 'CapEC', 'VeinEC', 'LymphEC', 'EndoEC',
               'SMC', 'PC', 'FB', 'FBa', 'Neurons', 'CM', 'B_cells',
               'T_cells', 'Myeloid', 'MP_recruit', 'MP_resident',
               'ImmuneCells', 'Epicardial', 'Adip', 'Mast']
        assert  isinstance(markers, dict), "Markers are not a dictionary"
        for ct in cts:
            assert ct in markers.keys(), f"{ct} not in the marker list"
    try:
        do.dt.heart_markers("unknown")
    except InputError:
        pass
    return

def test_standard_labels():
    labels = do.dt.standard_ct_labels_heart()
    assert isinstance(labels, dict), "Labels to updated is not a dictionary"
    return
