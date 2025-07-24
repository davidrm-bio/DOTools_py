from dotools_py.get import expr


def test_get(adata):
    data = expr(adata, "gene1", "group")
