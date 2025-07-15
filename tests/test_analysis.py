from dotools_py.tl import get_expr


def test_get(adata):
    data = get_expr(adata, "gene1", "group")
