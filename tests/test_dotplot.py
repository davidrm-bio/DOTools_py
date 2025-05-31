from dotools_py.pl import dotplot




def test_dotplot(adata):
    dotplot(adata, 'group', 'gene1')
