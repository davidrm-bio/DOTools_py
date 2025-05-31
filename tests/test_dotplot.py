from dotools_py.pl import dotplot
from dotools_py._settings import iOff



def test_dotplot(adata):
    iOff()
    dotplot(adata, 'group', 'gene1')
