from typing import Literal

from dotools_py import logger


def heart_markers(
    species: Literal["mouse", "human"] = "mouse",
) -> dict:
    """Marker genes for cell-types in the heart.

    :param species: Format for gene names. Set `human` for everything upper case and `mouse` for capitalise.
    :return: Returns a dictionary with a list of marker genes for major cell-types present in the heart.

    Example
    -------
    >>> import dotools_py as do
    >>> df_mouse = do.dt.heart_markers("mouse")
    2025-07-02 10:55:01,623 - Getting mouse markers
    >>> df_mouse["EndoEC"]
    ['Nfatc1', 'Npr3', 'Nrg1', 'Pecam1', 'Cdh5', 'Etv2']

    **Summary of Markers genes**

    .. image:: MarkersCellsHeart.png

    """
    logger.info(f"Getting {species} markers")
    species = species.lower()

    mouse = {
        "Art_EC": ["Rbp7", "Ly6a", "Id1", "Stmn2", "Fbln5", "Glul", "Cxcl12", "Sox17", "Hey1", "Mgll", "Dusp1",
                   "Alpl", "Btg2", "Klf4", "Crip1"],  # Refined with Kalucka et al., Cell, 2022
        "CapEC": ["Cd36", "Fabp4", "Aqp1", "Rgcc", "Gpihbp1", "Aplnr", "Lpl", "Sparcl1", "Car4", "Sparc",
                  "Tcf15", "Sgk1", "Kdr", "Cav1", "Vwa1"],  # Refined with Kalucka et al., Cell, 2022
        "VeinEC": ["Mgp", "Cfh", "Apoe", "Cpe", "Bgn", "Vwf", "Fabp5", "Vcam1", "H19", "Tmem108", "F2r",
                   "Ptgs1", "Il6st", "Vim", "Comp"],  # Refined with Kalucka et al., Cell, 2022
        "LymphEC": ["Prox1", "Lyve1", "Pdpn", "Ccl21a", "Fgl2", "Mmrn1", "Lcn2", "Nts", "Cp", "Reln",
                    "Cd63", "Maf", "Lmo2", "Ntn1", "Anxa1"],  # Refined with Kalucka et al., Cell, 2022
        "EndoEC": ["Nfatc1", "Npr3", "Nrg1", "Pecam1", "Cdh5", "Etv2"],  # ZMM_shared JW
        "SMC": ["Myh11", "Itga8", "Acta2", "Tagln", "Carmn", "Kcnab1", "Ntrk3", "Rcan2"],  # Refined
        "PC": ["Rgs5", "Abcc9", "Gucy1a2", "Egflam", "Dlc1", "Pdgfrb", "Des", "Cd248", "Mcam"],  # Refined
        "FB": ["Dcn", "Abca9", "Mgp", "Lama2", "Abca6", "Gsn", "Pdgfra", "Vim", "Fap", "Pdgfrb"],  # Refined
        "FBa": ["Postn"],  # Refined JW ZMM_shared
        "Neurons": ["Nrxn1", "Cadm2", "Chl1", "Kirrel3", "Sorcs1", "Ncam2", "Pax3"],  # Refined
        "CM": ["Ryr2", "Mlip", "Ttn", "Fhl2", "Rbm20", "Ankrd1", "Tecrl", "Mybpc3", "Tnni3", "Myh7", "Mybpc3",
               "Irx4"],  # Refined
        "B_cells": ["Igkc", "Ighm", "Aff3", "Cd74", "Bank1", "Ms4a1", "Cd79a", "Cd69"],  # Refined
        "T_cells": ["Il7r", "Themis", "Skap1", "Cd247", "Itk", "Ptprc", "Camk4", "Cd3e", "Cd3d",
                    "Cd4", "Cd8a"],  # Refined
        "Myeloid": ["F13a1", "Rbpj", "Cd163", "Rbm47", "Mrc1", "Fmn1", "Msr1", "Frmd4b", "Mertk", "Lyz2"],  # Refined
        "MP_recruit": ["Ccr2"],
        "MP_resident": ["Lyve1", "Timd4"],
        "ImmuneCells": ["Ptprc"],
        "Epicardial": ["Pdzrn4", "Slc39a8", "Gfpt2", "C3", "Wwc1", "Kcnt2", "Wt1", "Dpp4", "Ano1"],  # Refined
        "Adip": ["Gpam", "Adipoq", "Acacb", "Ghr", "Pde3b", "Fasn", "Prkar2b", "Plin1", "Pparg"],  # Refined
        "Mast": ["Il18r1", "Kit", "Slc24a3", "Ntm", "Cpa3", "Slc8a3", "Cdk15", "Hpgds", "Slc38a11",
                 "Rab27b"],  # Refined
    }  # Cell Type Markers in Mouse Format

    human = {cell: [gene.upper() for gene in mouse[cell]] for cell in mouse}  # Cell Type Markers in Human Format

    if species == "mouse":
        return mouse
    elif species == "human":
        return human
    else:
        raise Exception("Species not recognise")


def standard_ct_labels_heart() -> dict:
    """Set common cell-type labels in the Human Heart Model from Celltypist.

    This set a common an informative label for the cell-types in the heart model. For example, instead of using
    EC1_cap, EC3_cap, the common label CapEC can be used. The nature of the subtypes of cell-types needs to be
    investigated on the dataset. The model might assigned a cell-type based on similarity, however, the assignment
    might be incorrect if the cell-type is not present in the model.

    :return: Returns a dictionary with the labels from the model as keys and the updated labels as values.

    Example
    -------
    >>> import dotools_py as do
    >>> labels = do.dt.standard_ct_labels_heart()
    >>> labels
    {'PC1_vent': 'Pericytes',
    'SMC1_basic': 'SMC',
    'SMC2_art': 'SMC',
    'CD16+Mo': 'Ccr2_MP',
    'LYVE1+IGF1+MP': 'MP',
    'B_plasma': 'B_cells',
    'B': 'B_cells',
    'CD4+T_naive':
    'T_cells',
    'EC1_cap': 'CapEC',
    'EC3_cap': 'CapEC',
    'EC5_art': 'ArtEC',
    'EC6_ven': 'VeinEC',
    'EC7_endocardial': 'EndoEC',
    'EC8_ln': 'LymphEC',
    'FB3': 'Fibroblasts',
    'FB4_activated': 'Fibro_activ',
    'FB5': 'Fibroblasts',
    'Meso': 'Epi_cells',
    'vCM1': 'CM',
    'Adip1': 'Adip',
    'NC1_glial': 'Neural',
    'LYVE1+TIMD4+MP': 'MP',
    'MoMP': 'MP',
    'DC': 'Dendritic',
    'Mast': 'Mast',
    'FB1': 'Fibroblasts',
    'CD8+T_trans': 'T_cells',
    'vCM4': 'CM',
    'NC2_glial_NGF+': 'NC_glial_NGF+',
    'NK_CD16hi': 'NK'
    }

    """
    return {
        "PC1_vent": "Pericytes",
        "SMC1_basic": "SMC",  # SMC1_basic has transcripts that indicate immaturity
        "SMC2_art": "SMC",
        "CD16+Mo": "Ccr2_MP",
        "LYVE1+IGF1+MP": "MP",
        "B_plasma": "B_cells",
        "B": "B_cells",
        "CD4+T_naive": "T_cells",
        "EC1_cap": "CapEC",
        "EC3_cap": "CapEC",
        "EC5_art": "ArtEC",
        "EC6_ven": "VeinEC",
        "EC7_endocardial": "EndoEC",
        "EC8_ln": "LymphEC",
        "FB3": "Fibroblasts",  # FB3 is less abundant in the left ventricle, more in the atria
        "FB4_activated": "Fibro_activ",  # More abundant in LV
        "FB5": "Fibroblasts",  # Less abundant in the right atrium
        "Meso": "Epi_cells",
        "vCM1": "CM",  # Mainly in the left ventricle
        "Adip1": "Adip",  # Genes for PPAR pathway, metabolism of lipids and lipoproteins and lipolysis
        "NC1_glial": "Neural",  # Neural cells with a gene program required for glia development and axon myelination
        "LYVE1+TIMD4+MP": "MP",  # Positive for TIMD4 predicted to act on apoptotic cell clearance
        "MoMP": "MP",
        "DC": "Dendritic",
        "Mast": "Mast",
        "FB1": "Fibroblasts",
        "CD8+T_trans": "T_cells",
        "vCM4": "CM",
        "NC2_glial_NGF+": "NC_glial_NGF+",
        "NK_CD16hi": "NK",
    }
