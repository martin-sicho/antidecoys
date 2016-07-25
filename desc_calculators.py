import os

from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory

import numpy as np

FG_MIN_POINTS = 2
FG_MAX_POINTS = 3
FG_BINS = [(0, 2), (2, 5), (5, 8)]
def compute_fg_2D_pharm(smiles):
    FDEF_FILE = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef') # get basic feature definitions
    FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(FDEF_FILE) # make feature factory
    sig_fac = SigFactory(FEATURE_FACTORY, minPointCount=FG_MIN_POINTS, maxPointCount=FG_MAX_POINTS, trianglePruneBins=False)  # make signature factory
    sig_fac.SetBins(FG_BINS)
    sig_fac.Init()

    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fingerprints = [Generate.Gen2DFingerprint(mol, sig_fac) for mol in mols]

    fingerprints = np.array(fingerprints)
    return fingerprints, ['bit_{0}'.format(x) for x in range(len(fingerprints[0]))]