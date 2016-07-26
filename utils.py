import pickle
from rdkit.Chem import PandasTools

from desc_calculators import compute_fg_2D_pharm
import pandas as pd
from matplotlib import pyplot as plt

def get_dataset(paths, desc_calculator=compute_fg_2D_pharm):
    smiles_all = []
    for path in paths:
        smiles_all.extend(path[1:-1])

    data_frame = pd.DataFrame(columns=['SMILES'])
    data_frame['SMILES'] = smiles_all
    PandasTools.AddMoleculeColumnToFrame(data_frame,'SMILES')
    del data_frame['SMILES']

    descs, labels = desc_calculator(smiles_all)
    descs_frame = pd.DataFrame(descs, columns=labels)

    return pd.concat([data_frame, descs_frame], axis=1)

def read_paths(filepath):
    paths = pickle.load(open(filepath, 'rb'))
    return [x for x in paths if x]

def pickle_data(data, filepath):
    pickle.dump(data, open(filepath, 'wb'))

def unpickle_data(filepath):
    return pickle.load(open(filepath, 'rb'))

def do_PCA(data, n_components=5):
    from sklearn import decomposition

    pca = decomposition.PCA(n_components=n_components)
    pca.fit(data)
    pca_result = pca.transform(data)
    eigen_values = pca.explained_variance_ratio_

    return pca_result, eigen_values

def plot_eigenvalues(values, n_components=5, size=(5, 5)):
    plt.rcParams["figure.figsize"] = size
    plt.bar(range(1, n_components + 1), values)
    plt.show()

def plot_PCA_3D(data, c, n_components=5):
    from mpl_toolkits.mplot3d import Axes3D
    from itertools import combinations

    combos = list(combinations(range(n_components), 3))

    plt.rcParams["figure.figsize"] = [15, 30]
    fig = plt.figure(len(combos) / 2)

    for idx, combo in enumerate(combos):
        ax = fig.add_subplot(len(combos) / 2, 2, idx + 1, projection='3d')
        ax.scatter(
            data[:,combo[0]]
            , data[:,combo[1]]
            , data[:,combo[2]]
            , c=c
            , s=20
            , cmap='winter'
        )
        ax.view_init(elev=30, azim=45)
        ax.set_xlabel('PC%s' % (combo[0] + 1))
        ax.set_ylabel('PC%s' % (combo[1] + 1))
        ax.set_zlabel('PC%s' % (combo[2] + 1))

    plt.show()

def plot_PCA_2D(data, c=None, n_components=5, color='blue'):
    from itertools import combinations

    combos = list(combinations(range(n_components), 2))

    plt.rcParams["figure.figsize"] = [15, 30]
    fig = plt.figure(len(combos) / 2)

    for idx, combo in enumerate(combos):
        ax = fig.add_subplot(len(combos) / 2, 2, idx + 1)
        if c is not None:
            ax.scatter(
                data[:,combo[0]]
                , data[:,combo[1]]
                , c=c
                , s=20
                , cmap='winter'
            )
        else:
            ax.scatter(
                data[:,combo[0]]
                , data[:,combo[1]]
                , color=color
            )
        ax.set_xlabel('PC%s' % (combo[0] + 1))
        ax.set_ylabel('PC%s' % (combo[1] + 1))

    plt.show()
