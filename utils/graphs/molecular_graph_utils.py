import os
import pickle
import multiprocessing
import itertools
import time

import deepchem as dc
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import BondType


def get_atoms_adj_from_mol(mol, max_num_nodes, label_map, categorical_adj=False, no_edge_type=False):
    # Remove Aromatic bond marking
    Chem.Kekulize(mol)

    if categorical_adj:
        adj = np.zeros(shape=(max_num_nodes, max_num_nodes), dtype=np.float32)
        bond_types = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3} if not no_edge_type else {
            BondType.SINGLE: 1, BondType.DOUBLE: 1, BondType.TRIPLE: 1}
    else:
        if no_edge_type:
            adj = np.zeros(shape=(2, max_num_nodes, max_num_nodes), dtype=np.float32)
            bond_types = {BondType.SINGLE: 0, BondType.DOUBLE: 0, BondType.TRIPLE: 0}
        else:
            # single, double, triple, no-bond
            adj = np.zeros(shape=(4, max_num_nodes, max_num_nodes), dtype=np.float32)
            bond_types = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}
        adj[-1] = 1

    atoms = np.zeros(shape=max_num_nodes, dtype=np.int)
    # skip no bond molecules
    if len(mol.GetBonds()) == 0:
        return None, None, label_map

    for atom in mol.GetAtoms():

        atom_idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        if symbol not in label_map:
            label_map[symbol] = len(label_map) + 1
        atoms[atom_idx] = label_map[symbol]

        # If one atom doesn't have neighbors, it is probably because of the hydrogens removing, skip the molecule
        if len(atom.GetNeighbors()) == 0:
            return None, None, label_map

        for atom_neig in atom.GetNeighbors():
            neig_idx = atom_neig.GetIdx()
            bond_type = mol.GetBondBetweenAtoms(atom_idx, neig_idx).GetBondType()

            if categorical_adj:
                adj[atom_idx, neig_idx] = bond_types[bond_type]
            else:
                adj[bond_types[bond_type]][atom_idx, neig_idx] = 1
                adj[bond_types[bond_type]][neig_idx, atom_idx] = 1
                # remove from the no-bond channel
                adj[-1][atom_idx, neig_idx] = 0
                adj[-1][neig_idx, atom_idx] = 0

    return atoms, adj, label_map


def atoms_to_one_hot(atoms, label_map):
    atoms = np.array(atoms, dtype=np.int)
    # Add one more 0 cols for virtual node padding
    x = np.zeros((atoms.size, len(label_map) + 1)).astype(np.float32)
    x[np.where(atoms != 0), atoms[np.where(atoms != 0)] - 1] = 1

    # set to 1 the last column for virtual nodes
    x[np.where(atoms == 0), x.shape[1] - 1] = 1

    return x


def get_molecular_dataset(name='qm7', data_path=None):
    if 'qm7' in name:
        tasks, (train_dataset, valid_dataset, test_dataset), transformers = dc.molnet.load_qm7_from_mat(
            featurizer='Raw', data_dir=data_path, save_dir=data_path)
        label_map = {'O': 1, 'N': 2, 'C': 3, 'S': 4}
        max_num_nodes = 7
    elif 'qm9' in name:
        tasks, (train_dataset, valid_dataset, test_dataset), transformers = dc.molnet.load_qm9(featurizer='Raw',
                                                                                               data_dir=data_path,
                                                                                               save_dir=data_path)
        label_map = {'C': 1, 'O': 2, 'N': 3, 'F': 4}
        max_num_nodes = 9
    else:
        assert False, 'unknown dataset'

    label_map = {}
    input_li = []
    adjs = []
    xs = []
    all_mol = np.concatenate((train_dataset.X, valid_dataset.X, test_dataset.X), axis=0)
    # Remove Hs
    for i, mol in enumerate(all_mol):
        # mol2 = Chem.RemoveHs(mol, sanitize=False)
        all_mol[i] = Chem.RemoveAllHs(mol, sanitize=False)
    # max_num_nodes = max([len(all_mol[i].GetAtoms()) for i in range(len(all_mol))])
    for i, mol in enumerate(all_mol):
        atoms, adj, label_map = get_atoms_adj_from_mol(mol, max_num_nodes=max_num_nodes, label_map=label_map,
                                                       no_edge_type='no_edge' in name)
        if atoms is None or adj is None:
            continue
        # No dupes
        val = list(np.concatenate((atoms, adj.flatten())))
        if val not in input_li:
            input_li.append(val)
            adjs.append(adj)
            xs.append(atoms)

    for i, atoms in enumerate(xs):
        xs[i] = atoms_to_one_hot(atoms, label_map)

    return xs, adjs, label_map


def chunks(lst, n):
    for i in range(0, len(lst), int(len(lst) / n)):
        yield lst[i:i + int(len(lst) / n)]


def get_molecular_dataset_mp(name='qm7', data_path=None, categorical_values=False, return_smiles=False):
    if 'qm7' in name:
        tasks, (train_dataset, valid_dataset, test_dataset), transformers = dc.molnet.load_qm7_from_mat(
            featurizer='Raw', data_dir=data_path, save_dir=data_path)
        dupe_filter = True
        label_map = {'O': 1, 'N': 2, 'C': 3, 'S': 4}
        max_num_nodes = 7
        train_y = train_dataset.y
        val_y = valid_dataset.y
        test_y = test_dataset.y
    elif 'qm9' in name:
        tasks, (train_dataset, valid_dataset, test_dataset), transformers = dc.molnet.load_qm9(featurizer='Raw',
                                                                                               data_dir=data_path,
                                                                                               save_dir=data_path)
        dupe_filter = False
        label_map = {'C': 1, 'O': 2, 'N': 3, 'F': 4}
        max_num_nodes = 9

        # select property (atomisation energy at 0K -> 8
        train_y = train_dataset.y[:, 8].squeeze()
        val_y = valid_dataset.y[:, 8].squeeze()
        test_y = test_dataset.y[:, 8].squeeze()
    else:
        assert False, 'unknown dataset'

    train_mols = train_dataset.X
    val_mols = valid_dataset.X
    test_mols = test_dataset.X
    # all_mol = np.concatenate((train_dataset.X, valid_dataset.X, test_dataset.X), axis=0)
    # all_y = np.concatenate((train_dataset.y, valid_dataset.y, test_dataset.y), axis=0).astype(np.float)

    train_data = (np.concatenate((train_mols, val_mols), axis=0),
                  np.concatenate((train_y, val_y), axis=0))
    test_data = (test_mols, test_y)
    results, label_map = process_mols(name, train_data, test_data, max_num_nodes, label_map, dupe_filter,
                                      categorical_values, return_smiles)

    return results, label_map


def process_mols(name, train_data, test_data, max_num_nodes, label_map, dupe_filter, categorical_values=False,
                 return_smiles=False):
    results = []
    for mols, y in (train_data, test_data):

        pool = multiprocessing.Pool()
        returns = pool.map(multiprocessing_func,
                           [(chunk, i, max_num_nodes, label_map, name, categorical_values, return_smiles) for i, chunk
                            in
                            enumerate(chunks(list(zip(mols, y)), os.cpu_count()))])
        pool.close()
        returns = list(itertools.chain.from_iterable(returns))
        returns = list(filter(None, returns))

        if dupe_filter:
            # Filter Dupes
            seen = []
            res = []
            for x in returns:
                v = list(np.concatenate((x[0].flatten(), x[1].flatten())))
                if v not in seen and not seen.append(v):
                    res.append(x)
        else:
            res = returns

        if return_smiles:
            xs, adjs, smiles, y2 = map(list, zip(*res))
            results.append((xs, adjs, smiles, y2))
        else:
            xs, adjs, y2 = map(list, zip(*res))
            results.append((xs, adjs, y2))
    return results, label_map


def multiprocessing_func(input_args):
    chunk, i, max_num_nodes, label_map, name, categorical_values, return_smiles = input_args
    res = []
    for mol, y in chunk:

        mol = Chem.RemoveAllHs(mol, sanitize=False)
        atoms, adj, label_map = get_atoms_adj_from_mol(mol, max_num_nodes=max_num_nodes, label_map=label_map,
                                                       categorical_adj=categorical_values,
                                                       no_edge_type='no_edge' in name)
        if atoms is None or adj is None:
            continue

        if not categorical_values:
            atoms = atoms_to_one_hot(atoms, label_map)
        if return_smiles:
            smiles = Chem.MolToSmiles(mol)
            res.append([atoms, adj, smiles, y])
        else:
            res.append([atoms, adj, y])

    return res


if __name__ == '__main__':
    xs, adjs, label_map = get_molecular_dataset('qm9')
    graphs = list(zip(xs, adjs))
    with open('graphs_qm9.pkl', 'wb') as f:
        pickle.dump(graphs, f)
