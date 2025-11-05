import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem import rdmolops
from torch.utils.data import Dataset
from rdkit.Chem import rdMolDescriptors

def generate_atom_on_ring_mask(mol, atom_on_ring_list):
    atom_on_ring = np.zeros((mol.GetNumAtoms(),), dtype=int)
    for atom_idx in atom_on_ring_list:
        atom_on_ring[atom_idx - 1] = 1
    return atom_on_ring

def create_mask_matrix(atom_on_ring, node_vec_len, n_atoms):
    extended_atom_on_ring = np.zeros((n_atoms,), dtype=int)
    extended_atom_on_ring[:len(atom_on_ring)] = atom_on_ring
    mask_matrix = np.zeros((n_atoms, node_vec_len), dtype=int)
    mask_matrix[extended_atom_on_ring == 1, :] = 1
    return mask_matrix

def process_and_save_data(dataset_path, node_vec_len, max_atoms):
    df = pd.read_csv(dataset_path)
    data2 = {
        'node_mats': [],
        'adj_mats': [],
        'outputs': [],
        'mask_mats': [],
        'indices': df.index.to_list(),
        'smiles': df["smiles"].to_list(),
        'atom_on_ring': df['atom_on_ring'].apply(lambda x: eval(x)).tolist(),
        'outputs_list': df["homa_value"].to_list()
    }
    
    for i in range(len(df)):
        smile = df.iloc[i]['smiles']
        atom_on_ring = data2['atom_on_ring'][i]
        output = data2['outputs_list'][i]

        mol = Graph(smile, atom_on_ring, node_vec_len, max_atoms)
        node_mat = mol.node_mat
        adj_mat = mol.adj_mat
        mask_mat = mol.mask_mat

        data2['node_mats'].append(node_mat)
        data2['adj_mats'].append(adj_mat)
        data2['outputs'].append([output])
        data2['mask_mats'].append(mask_mat)

    return data2

class Graph:
    def __init__(self, molecule_smiles: str, atom_on_ring: list, node_vec_len: int, max_atoms: int = None):
        self.smiles = molecule_smiles
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.atom_on_ring = atom_on_ring
        self.smiles_to_mol()
        if self.mol is not None:
            self.smiles_to_graph()
        else:
            raise ValueError("Invalid molecule SMILES.")

    def smiles_to_mol(self):
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            self.mol = None
            return
        self.mol = Chem.AddHs(mol)
        AllChem.ComputeGasteigerCharges(self.mol)
        
    def smiles_to_graph(self):
        atoms = self.mol.GetAtoms()
        n_atoms = len(list(atoms)) if self.max_atoms is None else self.max_atoms
        node_mat = np.zeros((n_atoms, self.node_vec_len))

        for atom in atoms:
            atom_index = atom.GetIdx()
            if atom_index >= n_atoms:
                continue
                
            atom_no = atom.GetAtomicNum()
            node_mat[atom_index, atom_no] = 10
            node_mat[atom_index, self.node_vec_len - 1] = float(atom.GetProp('_GasteigerCharge'))
            
            # Hybridization state
            hybridization = atom.GetHybridization()
            if hybridization == Chem.rdchem.HybridizationType.SP:
                node_mat[atom_index, self.node_vec_len - 2] = 1
            elif hybridization == Chem.rdchem.HybridizationType.SP2:
                node_mat[atom_index, self.node_vec_len - 3] = 1
            elif hybridization == Chem.rdchem.HybridizationType.SP3:
                node_mat[atom_index, self.node_vec_len - 4] = 1
            
            # Aromaticity
            node_mat[atom_index, self.node_vec_len - 5] = int(atom.GetIsAromatic())
            # Number of neighbors
            node_mat[atom_index, self.node_vec_len - 6] = len(atom.GetNeighbors())
            # Total hydrogens
            node_mat[atom_index, self.node_vec_len - 7] = atom.GetTotalNumHs()
            
            # Neighbors with oxygen/nitrogen
            has_oxygen_neighbor = any(neighbor.GetSymbol() == "O" for neighbor in atom.GetNeighbors())
            has_nitrogen_neighbor = any(neighbor.GetSymbol() == "N" for neighbor in atom.GetNeighbors())
            
            # Bonds with oxygen/nitrogen
            has_double_bond_with_oxygen = False
            has_double_bond_with_nitrogen = False
            for neighbor in atom.GetNeighbors():
                bond = self.mol.GetBondBetweenAtoms(atom_index, neighbor.GetIdx())
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    if neighbor.GetSymbol() == "O":
                        has_double_bond_with_oxygen = True
                    elif neighbor.GetSymbol() == "N":
                        has_double_bond_with_nitrogen = True
            
            node_mat[atom_index, self.node_vec_len - 8] = int(has_oxygen_neighbor) * 10
            node_mat[atom_index, self.node_vec_len - 9] = int(has_double_bond_with_oxygen) * 10
            node_mat[atom_index, self.node_vec_len - 10] = int(has_nitrogen_neighbor) * 10
            node_mat[atom_index, self.node_vec_len - 11] = int(has_double_bond_with_nitrogen) * 10

        # Create dense adjacency matrix
        adj_mat = np.zeros((n_atoms, n_atoms))
        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol)
        self.std_adj_mat = np.copy(adj_mat)
        dist_mat = molDG.GetMoleculeBoundsMatrix(self.mol)
        dist_mat[dist_mat == 0.0] = 1
        adj_mat = adj_mat * (1 / dist_mat)
        dim_add = n_atoms - adj_mat.shape[0]
        adj_mat = np.pad(adj_mat, pad_width=((0, dim_add), (0, dim_add)), mode="constant")
        adj_mat = adj_mat + np.eye(n_atoms)

        self.node_mat = node_mat
        self.adj_mat = adj_mat

        if not isinstance(self.atom_on_ring, list):
            raise ValueError("atom_on_ring must be a list of atom indices.")
        
        atom_on_ring_mask = generate_atom_on_ring_mask(self.mol, self.atom_on_ring)
        self.mask_mat = create_mask_matrix(atom_on_ring_mask, self.node_vec_len, n_atoms)


class GraphData(Dataset):
    def __init__(self, data2: dict, node_vec_len: int, max_atoms: int):
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.data2 = data2  

    def __len__(self):
        return len(self.data2['node_mats'])

    def __getitem__(self, i: int):
        node_mat = torch.Tensor(self.data2['node_mats'][i])
        adj_mat = torch.Tensor(self.data2['adj_mats'][i])
        output = torch.Tensor(self.data2['outputs'][i])
        smile = self.data2['smiles'][i]
        mask_mat = torch.Tensor(self.data2['mask_mats'][i])

        return (node_mat, adj_mat), output, smile, mask_mat

    def get_atom_no_sum(self, i):
        node_mat = self.data2['node_mats'][i]
        one_pos_mat = np.argwhere(node_mat == 1)
        atomic_no_sum = one_pos_mat[:, -1].sum()
        return atomic_no_sum

def collate_graph_dataset(dataset: Dataset):
    node_mats = []
    adj_mats = []
    outputs = []
    smiles = []
    mask_mats = []
    
    for i in range(len(dataset)):
        (node_mat, adj_mat), output, smile, mask_mat = dataset[i]
        node_mats.append(node_mat)
        adj_mats.append(adj_mat)
        outputs.append(output)
        smiles.append(smile) 
        mask_mats.append(mask_mat)
    
    # Create tensors
    node_mats_tensor = torch.stack(node_mats)
    adj_mats_tensor = torch.stack(adj_mats)
    mask_mats_tensor = torch.stack(mask_mats)
    outputs_tensor = torch.stack(outputs).squeeze(1)

    return (node_mats_tensor, adj_mats_tensor, mask_mats_tensor), outputs_tensor, smiles