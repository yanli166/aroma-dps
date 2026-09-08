import pandas as pd
from rdkit import Chem
import torch

df=pd.read_csv('dearom_ring_pairs_A/ring_property_ml_samples.csv')

def parse_indices(s):
    return [int(x) for x in str(s).split(';') if x!='']

def get_sample(row):
    mol=Chem.MolFromSmiles(row['smiles'])
    idx=parse_indices(row['target_atom_indices'])
    ring_mask=torch.zeros(mol.GetNumAtoms(),dtype=torch.float32)
    ring_mask[idx]=1.0
    return {'sample_id':row['sample_id'],'mol':mol,'ring_mask':ring_mask,'side':row['side']}

sample=get_sample(df.iloc[0])
print(sample['sample_id'])
print(sample['ring_mask'])
