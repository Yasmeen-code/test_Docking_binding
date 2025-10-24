import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, MACCSkeys, DataStructs, AllChem
import sascorer
import os

# Read CSV file
df = pd.read_csv(r"E:\TEST_SMILES\inference_drugs.csv")
print("Initial compounds:", len(df))

# Calculate QED and SA
df['QED'] = df['SMILES'].apply(lambda x: QED.qed(Chem.MolFromSmiles(x)))
df['SA'] = df['SMILES'].apply(lambda x: sascorer.calculateScore(Chem.MolFromSmiles(x)))
df_filtered = df[(df['QED'] > 0.5) & (df['SA'] < 4)].copy()
print("After QED & SA filter:", len(df_filtered))

# Tanimoto similarity with known inhibitors
known_inhibitors = ["CCO", "CCN"]

def max_tanimoto(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    max_sim = 0
    for k in known_inhibitors:
        mol2 = Chem.MolFromSmiles(k)
        fp2 = MACCSkeys.GenMACCSKeys(mol2)
        sim = DataStructs.TanimotoSimilarity(fp, fp2)
        if sim > max_sim:
            max_sim = sim
    return max_sim

df_filtered['Tanimoto'] = df_filtered['SMILES'].apply(max_tanimoto)
df_filtered = df_filtered[(df_filtered['Tanimoto'] >= 0.2) & (df_filtered['Tanimoto'] <= 0.8)]
print("After Tanimoto filter:", len(df_filtered))

# Remove compounds with unwanted atoms
unwanted_atoms = ['Se', 'Hg', 'As']

def has_unwanted_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return any(a in unwanted_atoms for a in atoms)

df_filtered = df_filtered[~df_filtered['SMILES'].apply(has_unwanted_atoms)]
print("After unwanted atoms filter:", len(df_filtered))

# Lipinski filter (LogP, MolWt)
df_filtered['LogP'] = df_filtered['SMILES'].apply(lambda x: Crippen.MolLogP(Chem.MolFromSmiles(x)))
df_filtered['MolWt'] = df_filtered['SMILES'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))
df_filtered = df_filtered[(df_filtered['MolWt'] < 600) & (df_filtered['LogP'] < 6)]
print("After Lipinski filter:", len(df_filtered))

# Save filtered compounds
output_csv = r"E:\TEST_SMILES\filtered_compounds_v2.csv"
df_filtered.to_csv(output_csv, index=False)
print(f"Filtered compounds saved to '{output_csv}'")

# Generate PDB files
dock_folder = r"E:\TEST_SMILES\docking_ligands"
os.makedirs(dock_folder, exist_ok=True)

for i, row in df_filtered.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    fname = os.path.join(dock_folder, f"ligand_{i}.pdb")
    Chem.MolToPDBFile(mol, fname)

print(f"PDB files created in '{dock_folder}'")
print("Ready for docking.")
