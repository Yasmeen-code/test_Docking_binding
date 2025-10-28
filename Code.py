import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, MACCSkeys, DataStructs, AllChem
import sascorer
import os

# =====================================
# Read CSV file
# =====================================
df = pd.read_csv(r"E:\TEST_SMILES\inference_drugs.csv")
print("Initial compounds:", len(df))

# =====================================
# Calculate QED and SA
# =====================================
df['QED'] = df['SMILES'].apply(lambda x: QED.qed(Chem.MolFromSmiles(x)))
df['SA'] = df['SMILES'].apply(lambda x: sascorer.calculateScore(Chem.MolFromSmiles(x)))
df_filtered = df[(df['QED'] > 0.5) & (df['SA'] < 4)].copy()
print(df_filtered[['SMILES', 'QED', 'SA']])
print("After QED & SA filter:", len(df_filtered))

# =====================================
# Known inhibitors (scientifically validated)
# =====================================
known_inhibitors = [
    # EGFR inhibitor
    "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC=C3C#C)OCCOC",  # Erlotinib

    # SARS-CoV-2 main protease (3CLpro) inhibitor
    "N#C[C@H](C[C@@H]1CCNC1=O)NC(=O)[C@H]1N(C[C@H]2[C@@H]1C2(C)C)C(=O)[C@H](C(C)(C)C)NC(=O)C(F)(F)(F)",  # Nirmatrelvir

    # BCR-ABL tyrosine kinase inhibitor
    "CC1=CC(=O)NC2=CC=CC=C2N1CC3=CC=CC=C3",  # Imatinib

    # Multi-kinase inhibitor (VEGFR, PDGFR)
    "CCNC1=NC(=C(C(=N1)C2=CC=C(C=C2)OC)C(=O)NC3=CC=C(C=C3)Cl)C4=CC=C(C=C4)F"  # Sorafenib
]

# =====================================
# Tanimoto similarity function
# =====================================
def max_tanimoto(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    fp = MACCSkeys.GenMACCSKeys(mol)
    max_sim = 0
    for k in known_inhibitors:
        mol2 = Chem.MolFromSmiles(k)
        if mol2 is None:
            continue
        fp2 = MACCSkeys.GenMACCSKeys(mol2)
        sim = DataStructs.TanimotoSimilarity(fp, fp2)
        if sim > max_sim:
            max_sim = sim
    return max_sim

# =====================================
# Apply Tanimoto filtering
# =====================================
#df_filtered['Tanimoto'] = df_filtered['SMILES'].apply(max_tanimoto)
#df_filtered = df_filtered[(df_filtered['Tanimoto'] >= 0.1) & (df_filtered['Tanimoto'] <= 0.9)]
#print("After Tanimoto filter:", len(df_filtered))

# =====================================
# Remove compounds with unwanted atoms
# =====================================
unwanted_atoms = ['Se', 'Hg', 'As']

def has_unwanted_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return any(a in unwanted_atoms for a in atoms)

df_filtered = df_filtered[~df_filtered['SMILES'].apply(has_unwanted_atoms)]
print("After unwanted atoms filter:", len(df_filtered))

# =====================================
# Lipinski filter (LogP, MolWt)
# =====================================
df_filtered['LogP'] = df_filtered['SMILES'].apply(lambda x: Crippen.MolLogP(Chem.MolFromSmiles(x)))
df_filtered['MolWt'] = df_filtered['SMILES'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))
df_filtered = df_filtered[(df_filtered['MolWt'] < 600) & (df_filtered['LogP'] < 6)]
print("After Lipinski filter:", len(df_filtered))

# =====================================
# Save filtered compounds
# =====================================
output_csv = r"E:\TEST_SMILES\filtered_compounds_v2.csv"
df_filtered.to_csv(output_csv, index=False)
print(f"Filtered compounds saved to '{output_csv}'")

# =====================================
# Generate PDB files for docking
# =====================================
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