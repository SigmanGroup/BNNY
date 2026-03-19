import os
from rdkit import Chem

# <-- ENTER YOUR SMARTS HERE -->
SMARTS = ["[N]1C=C[N][Ni]1", '[N]1CC[N][Ni]1'] # replace with your substructure
SMARTS = "[N]1C=C[N][Ni]1"

def enforce_substructure_bonds(mol, smarts):
    
    pattern = Chem.MolFromSmarts(smarts)
    matches = mol.GetSubstructMatches(pattern)

    for match in matches:
        N1, C1, C2, N2, NI = match
        bond = mol.GetBondBetweenAtoms(C1, N1)
        if bond:
            bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
        bond = mol.GetBondBetweenAtoms(C2, N2)
        if bond:
            bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
        bond = mol.GetBondBetweenAtoms(C1, C2)
        if bond:
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        bond = mol.GetBondBetweenAtoms(N1, NI)
        if bond:
            bond.SetBondType(Chem.rdchem.BondType.DATIVE)
        bond = mol.GetBondBetweenAtoms(N2, NI)
        if bond:
            bond.SetBondType(Chem.rdchem.BondType.DATIVE)

    return mol

def process_directory(directory="."):
    for filename in os.listdir(directory):
        if filename.endswith(".sdf") or filename.endswith(".mol"):
            filepath = os.path.join(directory, filename)
            suppl = Chem.SDMolSupplier(filepath, removeHs=False)
            mol = next(iter(suppl), None)
            if mol:
                modified = enforce_substructure_bonds(mol, SMARTS)
                w = Chem.SDWriter(filepath)  # overwrite original file
                w.write(modified)
                w.close()
                print(f"Enforced bond orders in {filename}")
        elif filename.endswith(".xyz"):
            print(f"⚠️ Skipping {filename}: XYZ has no bonds; convert with OpenBabel first")

# <-- RUN SCRIPT -->
if __name__ == "__main__":
    process_directory(".")  # runs on current directory




