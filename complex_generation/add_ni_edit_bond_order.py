from rdkit import Chem
import os

SMARTS_WITH_NI = "[Ni]"
SMARTS_NO_NI   = "[#7]~c~C=[N]"   

def enforce_or_add_ni(mol):
    rw = Chem.RWMol(mol)  
    conf = rw.GetConformer() if rw.GetNumConformers() > 0 else None

    pattern_with_ni = Chem.MolFromSmarts(SMARTS_WITH_NI)
    pattern_no_ni   = Chem.MolFromSmarts(SMARTS_NO_NI)

    matches = rw.GetSubstructMatches(pattern_with_ni) #fixes bond orders if Ni present
    if matches:  
        for match in matches:
            N1, C1, C2, N2, NI = match

            def setbond(a, b, typ):
                bond = rw.GetBondBetweenAtoms(a, b)
                if bond:
                    bond.SetBondType(typ)
            setbond(C1, N1, Chem.rdchem.BondType.DOUBLE)
            setbond(C2, N2, Chem.rdchem.BondType.DOUBLE)
            setbond(C1, C2, Chem.rdchem.BondType.SINGLE)
            setbond(N1, NI, Chem.rdchem.BondType.DATIVE)
            setbond(N2, NI, Chem.rdchem.BondType.DATIVE)

    else:
        matches = rw.GetSubstructMatches(pattern_no_ni)
        if not matches:
            return mol 

        for match in matches:
            N1, C1, C2, N2 = match
            print (N1)
            ni_idx = rw.AddAtom(Chem.Atom("Ni"))
            rw.AddBond(N1, ni_idx, Chem.rdchem.BondType.DATIVE)
            rw.AddBond(N2, ni_idx, Chem.rdchem.BondType.DATIVE)

            def setbond(a, b, typ):
                bond = rw.GetBondBetweenAtoms(a, b)
                if bond:
                    bond.SetBondType(typ)
            print (C1)
            setbond(C1, N1, Chem.rdchem.BondType.DOUBLE)
            setbond(C2, N2, Chem.rdchem.BondType.DOUBLE)
            setbond(C1, C2, Chem.rdchem.BondType.SINGLE)

            if conf is not None:
                posN1 = conf.GetAtomPosition(N1)
                posN2 = conf.GetAtomPosition(N2)
                midpoint = (posN1 + posN2) / 2
                conf.SetAtomPosition(ni_idx, midpoint)

    return rw.GetMol()

def fix_pyridines (sdf_name, mol,smarts):
    pattern = Chem.MolFromSmarts(smarts[0])
    matches = mol.GetSubstructMatches(pattern)
    if len(matches) == 0:
        pattern = Chem.MolFromSmarts(smarts[1])
        matches = mol.GetSubstructMatches(pattern)
        if len(matches) == 0:
            pattern = Chem.MolFromSmarts(smarts[2])
            matches = mol.GetSubstructMatches(pattern)            
            if len(matches) == 0:
                print (f"some other pyridine issue {sdf_name}")
    for match in matches:
        C1, C3, C4, C5, C6, N1 = match
        print (C1, C3, C4, C5, C6, N1)
        bond = mol.GetBondBetweenAtoms(C1, C3)
        if bond:
            bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
        bond = mol.GetBondBetweenAtoms(C3, C4)
        if bond:
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        bond = mol.GetBondBetweenAtoms(C4, C5)
        if bond:
            bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
        bond = mol.GetBondBetweenAtoms(C5, C6)
        if bond:
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        bond = mol.GetBondBetweenAtoms(C6, N1)
        if bond:
            bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
    return mol

def process_directory(directory="."):
    for filename in os.listdir(directory):
        print (filename)
        if filename.endswith(".sdf") or filename.endswith(".mol"):
            filepath = os.path.join(directory, filename)
            suppl = Chem.SDMolSupplier(filepath, removeHs=False)
            mol = next(iter(suppl), None)
            if mol:
                modified = enforce_or_add_ni(mol)
                smi = Chem.MolToSmiles(modified)
                mol = Chem.MolFromSmiles(smi)  # sanitize
                fix_pyr = fix_pyridines(filename, mol, ['n1~cc~cc~c1', 'N1=CC=CC=C1', 'C1=CC=CC=N1'])
                # smi = Chem.MolToSmiles(fix_pyr)
                # mol = Chem.MolFromSmiles(smi)  # sanitize
                w = Chem.SDWriter(filepath)  # overwrite original file
                w.write(fix_pyr)
                w.close()
                print(f"added Ni to {filename}")

# <-- RUN SCRIPT -->
if __name__ == "__main__":
    process_directory(".")
