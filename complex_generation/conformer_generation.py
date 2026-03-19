import os
from rdkit import Chem
import shutil

# gen free ligand com files from ligand complex log file - code adapted from Dr. Jordan Dotson and Dr. Lucy Van Dijk 
class free_ligand_gen():
    work_dir ='/Users/theresewild/Sigman Group Dropbox/Therese Wild/NN_Library/dft_library_all/hydride_library_ligands/updated_basis_set/bpy/logs/problem_geometries'
    if not os.path.isdir(work_dir):
        raise ValueError("Directory does not exist")

    problem_dir = os.path.join(work_dir, "problem_geometries")
    os.makedirs(problem_dir, exist_ok=True)

def get_outstreams(log_path):
    """gets the compressed stream information at the end of a Gaussian job"""
    streams = []
    starts, ends = [], []
    error = "failed or incomplete job"

    try:
        with open(log_path) as f:
            loglines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot open {log_path}")

    for i, line in enumerate(loglines):
        if "1\\1\\" in line:
            starts.append(i)
        if "@" in line:
            ends.append(i)
        if "Normal termination" in line:
            error = ""

    if len(starts) != len(ends) or len(starts) == 0:
        return streams, "failed or incomplete job"

    for i in range(len(starts)):
        tmp = ""
        for j in range(starts[i], ends[i] + 1):
            tmp += loglines[j][1:-1]
        streams.append(tmp.split("\\"))

    return streams, error

def get_geom(streams):
    """extracts the geometry from the compressed stream"""
    geom = []
    try:
        for item in streams[-1][16:]:
            if item == "":
                break
            parts = item.split(",")
            geom.append([
                parts[0],
                float(parts[-3]),
                float(parts[-2]),
                float(parts[-1])
            ])
        return geom
    except Exception as e:
        print("Geometry extraction error:", e)
        return None

files_uncurated = os.listdir(work_dir)

log_files = [
    f[:-4] for f in files_uncurated
    if f.lower().endswith(".log") and "SPE" not in f
]

print(f"Processing {len(log_files)} log files")

for cat_name in log_files:
    log_path = os.path.join(work_dir, cat_name + ".log")

    streams, errors = get_outstreams(log_path)
    geometry = get_geom(streams)

    if geometry is None:
        print(f"geometry missing from input file {cat_name}")
        continue

    geometry_ok = True
    if geometry[-3][0] == 'Ni':
        geometry.pop(-1)
        geometry.pop(-1)
        geometry.pop(-1)
    elif geometry[-1][0] == 'Ni':
        geometry.pop(-1)
        geometry.pop(-1)
        geometry.pop(-1)
    else:
        print(f"metal geometry located elsewhere for file {cat_name}")
        geometry_ok = False

    free_filename = cat_name + "_free.com"
    free_path = os.path.join(work_dir, free_filename)

    with open(free_path, 'w') as new_com:
        new_com.write('# Put Keywords Here, check Charge and Multiplicity.\n\n')
        new_com.write(' title \n\n')
        new_com.write('0 1 \n')
        for atom in geometry:
            new_com.write(
                f"{atom[0]}\t{atom[1]}\t{atom[2]}\t{atom[3]}\n"
            )
        new_com.write('\n')

    # === MOVE PROBLEM FILES ===
    if not geometry_ok:
        shutil.move(log_path, os.path.join(problem_dir, os.path.basename(log_path)))
        shutil.move(free_path, os.path.join(problem_dir, free_filename))

# STUFF TO ADD NI
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


#### THIS IS ACTUALLY THE EDIT BOND ORDER STUFF 

SMARTS = ["[N]1C=C[N][Ni]1", '[N]1CC[N][Ni]1'] 
SMARTS = "[N]1C=C[N][Ni]1"

def enforce_substructure_bonds(mol, smarts):
    
    pattern = Chem.MolFromSmarts(smarts)
    matches = mol.GetSubstructMatches(pattern)

    for match in matches:
        N1, C1, C2, N2, NI = match
        print (N1, C1, C2, N2, NI)
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
        if filename.endswith(".sdf"):
            print (filename)
            filepath = os.path.join(directory, filename)
            suppl = Chem.SDMolSupplier(filepath, removeHs=False)
            mol = next(iter(suppl), None)
            print (mol)
            if mol:
                modified = enforce_substructure_bonds(mol, SMARTS)
                w = Chem.SDWriter(filepath)  # overwrite original file
                w.write(modified)
                w.close()
                print(f"Enforced bond orders in {filename}")
        elif filename.endswith(".xyz"):
            pass

# <-- RUN SCRIPT -->
if __name__ == "__main__":
    process_directory(".")  # runs on current directory




