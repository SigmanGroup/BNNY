import os
import shutil
import numpy as np
from collections import defaultdict
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D

def smiles_to_sdf(df, smiles_col, id_col, out_dir):
    for i, row in df.iterrows():
        smiles = row[smiles_col]
        id = row[id_col]

        mol = Chem.MolFromSmiles(smiles)
        out_path = os.path.join(out_dir, f"{id}.sdf")
        writer = Chem.SDWriter(out_path)
        writer.write(mol)
        writer.close()


SMARTS_PATTERNS = [
    "[#7]~c~C=[N]",
    "[#7]-c-c=[#7]",
    "[N]=CC=[N]",
    "[N]=CCC=[N]",
    '[#7]~c~c~[#7]'
]

def add_ni(sdf_directory):
    for filename in os.listdir(sdf_directory):
        if not filename.endswith(".sdf"):
            continue

        filepath = os.path.join(sdf_directory, filename)
        supplier = Chem.SDMolSupplier(filepath)
        modified_mols = []

        for mol in supplier:
            if mol is None:
                print(f"unable to read molecule in {filename}")
                continue

            # Skip molecules that already contain Ni
            has_ni = any(atom.GetSymbol() == "Ni" for atom in mol.GetAtoms())
            if has_ni:
                modified_mols.append(mol)
                continue

            rw = Chem.RWMol(mol)
            conf = rw.GetConformer() if rw.GetNumConformers() > 0 else None
            matches = ()
            matched_smarts = None


            for smarts in SMARTS_PATTERNS:
                pattern = Chem.MolFromSmarts(smarts)
                matches = rw.GetSubstructMatches(pattern)
                if matches:
                    matched_smarts = smarts
                    break

            if not matches:
                print(f"{filename}: no SMARTS matched")
                modified_mols.append(rw.GetMol())
                continue

            for match in matches:
                if len(match) == 4:
                    N1, C1, C2, N2 = match
                    ni_idx = rw.AddAtom(Chem.Atom("Ni"))
                    rw.AddBond(N1, ni_idx, Chem.rdchem.BondType.DATIVE)
                    rw.AddBond(N2, ni_idx, Chem.rdchem.BondType.DATIVE)

                    def setbond(a, b, bond_type):
                        bond = rw.GetBondBetweenAtoms(a, b)
                        if bond is not None:
                            bond.SetBondType(bond_type)

                    setbond(C1, N1, Chem.rdchem.BondType.DOUBLE)
                    setbond(C2, N2, Chem.rdchem.BondType.DOUBLE)
                    setbond(C1, C2, Chem.rdchem.BondType.SINGLE)


                elif len(match) == 5:
                    N1, C1, C2, C3, N2 = match
                    ni_idx = rw.AddAtom(Chem.Atom("Ni"))
                    rw.AddBond(N1, ni_idx, Chem.rdchem.BondType.DATIVE)
                    rw.AddBond(N2, ni_idx, Chem.rdchem.BondType.DATIVE)
                else:
                    print(f"{filename}: unexpected match length {len(match)}")
                    continue


                if conf is not None:
                    posN1 = conf.GetAtomPosition(N1)
                    posN2 = conf.GetAtomPosition(N2)

                    midpoint = (posN1 + posN2) / 2
                    conf.SetAtomPosition(ni_idx, midpoint)

            new_mol = rw.GetMol()
            modified_mols.append(new_mol)

        writer = Chem.SDWriter(filepath)
        for mol in modified_mols:
            writer.write(mol)
        writer.close()

    return modified_mols

## functions for free ligand generation 
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
    

def process_log_files(log_dir, failed_subdir):
    """
    Process Gaussian log files and generate corresponding free geometry `.com` files to be submitted for SPE

    Parameters
    ----------
    log_dir : str
        Directory containing `.log` files.

    failed_subdir : str
        Name of the subdirectory where problematic files are moved.

    """

    failed_generation_dir = os.path.join(log_dir, failed_subdir)
    os.makedirs(failed_generation_dir, exist_ok=True)

    files_uncurated = os.listdir(log_dir)

    log_files = [
        f[:-4]
        for f in files_uncurated
        if f.lower().endswith(".log") and "SPE" not in f
    ]

    print(f"Processing {len(log_files)} log files")

    for cat_name in log_files:
        log_path = os.path.join(log_dir, f"{cat_name}.log")

        streams, errors = get_outstreams(log_path)
        geometry = get_geom(streams)

        if geometry is None:
            print(f"Geometry missing from input file: {cat_name}")
            continue

        geometry_ok = True

        # Remove Ni metal geometry if located at expected positions to make unbound ligand but keep geom 
        if geometry[-3][0] == "Ni":
            geometry.pop(-1)
            geometry.pop(-1)
            geometry.pop(-1)

        elif geometry[-1][0] == "Ni":
            geometry.pop(-1)
            geometry.pop(-1)
            geometry.pop(-1)

        else:
            print(f"Metal geometry located elsewhere for file: {cat_name}")
            geometry_ok = False

        free_filename = f"{cat_name}_ligand_only.com"
        free_path = os.path.join(log_dir, free_filename)

        with open(free_path, "w") as new_com:
            new_com.write(
                "# Put Keywords Here, check Charge and Multiplicity.\n\n"
            )
            new_com.write("title\n\n")
            new_com.write("0 1\n")

            for atom in geometry:
                new_com.write(
                    f"{atom[0]}\t{atom[1]}\t{atom[2]}\t{atom[3]}\n"
                )

            new_com.write("\n")

        # Move problematic files
        if not geometry_ok:
            shutil.move(
                log_path,
                os.path.join(
                    failed_generation_dir,
                    os.path.basename(log_path),
                ),
            )

            shutil.move(
                free_path,
                os.path.join(
                    failed_generation_dir,
                    free_filename,
                ),
            )