
import os
import shutil
import numpy as np
import re
from collections import defaultdict
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D

def molecular_weight_filter(df, weight_limit=500.0):
    """
    Flags molecules with molecular weight above the specified limit.
    Modifies 'library_status' and 'status_notes' columns in-place.
    """
    removed = 0

    for i, row in df.iterrows():
        if row['library_status'] == 'removed':
            continue  # Skip already removed

        smi = row['SMILES']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            df.at[i, 'library_status'] = 'removed'
            df.at[i, 'status_notes'] = 'invalid SMILES'
            removed += 1
            continue

        weight = Descriptors.ExactMolWt(mol)
        if weight >= weight_limit:
            df.at[i, 'library_status'] = 'removed'
            df.at[i, 'status_notes'] = f'molecular weight {weight:.2f} exceeds limit ({weight_limit})'
            removed += 1

    print(f"{removed} ligands over molecular weight limit were marked for removal.")
    return df

def append_reason(df, idx, new_reason, status_col='library_status', reason_col='status_notes'):
    current_reason = df.at[idx, reason_col]
    if pd.isna(current_reason) or current_reason == '':
        df.at[idx, reason_col] = new_reason
    else:
        df.at[idx, reason_col] += f"; {new_reason}"
    df.at[idx, status_col] = 'removed'

def forbidden_substructures_filter(df):
    """
    Flags molecules with substructures excluded from library for removal.
    Modifies 'library_status' and 'status_notes' columns in-place.
    """
    forbidden = [
        Chem.MolFromSmarts(p) for p in [
            'c1cc(C2=NCCO2)nc(C2=NCCO2)c1',  # pybox
            'c1ccc(-c2cccc(C3=NCCO3)n2)nc1',  # bybox
            'c1ccc(-c2cccc(-c3ccccn3)n2)nc1',  # byby
            'c1(c2c(c3ncccc3)nccc2)ncccc1',  # triby
            'c1cnc2c(c1)ccc1ccc(C3=NCCO3)nc12',  # phenBox
            'c1cnc2nc(C3=NCCO3)ccc2c1',  # weirdbpy
            'c1ccc(C2CCCCN2)nc1',  # fakebpy
            'OB(*)O',  # boronic acids
            'O=C(*)N*',  # amides
            '[#6][OH]',  # free alcohol
            '[#6][NH2]',  # free amines
            '[#6]I',  # iodides
            '*P(O)(O)=O',  # phosphates
            '*C(O)=O',  # carboxylic acids
            '*C(Cl)=O',  # acid chlorides
            '*C#C*',  # alkynes
            '*C#C',  # terminal alkynes
            '[#6][NH][NH2]',  # N amines
            'c1ccsc1',  # sulfur heterocycle
            '[#6][SH]',  # free thiol
            'CN1CCOCCOCCOCCOCCOCC1',  # chelates
            'O=S=O',  # sulfones
            '*S*',  # general sulfur
            'c1(c2nc(C3[*]CCS3)ccc2)ncccc1',  # snn tri
            '*1:*c(-c2cccc(-c3ccccn3)n2)ncc1',  # nnn tri
            'S=C=N[*]',  # SCN
            '[N]=C=O',  # NCO
            '[O-]',  # O-
            '*[Se]*',  # selenium
            'C=C=C',  # allene
        ]
    ]
    # RDKit quirks
    forbidden[20].GetAtomWithIdx(1).SetNoImplicit(True)
    forbidden[21].GetAtomWithIdx(1).SetNoImplicit(True)
    forbidden[22].GetAtomWithIdx(1).SetNoImplicit(True)

    removed = 0

    for i, row in df.iterrows():
        if row['library_status'] == 'removed':
            continue

        smi = row['SMILES']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            append_reason(df, i, 'invalid SMILES')
            removed += 1
            continue

        for sub in forbidden:
            if mol.HasSubstructMatch(sub):
                append_reason(df, i, 'forbidden substructure')
                removed += 1
                break

    print(f"{removed} ligands had forbidden substructures and were marked for removal.")
    return df

def multiple_binding_sites_filter(df):
    """
    Flags molecules with more than one NN binding site on the ligand for removal.
    Modifies 'library_status' and 'status_notes' columns in-place.
    """
    forbidden_if_doubles = [
        Chem.MolFromSmarts(p) for p in [
            'c1(c2ncccc2)ccccn1',  # bpy
            'C1COC(C2=NCCO2)=N1',  # biox
            'C1COC(CC2=NCCO2)=N1',  # box
            'C1CNC(C2=NCCN2)=N1',  # bilm
            'c1ccc(C2=NCCO2)nc1',  # pyox
            'C1=CN=C(c2ccccn2)[N]1',  # pyNx
        ]
    ]

    removed = 0

    for i, row in df.iterrows():
        if row['library_status'] == 'removed':
            continue

        smi = row['SMILES']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            append_reason(df, i, 'invalid SMILES')
            removed += 1
            continue

        # Phase 1: remove if it matches multiple *unique* substructures
        found_subs = set()
        for sub in forbidden_if_doubles:
            if mol.HasSubstructMatch(sub):
                found_subs.add(sub)
            if len(found_subs) >= 2:
                append_reason(df, i, 'contains multiple possible binding sites')
                removed += 1
                break
        if df.at[i, 'library_status'] == 'removed':
            continue  # already flagged

        # Phase 2: remove if it has multiple copies of the same substructure
        for sub in forbidden_if_doubles:
            matches = mol.GetSubstructMatches(sub)
            if len(matches) > 1:
                append_reason(df, i, 'multiple binding site repeats')
                removed += 1
                break

    print(f"{removed} ligands had multiple binding sites and were marked for removal.")
    return df


def missing_bind_site_filter(df):
    """
    Flags molecules that don't have exactly one of the ligand substructures included in lib.
    Modifies 'library_status' and 'status_notes' columns in-place.
    """
    required = [
        Chem.MolFromSmarts(p) for p in [
            'c1(c2ncccc2)ccccn1',  # bpy
            'C1COC(C2=NCCO2)=N1',  # biox
            'C1COC(CC2=NCCO2)=N1',  # box
            'C1CNC(C2=NCCN2)=N1',  # bilm
            'c1cnc2c(c1)ccc1cccnc12',  # phen
            'c1ccc(C2=NCCO2)nc1',  # pyox
            'C1=CN=C(c2ccccn2)[N]1',  # pyNx
            'c1cnc2c(C3=NCCN3)cccc2c1'  # bnx
        ]
    ]

    removed = 0

    for i, row in df.iterrows():
        if row['library_status'] == 'removed':
            continue

        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            append_reason(df, i, 'invalid SMILES')
            removed += 1
            continue

        if not any(mol.HasSubstructMatch(p) for p in required):
            append_reason(df, i, 'missing required binding site')
            removed += 1

    print(f"{removed} ligands are missing a binding site and were marked for removal.")
    return df

def heterocycle_filter(df):
    """
    Primarily removes heterocycles that could provide other binding sites and thus behave unexpectedly.
    Modifies 'library_status' and 'status_notes' columns in-place.
    """
    dummy_het = [
        Chem.MolFromSmarts(p) for p in [
            '*n1nnn(*)c1=O', '*c1cn(*)nn1', '*c1ncn(*)n1', '*c1ncn(*)n1',
            '*n1cnc(=O)[nH]1', '*C1=NN=C(*)[*]1', '*n1cccn1', '*c1nc[nH]n1',
            '*c1nnn[nH]1', '*c1c[nH]cn1', 'c1cn[nH]c1', '*c1ncoc1*',
            'Cn1ccnn1', 'CC1=NN=C(C)O1', 'Cn1cnn(C)c1=O',
            'c1(c2nc(C3C[N]CO3)ccc2)nc(C4C[N]CO4)ccc1',
            'CCOc1nc(N2CCN(CC2)Cc3cnc(c4ncccc4)cc3)ncc1',
            '*NCc1cccc(-c2ccccn2)n1', 'C[#7]c1cccc(c2ncccc2)n1',
            'c1ccc(-c2ccc3c(n2)NCC3)nc1', '[*]Nc1nc(c2ncccc2)ccc1'
        ]
    ]

    removed = 0

    for i, row in df.iterrows():
        if row['library_status'] == 'removed':
            continue

        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            append_reason(df, i, 'invalid SMILES')
            removed += 1
            continue

        if any(mol.HasSubstructMatch(p) for p in dummy_het):
            append_reason(df, i, 'extra nitrogen heterocycles - possible multiple binding sites')
            removed += 1

    print(f"{removed} ligands contained problematic nitrogen heterocycles and were marked for removal.")
    return df

def isotopes_filter(df):
    """
    Flags molecules with isotopes so we don't have duplicates just from D/H.
    Modifies 'library_status' and 'status_notes' columns in-place.
    """
    removed = 0
    smiles_seen = set()

    for i, row in df.iterrows():
        if row['library_status'] == 'removed':
            continue

        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            append_reason(df, i, 'invalid SMILES')
            removed += 1
            continue

        for atom in mol.GetAtoms():
            if atom.GetIsotope():
                atom.SetIsotope(0)

        clean_smi = Chem.MolToSmiles(mol)

        if clean_smi in smiles_seen:
            append_reason(df, i, 'duplicate when considering isotopes')
            removed += 1
        else:
            smiles_seen.add(clean_smi)

    print(f"{removed} ligands were duplicates due to isotope encoding and were marked for removal.")
    return df

def is_metal(atom):
    """
    identify metal atoms
    """
    n = atom.GetAtomicNum()
    return (n == 5) or (21 <= n <= 34) or (37 <= n <= 52) or (n >= 54)

def other_metals_filter(df):
    """
    Flags molecules with any metal in them (pre Ni addition)
    Modifies 'library_status' and 'status_notes' columns in-place.
    """
    removed = 0

    for i, row in df.iterrows():
        if row['library_status'] == 'removed':
            continue

        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            append_reason(df, i, 'invalid SMILES')
            removed += 1
            continue

        if any(is_metal(atom) for atom in mol.GetAtoms()):
            append_reason(df, i, 'ligand contains metal outside bonding site')
            removed += 1

    print(f"{removed} ligands contained metals and were marked for removal.")
    return df