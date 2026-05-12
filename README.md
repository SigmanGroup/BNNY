# BNNY

This repo contains scripts with examples for:
    1. adding a new ligand to the library
    2. filtering ligands to see if they meet criteria described in publication
    3. property collection
    4. reproducing models described in the paper

## Additional Information

### Adding New Ligands to the Library

Adding a ligand begins with a smiles string of the ligand only(here shown as a df of smiles strings). The smiles strings can be converted to an SDF and constrained with a metal using the notebook 'conformer_generation.ipynb'

Conformer searches are performed using the macrmodel scripts in the directory 'complex_and_free_lig_generation'. 


### Filtering Ligands

To evaluate if a ligand meets original library criteria the filtering found in 'ligand_filtering' can be applied.

### Property Collection

Properties can be collected utilizing the scripts found in 'property_collection'. Sample log files are included as an example.

### Modeling

The non-linear models described in the paper can be reproduced and found using the scripts in the 'modeling' folder

