# BNNY

This should contain information about the repo including environment installation and usage.

## Major edits

1. Add README.md
2. Clear entrypoint to scripts (what do I run to get the models? what about parameterization)
3. In the add_ni_edit_bond_order.py, there are multiple main guards (if __name__ == "__main__") when there should be one
4. complex_generation/ has no data to run the script on
5. Remove duplicate code between new_ligand_filtering.ipynb and conformer_generation.ipynb
6. Add environment file. I could not run the code with my modeling env (will require edits later)
7. I can't run NN_get_props_hydride.ipynb because of no data and broken paths