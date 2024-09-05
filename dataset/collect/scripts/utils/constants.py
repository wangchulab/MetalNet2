hetero_atoms = {'N', 'O', 'S'}
main_chain_atom_names = {'CA', 'O', 'N', 'C', 'OXT'}

common_amino_acids = {
    'Ala', 'Cys', 'Asp', 'Glu', 'Phe',
    'Gly', 'His', 'Ile', 'Lys', 'Leu',
    'Met', 'Asn', 'Pro', 'Gln', 'Arg',
    'Ser', 'Thr', 'Val', 'Trp', 'Tyr',
}

# metal types in common metal-binding proteins (Na, K are excluded)
common_metal_elements = {
    'Zn', 'Ca', 'Mg', 'Mn', 'Fe',
    'Cu', 'Ni', 'Co'
}

metal_elements = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 
    'K', 'Ca', 'Sc', 'Ti', 'V', 
    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 
    'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 
    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
    'In', 'Sn', 'Cs', 'Ba', 'La', 
    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 
    'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 
    'Er', 'Tm', 'Yb', 'Lu', 'Hf', 
    'Ta', 'W', 'Re', 'Os', 'Ir', 
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 
    'Bi', 'Fr', 'Ra', 'Ac', 'Th', 
    'Pa', 'U', 'Np', 'Pu', 'Am', 
    'Cm', 'Bk', 'Cf', 'Es', 'Fm', 
    'Md', 'No', 'Lr'
}

# three-letter name in pdb
# metal compounds with large ligand, such as HEM (heme), CLA (chlorophy), are not included
# some other metal compounds, such as EMC, IUM, are not included 
common_metal_compound_names = {
    'SF4', 'FES', 'F3S'
}