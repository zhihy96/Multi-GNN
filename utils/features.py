import numpy as np
from rdkit import Chem
import torch
from torch_geometric.data import Data

# Static information
# For organic chemicals only symbol list was restricted to the following:
SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'H', 'UNK']

hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
hydrogen_acceptor = Chem.MolFromSmarts("[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
basic = Chem.MolFromSmarts("[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")


possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
possible_number_radical_e_list = [0, 1, 2]
possible_chirality_list = ['R', 'S']
 
reference_lists = [SYMBOLS, possible_numH_list, possible_valence_list,
    possible_formal_charge_list, possible_number_radical_e_list,
    possible_hybridization_list, possible_chirality_list
]

 

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))
 

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
 

 

def safe_index(l, e):
    """Gets the index of e in l, providing an index of len(l) if not found"""
    try:
        return l.index(e)
    except:
        return len(l)
 




    
def atom_features(atom,mol,explicit_H=False,use_chirality=False,use_hydrogen_bonding=False,
                  use_acid_base = False, num_atoms = False):
    """ collect all the atom features """
    ring = mol.GetRingInfo()
    atom_idx = atom.GetIdx()  
    results  = one_of_k_encoding_unk(atom.GetSymbol(), SYMBOLS) 
    results += one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6])
    results += one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, 
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, 
                Chem.rdchem.HybridizationType.SP3D, 
                Chem.rdchem.HybridizationType.SP3D2])

    results += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
    results += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    results += [ring.IsAtomInRingOfSize(atom_idx, 3),
                ring.IsAtomInRingOfSize(atom_idx, 4),
                ring.IsAtomInRingOfSize(atom_idx, 5),
                ring.IsAtomInRingOfSize(atom_idx, 6),
                ring.IsAtomInRingOfSize(atom_idx, 7),
                ring.IsAtomInRingOfSize(atom_idx, 8)]

    results += [atom.GetIsAromatic()]

    if not explicit_H:
        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    if use_chirality:
        try:
            results += one_of_k_encoding_unk(atom.GetProp('_CIPCode'),['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results += [False, False] + [atom.HasProp('_ChiralityPossible')]

    if use_hydrogen_bonding:
        hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
        results += [atom_idx in hydrogen_donor_match]
        results += [atom_idx in hydrogen_acceptor_match]
    if use_acid_base:
        acidic_match = sum(mol.GetSubstructMatches(acidic), ())
        basic_match = sum(mol.GetSubstructMatches(basic), ())
        results +=  [atom_idx in acidic_match]
        results += [atom_idx in basic_match]

    if num_atoms:
        results = results + [1./float(mol.GetNumAtoms())]
 
    return np.array(results)
   
  
def bond_features(bond, use_chirality=False):
    """ collects all the bond features """
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)


def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res


def mol2vec(mol, num_atoms=True, use_hydrogen_bonding=True, use_acid_base=True):
    """ Main function that is called to create the features for the dataloader """
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    
    
    # node features
    node_f= [atom_features(atom, mol, use_hydrogen_bonding=use_hydrogen_bonding,use_acid_base=use_acid_base,
                           num_atoms=num_atoms) for atom in atoms]
    # adjacency matrix
    edge_index = torch.tensor(get_bond_pair(mol), dtype=torch.long)

    # edge features, not used in the paper
    edge_attr = [bond_features(bond, use_chirality=False) for bond in bonds]

    for bond in bonds:
        edge_attr.append(bond_features(bond))
    data = Data(x=torch.tensor(node_f, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    return data


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes



