# -*- coding: utf-8 -*-
## import libraries
import warnings
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from deepchem.feat.graph_features import *
from rdkit.Chem import AllChem
import deepchem as dc
from rdkit import Chem, DataStructs
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from collections import OrderedDict
import matplotlib.pyplot as plt
from cycler import cycler
import argparse
import sys
import tensorflow as tf
from pathlib import Path
import logging

def configure_logging():
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Prints logs to console
            logging.FileHandler("logfile.log")  # Saves logs to a file
        ]
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict Chemical properties from a SMILES string.")
    parser.add_argument("smiles", type=str, help="Enter Valid SMILES string of the compound")
    #parser.add_argument("help" )
    return parser.parse_args()


def predict_properties(x_smiles):
    mol = Chem.MolFromSmiles(x_smiles)

    if mol is None:
        print("Invalid SMILES string.")
        return

    # N = 10000# number of molecules in the dataset
    N = 133885
    D = 75     # hidden dimension of each atom
    E = 6      # dimension of each edge
    T = 3      # number of time steps the message phase will run for
    P = 32     # dimensions of the output from the readout phase, the penultimate output before the target layer
    V = 12     # dimensions of the molecular targets or tasks


    path_weights = 'weights.pth'


    chemical_accuracy_dict = {'mu': [0.1],
                              'alpha': [0.1],
                              'homo': [0.043],
                              'lumo': [0.043],
                              'gap': [0.043],
                              'r2': [1.2],
                              'zpve': [0.0012],
                              'u0': [0.043],
                              'u298': [0.043],
                              'h298': [0.043],
                              'g298': [0.043],
                              'cv': [0.50]}

    chemical_accuracy = pd.DataFrame(chemical_accuracy_dict)

    chemical_accuracy

    structures = ['smiles']



    class MasterEdge(nn.Module):

        def __init__(self):
            super(MasterEdge, self).__init__()

            self.l1 = nn.Linear(D, P)
            nn.init.kaiming_normal_(self.l1.weight)
            self.l2 = nn.Linear(P, 2*E)
            nn.init.kaiming_normal_(self.l2.weight)
            self.l3 = nn.Linear(2*E, E)
            nn.init.kaiming_normal_(self.l3.weight)

        def forward(self, x):
            return F.elu(self.l3(F.elu(self.l2(F.elu(self.l1(x))))))


    master_edge_learner = MasterEdge()

    # def dfs(adjacency_matrix,visited_array,i):
    #   visited_array[i] = 1
    #   for j in range(len(visited_array)):
    #     if(!visited_array[j] && adjacency_matrix[i][j]==1):
    #       dfs(adjacency_matrix,visited_matrix,j)

    # printing all cycles in an undirected graph
    # Function to mark the vertex with different colors for different cycles


    def dfs_cycle(u, p, color, mark, par, cyclenumber, g):
        # already (completely) visited vertex.
        if(color[u] == 2):
            return
        # seen vertex, but was not completely visited -> cycle detected.
        # backtrack based on parents to find the complete cycle.
        if(color[u] == 1):
            cyclenumber = cyclenumber + 1
            cur = p
            mark[cur] = cyclenumber
            # backtrack the vertex which are
            # in the current cycle thats found
            while(cur != u):
                # print(cur,u)
                cur = par[cur]
                mark[cur] = cyclenumber
            return

        par[u] = p
        # partially visited.
        color[u] = 1
        # simple dfs on graph
        for j in range(len(g[u])):
            if(g[u][j][1] == par[u]):
                continue
            dfs_cycle(g[u][j][1], u, color, mark, par, cyclenumber, g)
        # completely visited
        color[u] = 2

    # function to print all cycles


    def cycles_list_function(edges_1, mark, cycle_number, cycles):
        # push the edges that into the cycle adjacency list
        for i in range(edges_1):
            if(mark[i] != 0):
                # print(mark[i],i)
                cycles[mark[i]].append(i)
        # for i in range(cycle_number+1):
            # print(cycles[i])


    def construct_multigraph(smile):
        g = OrderedDict({})
        h = OrderedDict({})
        #h[-1] = 0
        molecule = Chem.MolFromSmiles(smile)
        #mol_matrix = [['0','1','0','0','0'],['1','0','2','0','1'],['0','2','0','1','1'],['0','0','1','0','0'],['0','1','1','0','0']]
        for i in range(molecule.GetNumAtoms()):
            atom_i = molecule.GetAtomWithIdx(i)
            atom_i_featurized = dc.feat.graph_features.atom_features(atom_i)
            atom_i_tensorized = torch.FloatTensor(atom_i_featurized).view(1, D)
            h[i] = atom_i_tensorized
            #h[-1] += h[i]
            master_edge = master_edge_learner(h[i])
            g.setdefault(i, [])
            # .append((master_edge, -1))
            #g.setdefault(-1, [])
            # .append((master_edge, i))
            for j in range(molecule.GetNumAtoms()):
                bond_ij = molecule.GetBondBetweenAtoms(i, j)
                if bond_ij:  # bond_ij is None when there is no bond.
                    #atom_j = molecule.GetAtomWithIdx(j)
                    #atom_j_featurized = dc.feat.graph_features.atom_features(atom_j)
                    #atom_j_tensorized = torch.FloatTensor(atom_j_featurized).view(1, 75)
                    bond_ij_featurized = dc.feat.graph_features.bond_features(
                        bond_ij).astype(int)
                    bond_ij_tensorized = torch.FloatTensor(
                        bond_ij_featurized).view(1, E)
                    g.setdefault(i, []).append((bond_ij_tensorized, j))
        # novelty
        edges = molecule.GetNumBonds()
        mark = [0]*molecule.GetNumAtoms()
        par = [-2]*molecule.GetNumAtoms()
        color = [0]*molecule.GetNumAtoms()
        cyclenumber = 0
        cycles_list = [[], [], []]
        dfs_cycle(0, -1, color, mark, par, cyclenumber, g)
        cycles_list_function(molecule.GetNumAtoms(), mark, max(mark), cycles_list)
        # print(max(mark))
        num_of_atoms = molecule.GetNumBonds()
        for i in range(len(cycles_list)):
            if(len(cycles_list[i]) >= 3):
                h[len(h)] = 0
                for j in range(len(cycles_list[i])):
                    # print(len(h)-1)
                    h[len(h)-1] += h[cycles_list[i][j]]  # semi master node
                    master_edge = master_edge_learner(h[cycles_list[i][j]])
                    g[cycles_list[i][j]].append((master_edge, len(h)-1))
                    g.setdefault(
                        len(h)-1, []).append((master_edge, cycles_list[i][j]))
        h[-1] = 0
        # print(cycles_list)
        for x in range(num_of_atoms, len(h)-1):
            # print(len(h))
            h[-1] += h[x]
            master_edge = master_edge_learner(h[x])
            g[x].append((master_edge, -1))
            g.setdefault(-1, []).append((master_edge, x))
        return g, h


    class EdgeMappingNeuralNetwork(nn.Module):

        def __init__(self):
            super(EdgeMappingNeuralNetwork, self).__init__()

            self.fc1 = nn.Linear(E, D)
            nn.init.kaiming_normal_(self.fc1.weight)
            self.fc2 = nn.Linear(1, D)
            nn.init.kaiming_normal_(self.fc2.weight)

        def f1(self, x):
            return F.elu(self.fc1(x))

        def f2(self, x):
            return F.elu(self.fc2(x.permute(1, 0)))

        def forward(self, x):
            return self.f2(self.f1(x))


    class MessagePhase(nn.Module):

        def __init__(self):
            super(MessagePhase, self).__init__()
            self.A = EdgeMappingNeuralNetwork()
            self.U = {i: nn.GRUCell(D, D) for i in range(T)}

        def forward(self, smile):

            g, h = construct_multigraph(smile)
            g0, h0 = construct_multigraph(smile)

            for k in range(T):
                h = OrderedDict(
                    {
                        v:
                        self.U[k](
                            sum(torch.matmul(h[w], self.A(e_vw))
                                for e_vw, w in en),
                            h[v]
                        )
                        for v, en in g.items()
                    }
                )

            return h, h0


    class Readout(nn.Module):

        def __init__(self):
            super(Readout, self).__init__()

            self.i1 = nn.Linear(2*D, 2*P)
            nn.init.kaiming_normal_(self.i1.weight)
            self.i2 = nn.Linear(2*P, P)
            nn.init.kaiming_normal_(self.i2.weight)

            self.j1 = nn.Linear(D, P)
            nn.init.kaiming_normal_(self.j1.weight)

        def i(self, h_v, h0_v):
            return F.elu(self.i2(F.elu(self.i1(torch.cat([h_v, h0_v], dim=1)))))

        def j(self, h_v):
            return F.elu(self.j1(h_v))

        def r(self, h, h0):
            return sum(torch.sigmoid(self.i(h[v], h0[v])) * self.j(h[v]) for v in h.keys())

        def forward(self, h, h0):
            return self.r(h, h0)


    class MPNN(nn.Module):
        def __init__(self):
            super(MPNN, self).__init__()

            self.M = MessagePhase()
            self.R = Readout()

            self.p1 = nn.Linear(P, P)
            nn.init.kaiming_normal_(self.p1.weight)
            self.p2 = nn.Linear(P, P)
            nn.init.kaiming_normal_(self.p2.weight)
            self.p3 = nn.Linear(P, V)
            nn.init.kaiming_normal_(self.p3.weight)

        def p(self, ro):
            return F.elu(self.p3(F.elu(self.p2(F.elu(self.p1(ro))))))

        def forward(self, smile):
            h, h0 = self.M(smile)
            embed = self.R(h, h0)
            return self.p(embed)


    model = MPNN()
    model.load_state_dict(torch.load('weights.pth'))
    model.eval()

    y = model(x_smiles)
    y_list = y.tolist()
    tasks = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
             'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    
    compound_properties = {}
    for i,a in enumerate(tasks):

        compound_properties[a] = y_list[0][i]
        
    print("Compound Properties:")
    print(f"   Norm of the dipole moment (mu): {compound_properties['mu']}")
    print(f"   Norm of the static polarizability (alpha): {compound_properties['alpha']}")
    print(f"   HOMO Energy (homo): {compound_properties['homo']}")
    print(f"   LUMO Energy (lumo): {compound_properties['lumo']}")
    print(f"   Electron energy Gap (gap): {compound_properties['gap']}")
    print(f"   Electronic Spatial Extent (r2): {compound_properties['r2']}")
    print(f"   Zero-Point Vibrational Energy (zpve): {compound_properties['zpve']}")
    print(f"   Atomizatio Energy at 0K (u0): {compound_properties['u0']}")
    print(f"   Atomizatio Energy at 298K (u298): {compound_properties['u298']}")
    print(f"   Enthalpy of atomization at 298K (h298): {compound_properties['h298']}")
    print(f"   Free energy of atomization (g298): {compound_properties['g298']}")
    print(f"   Heat Capacity (cv): {compound_properties['cv']}")
        



def main():
    configure_logging()
    tf.get_logger().setLevel('ERROR')
    args = parse_arguments()
    smiles = str(args.smiles)

    predict_properties(smiles)

if __name__ == "__main__":
    main()
