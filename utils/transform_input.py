#!/usr/bin/env python3

import crystal_interface as ci
import argparse
import numpy as np
import os
from ast import literal_eval

if __name__ == "__main__":
    
    #print("#####################################################")
    #print("##                                                 ##")
    #print("##        Running geometry mod-tool                ##")
    #print("""##        Use keyword "--help" for more info       ##""")
    #print("##                                                 ##")
    #print("#####################################################")
    #print("")
    parser = argparse.ArgumentParser(prog = "Transform input",
                                        description = "Transformations of input files for LSDalton",
                                        epilog = "Author: Audun Skau Hansen (e-mail: audunsh@student.matnat.uio.no)")
    parser.add_argument("source_file", type = str, help ="input file (ex. MOLECULE.INP)")
    parser.add_argument("output_file", type = str, help ="input file (ex. MOLECULE_OUT.INP)")
    parser.add_argument("-ra", nargs = 2, help = "Rotation of atoms (axis, theta) of geometry." )
     
    
    #parser.add_argument("-t", help="")
    args = parser.parse_args()
    
    wd = os.getcwd()
    
    ## Parse geometry
    f = open(wd + "/" + args.source_file, "r")
    F = f.read()
    f.close()
    
    begf = F.split("Charge")[0]
    endf = "a1 =" + F.split("a1 =")[1]
    
    
    atoms = F.split("Charge=")[1:]
    a_num = []
    for a in atoms:
        charge= literal_eval(a.split()[0])
        natoms = literal_eval(a.split("Atoms=")[1].split("\n")[0])

        a_num.append([charge,natoms, []])
        for atom in a.split("\n")[1:natoms+1]:
            segment = [atom.split()[0],np.array([literal_eval(i) for i in atom.split()[1:]])]

            a_num[-1][2].append(segment)
    #for a in a_num:
    #    print(a)

    
    if args.ra != None:
        c = np.cos(np.pi*literal_eval(args.ra[1]))
        s = np.sin(np.pi*literal_eval(args.ra[1]))
        
        if args.ra[0] == "1":
            rm = np.array([[1, 0,  0],
                           [0,  c,-s],
                           [0,  s, c]])
                           
        if args.ra[0] == "2":
            rm = np.array([[c,  0, s],
                           [0,  1, 0],
                           [-s, 0, c]]) 
                           
        if args.ra[0] == "3":
            rm = np.array([[c, -s, 0],
                           [s,  c, 0],
                           [0,  0, 1]])
        newgeom = []
        for charge in range(len(a_num)):
            newgeom.append([a_num[charge][0],a_num[charge][1], []])
            
            for atom in range(a_num[charge][1]):
                name = a_num[charge][2][atom][0]
                pos = np.array(a_num[charge][2][atom][1])
                newgeom[-1][2].append([name, 
                                       np.dot(rm, pos) ])
        a_num = newgeom

    geom_string = ""
    for a in range(len(a_num)):
        geom_string += "Charge=%i. Atoms=%i\n"  % (a_num[a][0] , a_num[a][1])
        for atom in range(a_num[a][1]):
            geom_string += "%s %.8e %.8e %.8e\n"  % (a_num[a][2][atom][0],
                                                   a_num[a][2][atom][1][0],
                                                   a_num[a][2][atom][1][1],
                                                   a_num[a][2][atom][1][2])

    
    print(begf+geom_string+endf)
    if args.output_file:
        f = open(args.output_file, "w")
        f.write(begf+geom_string+endf)
        f.close()

    
    
        