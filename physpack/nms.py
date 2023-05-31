import argparse
import numpy as np
import cclib as cc
from cclib.parser.utils import PeriodicTable as pt
#Note: This uses cclib to read the frequencies from quantum chemistry calculations. 

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input out",  required=True)
required.add_argument("-o", "--output", type=str, help="Name of your output", required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("-l", "--linear", type=int, default=0, help="Use if your molecule is linear")
optional.add_argument("-ns", "--number_of_samples",type=int, default=1000, help="Number of samples to be generated")
optional.add_argument("-T", "--temperature", type=float, default=300, help="Temperature of the sampling")
args = parser.parse_args()

t = pt()
#calculate R

def R(natom,fct,T=300,linear=False):
    if linear:
        size=3*natom-5
    else:
        size=3*natom-6
    kB =1.38064852E-16 # This is in CGS system
    random_num = np.random.uniform(size=size)**2
    sign = [-1 if i < 0.5 else 1 for i in random_num]
    # Unit conversions
    fct_conv = [((i/1000)*1E8) for i in fct]
    R_vect = []
    for i,j in enumerate(fct_conv):
        R_i = sign[i]*np.sqrt((3*random_num[i]*kB*T)/j)
        R_vect.append(R_i/1e-8)

    R_vect = np.array(R_vect,dtype=object)
    return R_vect

def new_coord(natom,vib_disp,mass,fcts,T=300,linear=False):
    fix_fcts = [0.05 if i < 0.05 else i for i in fcts]
    Rx = R(natom, fix_fcts, T=T,linear=linear)
    mass_sqrt = [np.sqrt(1/i) for i in mass]
    new_disp = []
    for i,j in enumerate(vib_disp):
        disp_i = mass_sqrt[i]*Rx[i]*j
        new_disp.append(disp_i)
    disp = np.sum(new_disp,axis=0)
    return disp

def write_xyz(eq_coord,mass,vib_disp,fcts,elem_num,natom,out_name, T=300,linear=False):
    coord = eq_coord + new_coord(natom,vib_disp,mass,fcts,T=T,linear=linear)

    elem_name = [t.element[i] for i in elem_num]
    comment = "Molecule generated with normal mol sampling"
    atom_template = '{:3s} \t {:15.10f} \t {:15.10f} \t {:15.10f} '
    block = []
    block.append(natom)
    block.append(comment)
    for element, (x, y, z) in zip(elem_name, coord):
        block.append(atom_template.format(element, x, y, z))

    with open(out_name, 'w', newline='') as file:
        for item in block:
            file.write("%s\n" % item)



def main(T=300,n_samples=1000):
    print("input ", args.input)
    print("output", args.output)
    print("Linear molecule", 'No' if args.linear==0 else 'Yes')
    print("You are generating {} samples at {} K".format(n_samples,T))
    filename = args.input
    parser = cc.io.ccopen(filename)
    data = parser.parse()
    natom = data.natom
    eq_coord = data.atomcoords[-1]
    elem_num = data.atomnos
    fcts = data.vibfconsts
    mass = data.vibrmasses
    vib_disp = data.vibdisps

    for i in range(n_samples):
        out_name = '{}_{}.xyz'.format(args.output,i)
        write_xyz(eq_coord,mass,vib_disp,fcts,elem_num,natom,out_name,T=T,linear=args.linear)

main(T=args.temperature,n_samples=args.number_of_samples)
