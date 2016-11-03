#! /usr/bin/env python
from momo import sys, np, osio, endl, flush
try:
    from __qmshell__ import e_xyz_from_xyz
    from __molecules__ import Atom, Molecule
except ImportError:
    def e_xyz_from_xyz(xyzfile):
        raise ImportError("__qmshell__::e_xyz_from_xyz not available")
    class Atom(object):
        def __init__(self, *args, **kwargs):
            raise ImportError("__molecules__::Atom not available")
    class Molecule(object):
        def __init__(self, *args, **kwargs):
            raise ImportError("__molecules__::Molecule not available")

# TODO Atoms docked to a core atom are all moved to the same ligand group 
#      This is not appropriate for branched side chains

#osio.Connect()
#osio.AddArg('file',         typ=str,             default=None,             help='Input xyz-file')
#osio.AddArg('molname',         typ=str,             default='UNSCRAMBLED',     help='Molecule name')
#osio.AddArg('gro',             typ=str,             default='',             help='Output gro-file')
#osio.AddArg('xyz',             typ=str,             default='',             help='Output xyz-file')
#osio.AddArg('ring_exclude', typ=(list, str),     default=['Al','Zn'],     help='Exclude these atoms from ring structure')
#opts = osio.Parse()

xyzfile             = "in.xyz" #opts.file
molname             = "heypretty" #opts.molname
outfile_xyz         = '' #opts.xyz
outfile_gro         = '' #opts.gro
exclude_bonds_to    = [] #opts.ring_exclude
wordy               = False
leaky               = False
path_cutoff_length  = 7
ligand_size_lower   = 4

def covalent_cutoff(rad1, rad2):
    return 1.15*(rad1+rad2)

def calculate_connectivity_mat(distance_mat, type_vec, cutoff=None):
    dim = distance_mat.shape[0]
    connectivity_mat = np.zeros((dim,dim), dtype=bool)
    if cutoff != None:
        for i in range(dim):
            for j in range(dim):
                if distance_mat[i,j] < constant:
                    connectivity_mat[i,j] = True
                else:
                    connectivity_mat[i,j] = False
    else:
        for i in range(dim):
            for j in range(dim):
                t1 = type_vec[i]
                t2 = type_vec[j]
                r1 = COVRAD_TABLE[t1]
                r2 = COVRAD_TABLE[t2]
                r12 = distance_mat[i,j]
                if r12 <= covalent_cutoff(r1, r2):
                    connectivity_mat[i,j] = True
                else:
                    connectivity_mat[i,j] = False
    return connectivity_mat

# COVALENCE RADII (from Cambridge Structural Database, table see http://en.wikipedia.org/wiki/Covalent_radius)
COVRAD_TABLE = {}
COVRAD_TABLE['H'] = 0.31
COVRAD_TABLE['C'] = 0.76
COVRAD_TABLE['N'] = 0.71
COVRAD_TABLE['F'] = 0.57
COVRAD_TABLE['O'] = 0.66
#COVRAD_TABLE['Se'] = 1.20
COVRAD_TABLE['S'] = 1.05
#COVRAD_TABLE['Zn'] = 1.22
COVRAD_TABLE['Br'] = 1.20
COVRAD_TABLE['Cl'] = 1.02
COVRAD_TABLE['I'] = 1.39
COVRAD_TABLE['P'] = 1.07

# FORCEFIELD TYPE TABLE
TYPE_TABLE = {\
'C:CCH'     : 'CA',        # Aromatic
'C:CCN'     : 'CA',        # Aromatic + Nitrogen (TODO)
'C:CHS'     : 'CA',        #
'C:CCSe'    : 'CB',        # Aromatic + Selenium (TODO)
'C:CCS'     : 'CB',        # Aromatic + Sulphur
'C:CCO'     : 'CO',        # Aromatic + Carboxylic
'C:CNSe'    : 'CS',        # Aromatic + Selenium + Nitrogen (TODO)
'C:CCHH'    : 'CR',        # Aliphatic
'C:CHHN'    : 'CR',        # Aliphatic
'C:CHHH'    : 'CR',        # Methyl
'C:CCC'     : 'CC',        # 
'C:CN'      : 'CN',        # Cyano-group (TODO)
'H:C'       : 'HC',        # 
'N:CCC'     : 'NA',        # Aromatic
'N:C'       : 'NC',        # 
'O:C'       : 'OC',        # Carboxylic group
'S:CC'      : 'S',         # Thiophene sulphur
'Se:CC'     : 'Se'}        #

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
NUMBERS = '1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

class AtomXyz(object):
    def __init__(self, e, xyz, id):
        # PROPERTIES
        self.e = e
        self.id_initial = id
        self.id = id
        self.xyz = xyz
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]
        self.covrad = COVRAD_TABLE[self.e]
        if self.e in exclude_bonds_to: self.covrad = 0.0
        # FRAGMENT INFO
        self.name = ''
        self.type = ''
        self.fragname = '___'
        self.fragid = 0
        # BONDING LEVEL (-1 = CORE)
        self.level = -1
        # ALL BONDS
        self.bonded = []
        self.bonded_short = []
        # CORE BONDS
        self.bonded_core = []
        self.bonds_core = []
        # NON-RING CORE BONDS
        self.bonded_non_ring = []
        self.bonds_non_ring = []
        # DOCKED        
        self.docked_to = []
        self.dock_for = []
        # PATHS OF SOME LENGTH
        self.path_length = -1
        self.paths = []
        # NON-RING PATHS OF SOME LENGTH
        self.path_length_non_ring = -1
        self.paths_non_ring = []
    def info_str(self):
        return "%d:%s:%s" % (self.id, self.name, self.fragname)
    def generate_type(self):
        type_key = ''
        bonded_elems = []
        for b in self.bonded:
            bonded_elems.append(b.e)
        bonded_elems.sort()    
        for e in bonded_elems:
            type_key += e
        type_key = self.e + ':' + type_key
        try:
            self.type = TYPE_TABLE[type_key]
        except KeyError:
            default_type = self.e+'X'
            #print "Type definition missing for '%s', defaulting to '%s'" % (type_key, default_type)
            self.type = default_type
        return
    def get_all_docked_atoms(self):
        docked_atoms = []
        docked_atoms = docked_atoms + self.dock_for
        for d in self.dock_for:
            docked_atoms = docked_atoms + d.get_all_docked_atoms()
        return docked_atoms
    def add_bond_core(self, bonded_atom, bond):
        self.bonded_core.append(bonded_atom)
        self.bonds_core.append(bond)
        return
    def add_bond_non_ring(self, bonded_atom, bond):
        self.bonded_non_ring.append(bonded_atom)
        self.bonds_non_ring.append(bond)
        return
    def find_paths(self, length, exclusion_list=[], start_here=True):
        if length == 0: return []
        paths = []
        #exclusion_list.append(self)
        for bond in self.bonds_core:
            new_path = Path(start_with_atom=self if start_here else None)
            if not bond.b in exclusion_list:
                new_path.extend(bond)
                paths.append(new_path)
                other = bond.b
                #other_paths = other.find_paths(length=length-1, exclusion_list=exclusion_list, start_here=False)
                other_paths = other.find_paths(length=length-1, exclusion_list=[self], start_here=False)
                for o in other_paths:
                    joined = JoinPaths(new_path, o)
                    paths.append(joined)
        if start_here:
            self.path_length = length
            self.paths = paths
        return paths
    def find_paths_non_ring(self, length=4, exclusion_list=[], start_here=True):
        if length == 0: return []
        paths = []
        #exclusion_list.append(self)
        for bond in self.bonds_non_ring:
            new_path = Path(start_with_atom=self if start_here else None)
            if not bond.b in exclusion_list:
                new_path.extend(bond)
                paths.append(new_path)
                other = bond.b
                #other_paths = other.find_paths(length=length-1, exclusion_list=exclusion_list, start_here=False)
                other_paths = other.find_paths_non_ring(length=length-1, exclusion_list=[self], start_here=False)
                for o in other_paths:
                    joined = JoinPaths(new_path, o)
                    paths.append(joined)
        if start_here:
            self.path_length_non_ring = length
            self.paths_non_ring = paths
        return paths            
    
class Path(object):
    def __init__(self, start_with_atom=None):
        self.visited = []
        if start_with_atom != None: self.visited.append(start_with_atom)
        self.bonds = []
    def info_str(self):
        info_str = ''
        for v in self.visited:
            info_str += v.info_str()+' '
        return info_str
    def add_visited(self, v):
        self.visited.append(v)
    def add_bond(self, b):
        self.bonds.append(b)
    def extend(self, bond):
        self.add_visited(bond.b)
        self.add_bond(bond)
    def print_info(self):
        for v in self.visited:
            print "[%d:%s]" % (v.id,v.e),
        print ""
    def get_first(self):
        return self.visited[0]
    def get_last(self):
        return self.visited[-1]        
        
class BondXyz(object):
    def __init__(self, atom1, atom2):
        self.a = atom1
        self.b = atom2        
        
def JoinPaths(path1, path2):
    joined_path = Path()
    joined_path.visited = path1.visited + path2.visited
    joined_path.bonds = path1.bonds + path2.bonds
    return joined_path    

class Ring(object):
    def __init__(self, first_pair=None):
        self.atoms = []
        self.bonded_structures = []
        if first_pair != None:
            self.atoms.append(first_pair[0])
            self.atoms.append(first_pair[1])
    def type(self):
        return "ring"
    def has_atom(self, atom):
        return self.check_atom(atom)
    def check_atom(self, atom):
        if atom in self.atoms: return True
        else: return False
    def check_add_pair(self, pair):
        has_a = self.check_atom(pair[0])
        has_b = self.check_atom(pair[1])
        if has_a and not has_b:
            self.atoms.append(pair[1])
            return True
        elif has_b and not has_a:
            self.atoms.append(pair[0])
            return True
        elif has_b and has_a:
            return True
        else:
            return False
    def print_info(self):
        for a in self.atoms:
            print "%2d" % a.id,
        print ""
    def intersects(self, other):
        intersects = False
        for a in self.atoms:
            if a in other.atoms:
                intersects = True
                break
        return intersects
    def add(self, other):
        for atom in other.atoms:
            if not atom in self.atoms:
                self.atoms.append(atom)
        return
    def find_round_trip_path(self, visited=[], start_atm=None):
        started_here = False
        if start_atm == None:            
            self.atoms = sorted(self.atoms, key=lambda atm: len(atm.bonded))
            start_atm = self.atoms[0]
            visited.append(start_atm)
            started_here = True
        for bond in start_atm.bonds_core:
            # Backup visited list to be able to revert to this point
            visited_0 = []
            for v in visited: visited_0.append(v)
            if bond.b in visited or not bond.b in self.atoms: continue
            # Partner atom not yet visited, proceed            
            visited.append(bond.b)
            visited = self.find_round_trip_path(visited, bond.b)
            # All atoms visited = round path?
            if len(visited) == len(self.atoms):
                break
            # A dead end. Revert & try next bond
            else:
                visited = []
                for v in visited_0: visited.append(v)
        #assert len(visited) <= len(self.atoms)
        if started_here:
            if len(visited) != len(self.atoms):
                if wordy: osio << osio.mr << "WARNING: Ring - Failed to generate round-trip path (atom order may be compromised)" << endl
                visited = self.atoms
            else:
                if wordy:
                    osio << "Found round-trip path" << endl
        return visited
    def order_atoms(self):
        self.atoms = self.find_round_trip_path(visited=[], start_atm=None)
        return
    def get_all_bonded_structures(self, exclude_list=[]):
        bonded = []
        exclude_list.append(self)
        for b in self.bonded_structures:
            if b in exclude_list:
                continue
            bonded.append(b)
            bonded = bonded + b.get_all_bonded_structures(exclude_list)
        return bonded

def JoinRings(ring1, ring2):
    joined_ring = Ring()
    for a in ring1.atoms:
        joined_ring.atoms.append(a)
    for b in ring2.atoms:
        if not joined_ring.has_atom(b):
            joined_ring.atoms.append(b)
    return    

class Chain(object):
    def __init__(self, first_pair=None):
        self.atoms = []
        self.bonded_structures = []
        if first_pair != None:
            self.atoms.append(first_pair[0])
            self.atoms.append(first_pair[1])
    def type(self):
        return "chain"
    def has_atom(self, atom):
        return self.check_atom(atom)
    def intersects(self, other):
        intersects = False
        for a in self.atoms:
            if a in other.atoms:
                intersects = True
                break
        return intersects
    def add(self, other):
        for atom in other.atoms:
            if not atom in self.atoms:
                self.atoms.append(atom)
        return
    def print_info(self):
        for a in self.atoms:
            print "%2d" % a.id,
        print ""
    def find_round_trip_path(self, visited=[], start_atm=None):
        started_here = False
        # Figure out end atoms
        ranks = []
        for atm in self.atoms:
            n_bonds_in_core_chain = 0
            for bond in atm.bonds_core:
                if bond.b in self.atoms:
                    n_bonds_in_core_chain += 1
            ranks.append(n_bonds_in_core_chain)
        if start_atm == None:            
            self.atoms = sorted(self.atoms, key=lambda atm: ranks[self.atoms.index(atm)])
            start_atm = self.atoms[0]
            visited.append(start_atm)
            started_here = True
        for bond in start_atm.bonds_core:
            # Backup visited list to be able to revert to this point
            visited_0 = []
            for v in visited: visited_0.append(v)
            if bond.b in visited or not bond.b in self.atoms: continue
            # Partner atom not yet visited, proceed            
            visited.append(bond.b)
            visited = self.find_round_trip_path(visited, bond.b)
            # All atoms visited = round path?
            if len(visited) == len(self.atoms):
                break
            # A dead end. Revert & try next bond
            else:
                visited = []
                for v in visited_0: visited.append(v)
        #assert len(visited) <= len(self.atoms)
        if started_here:
            if len(visited) != len(self.atoms):
                if wordy: osio << osio.mr << "WARNING: Chain - Failed to generate round-trip path (expected for branched core units, atom order may be compromised)" << endl
                #visited = self.atoms
                for atm in self.atoms:
                    if not atm in visited:
                        visited.append(atm)
            else:
                if wordy:
                    osio << "Found round-trip path" << endl
        return visited
    def order_atoms(self):
        self.atoms = self.find_round_trip_path(visited=[], start_atm=None)
        return
    def get_all_bonded_structures(self, exclude_list=[]):
        bonded = []
        exclude_list.append(self)
        for b in self.bonded_structures:
            if b in exclude_list:
                continue
            bonded.append(b)
            bonded = bonded + b.get_all_bonded_structures(exclude_list)
        return bonded

def CreateMolecule(name, atoms, xyz_conv_fact=0.1):
    molecule = Molecule(0, name)
    for atom in atoms:
        gro_atom = Atom(ln='')
        gro_atom.fragId        = atom.fragid
        gro_atom.fragName    = atom.fragname
        gro_atom.name        = atom.name
        gro_atom.Id            = atom.id
        gro_atom.pos        = np.array(atom.xyz)*xyz_conv_fact
        gro_atom.vel        = np.array([0.,0.,0.])
        molecule.atoms.append(gro_atom)
    return molecule

def PartitionStructure(e=[], xyz=[], xyzfile=None, generate_bonded=False, verbose=False, outfile_xyz='', outfile_gro=''):
    if xyzfile:
        e,xyz = e_xyz_from_xyz(xyzfile)
    # LOAD ATOMS
    atoms = []
    count = 0
    for e,r in zip(e,xyz):
        count += 1
        atoms.append(AtomXyz(e,r,count))

    # ESTABLISH BONDING VIA COVALENCE CRITERION
    if verbose: osio << osio.mg << "Find bonds using covalence criterion" << endl
    bond_count = 0
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            a = atoms[i]
            b = atoms[j]
            dr = np.dot(a.xyz-b.xyz,a.xyz-b.xyz)**0.5
            dv = covalent_cutoff(a.covrad,b.covrad)        
            if dr < dv:
                bond_count += 1
                a.bonded.append(b)
                b.bonded.append(a)
    if verbose: osio << "%d bonds in molecule" % bond_count << endl
    if leaky:
        for a in atoms:
            print "%2s bonded to %d" % (a.e, len(a.bonded))

    for a in atoms:
        if len(a.bonded) == 0:
            osio << osio.my << "NOTE: Unbonded atom" << a.e << a.id << endl

    # SEQUENTIALLY SPLIT OFF LIGAND UNITS
    if verbose: osio << osio.mg << "Find core using sequential reduction" << endl
    short_list = []
    for a in atoms:
        short_list.append(a)

    selection_levels = []
    this_level = 0

    while True:
        if leaky:
            print "Level", this_level
            print "Short-listed", len(short_list)
        for a in short_list:
            a.bonded_short = []
        for i in range(len(short_list)):
            for j in range(i+1, len(short_list)):
                a = short_list[i]
                b = short_list[j]
                dr = np.dot(a.xyz-b.xyz,a.xyz-b.xyz)**0.5
                dv = covalent_cutoff(a.covrad,b.covrad)        
                if dr < dv:
                    a.bonded_short.append(b)
                    b.bonded_short.append(a)    
        rm_atoms = []
        for s in short_list:
            if len(s.bonded_short) == 1:
                rm_atoms.append(s)    
        if len(rm_atoms) == 0:
            break
        if leaky:
            print "Removing", len(rm_atoms)
        for r in rm_atoms:
            r.level = this_level
            for b in r.bonded_short:
                b.dock_for.append(r)
                r.docked_to.append(b)
            short_list.remove(r)
        if leaky:
            ofs = open('level_%d.xyz' % this_level, 'w')
            ofs.write('%d\n\n' % len(short_list))
            for s in short_list:
                ofs.write('%s %+1.7f %+1.7f %+1.7f\n' % (s.e, s.x, s.y, s.z))
            ofs.close()    
        this_level += 1

    # READ OFF CORE ATOMS
    core = []
    for a in atoms:
        assert len(a.docked_to) <= 1    
        docked_atoms = a.get_all_docked_atoms()
        if wordy:
            osio << "%-2s bonded to %d, docked to %d, dock for %d/%-2d at level %+d" \
                % (a.e, len(a.bonded), len(a.docked_to), len(a.dock_for), len(docked_atoms), a.level) << endl
        if len(a.docked_to) < 1:
            core.append(a)
    if verbose: osio << "%d atoms in core" % len(core) << endl

    # ESTABLISH BONDING AMONG CORE ATOMS
    if verbose: osio << osio.mg << "Find core-atom bonds using covalence criterion" << endl
    bonds = []
    for i in range(len(core)):
        for j in range(i+1, len(core)):
            a = core[i]
            b = core[j]
            dr = np.dot(a.xyz-b.xyz,a.xyz-b.xyz)**0.5
            dv = covalent_cutoff(a.covrad,b.covrad)
            if dr < dv:
                a.bonded_short.append(b)
                b.bonded_short.append(a)
                bond_ab = BondXyz(a,b)
                bond_ba = BondXyz(b,a)
                a.add_bond_core(b, bond_ab)
                b.add_bond_core(a, bond_ba)
                bonds.append(bond_ab)
    if verbose: osio << "%d bonds in core" %  len(bonds) << endl

    # GENERATE PATHS ALONG CORE BONDS
    if verbose: osio << osio.mg << "Find connecting paths (max. length %d)" % path_cutoff_length << endl
    path_count = 0
    for c in core:
        paths = c.find_paths(length=path_cutoff_length, exclusion_list=[], start_here=True)
        if wordy:
            osio << "%2d paths of length <= %d from atom %2d" % (len(paths), path_cutoff_length, c.id) << endl
        if leaky:
            for p in paths:
                p.print_info()
        path_count += len(paths)
    if verbose: osio << "Generated a total of %d bond paths" % path_count << endl

    # FROM PATHS FIND RING-CONNECTED ATOMS
    ring_pairs = []
    for i in range(len(core)):
        for j in range(i+1, len(core)):
            a = core[i]
            b = core[j]
            
            paths_ab = []
            for p in a.paths:
                if p.get_last() == b:
                    paths_ab.append(p)
            
            paths_ba = []
            for p in b.paths:
                if p.get_last() == a:
                    paths_ba.append(p)
            
            if leaky:
                print "ID1 %d ID2 %d" % (a.id, b.id)
                print "a => b: %d" % len(paths_ab)
                #for p in paths_ab:
                #    p.print_info()
                print "b => a: %d" % len(paths_ba)
                #for p in paths_ba:
                #    p.print_info()
            
            assert len(paths_ab) == len(paths_ba)        
            if len(paths_ab) == 1: continue
            
            has_disjoint_paths = False
            for k in range(len(paths_ab)):
                for l in range(k+1, len(paths_ab)):
                    intersects = False
                    p1 = paths_ab[k]
                    p2 = paths_ab[l]
                    b1 = p1.bonds
                    b2 = p2.bonds
                    
                    for bond in b1:
                        if bond in b2:
                            intersects = True
                    if not intersects:
                        has_disjoint_paths = True
                    
            if has_disjoint_paths:
                pair = [a,b]
                ring_pairs.append(pair)
                if leaky:
                    osio << osio.mg << "Ring pair:" << a.id-1 << b.id-1 << endl

    # FROM RING PAIRS, FIND RINGS VIA SUCCESSIVE ADDITION
    if verbose: osio << osio.mg << "Find rings using set of ring pairs" << endl
    rings = []

    for pair in ring_pairs:
        new_ring = Ring(first_pair=pair)
        rings.append(new_ring)

    i = 0
    while i <= len(rings)-1:
        ring = rings[i]
        rm_rings = []
        for j in range(i+1, len(rings)):
            other = rings[j]
            if ring.intersects(other):
                rm_rings.append(other)
                ring.add(other)
        for r in rm_rings:
            rings.remove(r)
        i += 1
        
    if verbose: osio << "Core rings (# = %d)" % len(rings) << endl
    if wordy:
        for r in rings:
            r.print_info()

    # READ OFF NON-RING ATOMS
    non_ring_core_atoms = []
    for c in core:
        in_ring = False
        for r in rings:
            if r.has_atom(c):
                in_ring = True
        if not in_ring:
            non_ring_core_atoms.append(c)

    if verbose: osio << "Non-ring core atoms: %d" % len(non_ring_core_atoms) << endl

    # ESTABLISH BONDING AMONG NON-RING CORE ATOMS
    if verbose: osio << osio.mg << "Find non-ring core-atom bonds using covalence criterion" << endl
    bonds = []
    for i in range(len(non_ring_core_atoms)):
        for j in range(i+1, len(non_ring_core_atoms)):
            a = non_ring_core_atoms[i]
            b = non_ring_core_atoms[j]
            dr = np.dot(a.xyz-b.xyz,a.xyz-b.xyz)**0.5
            dv = covalent_cutoff(a.covrad,b.covrad)
            if dr < dv:
                a.bonded_short.append(b)
                b.bonded_short.append(a)
                bond_ab = BondXyz(a,b)
                bond_ba = BondXyz(b,a)
                a.add_bond_non_ring(b, bond_ab)
                b.add_bond_non_ring(a, bond_ba)
                bonds.append(bond_ab)
    if verbose: osio << "%d bonds in non-ring core" %  len(bonds) << endl

    # GENERATE PATHS ALONG NON-RING CORE BONDS
    if verbose: osio << osio.mg << "Find connecting non-ring paths (max. length %d)" % path_cutoff_length << endl
    path_count = 0
    for c in non_ring_core_atoms:
        paths = c.find_paths_non_ring(length=path_cutoff_length, exclusion_list=[], start_here=True)
        if wordy:
            print "%2d paths of length <= %d from atom %2d" % (len(paths), path_cutoff_length, c.id)
        if leaky:
            for p in paths:
                p.print_info()
        path_count += len(paths)
    if verbose: osio << "Generated a total of %d non-ring bond paths" % path_count << endl

    # FROM PATHS FIND NON-RING-CONNECTED ATOMS
    non_ring_pairs = []
    for i in range(len(non_ring_core_atoms)):
        for j in range(i+1, len(non_ring_core_atoms)):
            a = non_ring_core_atoms[i]
            b = non_ring_core_atoms[j]
            
            paths_ab = []
            for p in a.paths_non_ring:
                if p.get_last() == b:
                    paths_ab.append(p)
            
            paths_ba = []
            for p in b.paths_non_ring:
                if p.get_last() == a:
                    paths_ba.append(p)
            
            if leaky:
                print "ID1 %d ID2 %d" % (a.id, b.id)
                print "a => b: %d" % len(paths_ab)
                #for p in paths_ab:
                #    p.print_info()
                print "b => a: %d" % len(paths_ba)
                #for p in paths_ba:
                #    p.print_info()
            
            assert len(paths_ab) == len(paths_ba)        
            assert len(paths_ab) <= 1
            
            if len(paths_ab) > 0:
                pair = [a,b]
                non_ring_pairs.append(pair)
                if leaky:
                    osio << osio.mg << "Non-ring pair:" << a.id-1 << b.id-1 << endl

    # FROM NON-RING PAIRS, FIND NON-RINGS (= CHAINS) VIA SUCCESSIVE ADDITION
    if verbose: osio << osio.mg << "Find non-ring structures using set of non-ring pairs" << endl
    chains = []

    # ADD NON-RING CHAINS WITH > 1 CORE ATOM
    for pair in non_ring_pairs:
        new_chain = Chain(first_pair=pair)
        chains.append(new_chain)

    # ADD NON-RING CHAINS WITH ONLY 1 CORE ATOM (HENCE WITHOUT CHAIN PATHS)
    for a in non_ring_core_atoms:
        if a.paths_non_ring == []:
            new_chain = Chain()
            new_chain.atoms.append(a)
            chains.append(new_chain)

    i = 0
    while i <= len(chains)-1:
        chain = chains[i]
        rm_chains = []
        for j in range(i+1, len(chains)):
            other = chains[j]
            if chain.intersects(other):
                rm_chains.append(other)
                chain.add(other)
        for r in rm_chains:
            chains.remove(r)
        i += 1

    if len(chains) == 0:
        for atom in non_ring_core_atoms:
            new_chain = Chain()
            new_chain.atoms.append(atom)
            chains.append(new_chain)

    if verbose: osio << "Core chains (# = %d)" % len(chains) << endl
    if wordy:
        for c in chains:
            c.print_info()

    # REORDER STRUCTURAL ELEMENTS (CORE RINGS & CORE CHAINS)
    molecule = []
    structures = rings + chains
    for i in range(len(structures)):
        for j in range(i+1, len(structures)):
            s1 = structures[i]
            s2 = structures[j]
            bond_count = 0
            for a in s1.atoms:
                for b in s2.atoms:
                    dr = np.dot(a.xyz-b.xyz,a.xyz-b.xyz)**0.5
                    dv = covalent_cutoff(a.covrad,b.covrad)
                    if dr < dv:
                        bond_count += 1
            assert bond_count <= 1
            if bond_count:
                s1.bonded_structures.append(s2)
                s2.bonded_structures.append(s1)

    start_struct_idx = 0
    if len(structures) == 1:
        pass
    else:
        structures = sorted(structures, key=lambda s: len(s.bonded_structures))
        while structures[start_struct_idx].bonded_structures == []:
            molecule.append(structures[start_struct_idx])
            if start_struct_idx+1 == len(structures): break
            start_struct_idx += 1
    start_struct = structures[start_struct_idx]

    docked_structures = start_struct.get_all_bonded_structures(exclude_list=[])
    molecule =  molecule + [start_struct] + docked_structures

    # REORDER ATOMS IN EACH STRUCTURE
    for struct in molecule:
        if verbose: osio << "Structure type %-10s" % ("'%s'" % struct.type()) << "(bonded to %d)" % len(struct.bonded_structures) << endl
        struct.order_atoms()

    # ASSIGN SMALL CORE STRUCTURES TO NEIGHBOUR STRUCTURES
    rm_structs = []
    for struct in molecule:
        if len(struct.atoms) < 2:
            rm_atoms = []
            for atom in struct.atoms:
                if atom.bonded_core != []:
                    rm_atoms.append(atom)
                    molecule[molecule.index(struct)-1].atoms.append(atom)
            for r in rm_atoms:
                struct.atoms.remove(r)
            if len(struct.atoms) == 0:
                rm_structs.append(struct)
    for r in rm_structs:
        molecule.remove(r)

    # GENERATE ATOM TYPES
    if verbose: osio << osio.mg << "Assign atom types" << endl
    frag_atom_type_count = {}
    for atm in atoms:
        atm.generate_type()
        frag_atom_type_count[atm.type] = 0

    # SORT ATOMS AND ASSIGN FRAGMENT NAMES & IDs
    if verbose: osio << osio.mg << "Sort atoms, assign fragment names & IDs" << endl
    atoms_ordered = []
    frag_count = 0
    core_count = 0
    ligand_count = 0
    core_alphabet_index = 0
    ligand_alphabet_index = 1
    atoms_top = []
    top = [] # entries of the form (name, repeats, n_atoms_in_frag)
    for struct in molecule:
        # Core atoms
        frag_count += 1
        core_count += 1
        ligand_sets = []
        # Reset fragment atom-type counter
        for key in frag_atom_type_count.keys():
            frag_atom_type_count[key] = 0
        if verbose: osio << "Core   '%s' (size at core level: %2d)" % (ALPHABET[core_count-1], len(struct.atoms)) << endl
        # Topology entry
        atoms_top.append([])
        top.append(['CO'+ALPHABET[core_count-1],1,0])
        for atm in struct.atoms:
            top[-1][2] += 1
            atoms_top[-1].append(atm)
            atm.fragid = frag_count
            atm.fragname = 'CO' + ALPHABET[core_count-1]        
            atm.name = atm.type + NUMBERS[frag_atom_type_count[atm.type]]
            frag_atom_type_count[atm.type] += 1
            atoms_ordered.append(atm)        
            docked = atm.get_all_docked_atoms()
            if len(docked) <= ligand_size_lower:
                for datm in docked:
                    top[-1][2] += 1
                    atoms_top[-1].append(datm)
                    datm.fragid = frag_count
                    datm.fragname = 'CO' + ALPHABET[core_count-1]
                    datm.name = datm.type + NUMBERS[frag_atom_type_count[datm.type]]
                    atoms_ordered.append(datm)
            else:
                ligand_sets.append(docked)    
        # Reset fragment atom-type counter
        for key in frag_atom_type_count.keys():
            frag_atom_type_count[key] = 0
        # Ligand atoms
        ligand_count_this_core = 0
        for lset in ligand_sets:
            frag_count += 1
            ligand_count += 1
            ligand_count_this_core += 1
            # Topology entry
            top.append(['L%s%d' % (ALPHABET[core_count-1], ligand_count_this_core),1,0])
            atoms_top.append([])
            if verbose: osio  << "Ligand '%s' (size at core level: %2d)" % (ALPHABET[core_count-1], len(lset)) << endl        
            for atm in lset:
                top[-1][2] += 1
                atoms_top[-1].append(atm)
                atm.fragid = frag_count
                atm.fragname = 'L%s%d' % (ALPHABET[core_count-1], ligand_count_this_core)
                atm.name = atm.type + NUMBERS[frag_atom_type_count[atm.type]]
                frag_atom_type_count[atm.type] += 1
                atoms_ordered.append(atm)
    
    # FIX ATOM IDs        
    atom_count = 0
    for atom in atoms_ordered:
        atom_count += 1
        atom.id = atom_count

    # DEFINE INTERACTIONS VIA BONDED PATHS
    # ... Paths of length 2 => bonds, paths of length 3 => angles
    # ... Atoms with three bonded partners => star impropers
    # ... Remove dihedrals across ring units (terminal atoms connected by a 3-path and (n>3)-path)
    if generate_bonded:
        for atom in atoms_ordered:
            paths = atom.find_paths(3, [], True)
            if verbose:
                for p in paths:
                    osio << p.info_str() << endl
        
    # OUTPUT XYZ
    if outfile_xyz != '':
        ofs = open(outfile_xyz, 'w')
        ofs.write('%d\n\n' % len(atoms_ordered))        
        for atm in atoms_ordered:
            if wordy:
                print atm.e, atm.fragname, atm.fragid, atm.type, atm.name
            ofs.write('%-2s %+1.7f %+1.7f %+1.7f\n' % (atm.e, atm.x, atm.y, atm.z))
        ofs.close()

    # OUTPUT GRO
    if outfile_gro != '':
        molecule = CreateMolecule(molname, atoms_ordered)
        molecule.write_gro(outfile_gro)

    return atoms_top, top

        
        
        
        







