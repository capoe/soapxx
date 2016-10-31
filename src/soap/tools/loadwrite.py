import os
import numpy as np
import itertools
import partition
from .. import _soapxx as soap
from ..soapy import momo
from ..soapy import elements

try:
    import ase
    import ase.io
except ImportError:
    print("Note: ase.io import failed. Install PYTHON-ASE to harvest full reader functionality.")

def structures_from_xyz(xyz_file, do_partition=True, add_fragment_com=True):
    # Read xyz via ASE
    ase_configs = ase.io.read(xyz_file, index=':')
    return structures_from_ase(
        ase_configs=ase_configs, 
        do_partition=do_partition,
        add_fragment_com=add_fragment_com)

# TODO def structures_from_ase in addition to structure_from_ase

def structure_from_ase(
        config, 
        do_partition=True, 
        add_fragment_com=True, 
        use_center_of_geom=False, 
        log=None):
    # NOTE Center of mass is computed without considering PBC => Requires unwrapped coordinates
    structure = None
    frag_bond_matrix = None
    atom_bond_matrix = None
    frag_labels = []
    atom_labels = []
    # System properties
    label = config.info['label']
    positions = config.get_positions()
    types = config.get_chemical_symbols()
    if log: log << log.back << "Reading '%s'" % label << log.flush
    # Simulation cell
    if config.pbc.all(): 
        box = np.array([config.cell[0], config.cell[1], config.cell[2]])
    elif not config.pbc.any(): 
        box = np.zeros((3,3))
    else: 
        raise NotImplementedError("<structures_from_xyz> Partial periodicity not implemented.")
    # Partition 
    if do_partition:
        # Partition => Frags, top, reordered positions
        frags, top = partition.PartitionStructure(types, positions, outfile_gro='%s.gro' % label)
        atms = [ atm for atm in itertools.chain(*frags) ]
        positions = [ atm.xyz for atm in itertools.chain(*frags) ]
        types = [ atm.e for atm in itertools.chain(*frags) ]
        # Generate fragment labels
        for frag in frags:
            label = '-'.join(atm.e for atm in frag)
            frag_labels.append(label)
        # Reorder map
        id_reorder_map = {}
        for atm in atms:
            id_reorder_map[atm.id_initial] = atm.id
        # Connectivity matrix: fragments
        frag_bond_matrix = np.zeros((len(frags),len(frags)), dtype=bool)
        for i in range(len(frags)):
            frag_bond_matrix[i,i] = True
            for j in range(i+1, len(frags)):
                frags_are_bonded = False
                for ai in frags[i]:
                    for aj in frags[j]:
                        if aj in ai.bonded:
                            frags_are_bonded = True
                            break
                    if frags_are_bonded: break
                frag_bond_matrix[i,j] = frags_are_bonded
                frag_bond_matrix[j,i] = frags_are_bonded
        # Connectivity matrix: atoms
        atom_bond_matrix = np.zeros((len(positions),len(positions)), dtype=bool)
        for i in range(len(atms)):
            atom_bond_matrix[i,i] = True
            ai = atms[i]
            for j in range(i+1, len(atms)):
                atoms_are_bonded = False
                aj = atms[j]
                if aj in ai.bonded:
                    atoms_are_bonded = True
                atom_bond_matrix[i,j] = atoms_are_bonded
                atom_bond_matrix[j,i] = atoms_are_bonded
        # Reorder ASE atoms
        config = ase.Atoms(
            sorted(config, key = lambda atm: id_reorder_map[atm.index+1]), 
            info=config.info, 
            cell=config.cell, 
            pbc=config.pbc)
    else:
        top = [('SEG', 1, len(positions))]
    # Check particle count consistent with top
    atom_count = 0
    for section in top:
        atom_count += section[1]*section[2]
    assert atom_count == len(positions) # Does topology match structure?
    # Create segments, particles
    structure = soap.Structure(label)
    structure.box = box
    atom_idx = 0
    for section in top:
        seg_type = section[0]
        n_segs = section[1]
        n_atoms = section[2]
        for i in range(n_segs):
            segment = structure.addSegment()
            segment.name = seg_type
            # Add particles, compute CoM
            com = np.array([0.,0.,0.])
            com_weight_total = 0
            for j in range(n_atoms):
                particle = structure.addParticle(segment)
                particle.pos = positions[atom_idx]
                particle.weight = 1.
                particle.sigma = 0.5
                particle.type = types[atom_idx]
                atom_labels.append(particle.type)
                # Compute CoMass/CoGeom
                if use_center_of_geom:
                    com_weight = 1.
                else:
                    com_weight = elements.periodic_table[particle.type].mass
                com_weight_total += com_weight
                com = com + com_weight*positions[atom_idx]
                atom_idx += 1
            com = com/com_weight_total
            # Add CoM particle if requested
            if add_fragment_com:
                segment = structure.addSegment()
                segment.name = "%s.COM" % seg_type
                particle = structure.addParticle(segment)
                particle.pos = com
                particle.weight = 0.
                particle.sigma = 0.5
                particle.type = "COM"
    if log: log << log.endl
    return config, structure, top, frag_bond_matrix, atom_bond_matrix, frag_labels, atom_labels

def write_xyz(xyz_file, structure):
    ofs = open(xyz_file, 'w')
    ofs.write('%d\n%s\n' % (len(list(structure.particles)), structure.label))
    for p in structure.particles:
        ofs.write('%2s %+1.7e %+1.7e %+1.7e\n' % (p.name, p.pos[0], p.pos[1], p.pos[2]))
    ofs.close()
    return    

def ase_load_all(folder, log=None, n=None):
    cwd = os.getcwd()
    os.chdir(folder)
    config_listdir = sorted(os.listdir('./'))
    config_files = []
    for config_file in config_listdir:
        if os.path.isfile(config_file):
            config_files.append(config_file)
    if n: config_files = config_files[0:n]
    ase_config_list = AseConfigList(config_files, log=log)
    os.chdir(cwd)
    return ase_config_list

def ase_load_single(config_file, log=None):
    ase_config_list = AseConfigList([config_file], log=log)
    return ase_config_list[0]

def setup_structure_ase(label, ase_config, top=None):
    # DEFINE SYSTEM
    structure = soap.Structure(label)    
    # DEFINE BOUNDARY
    box = np.array([ase_config.cell[0], ase_config.cell[1], ase_config.cell[2]])
    if ase_config.pbc[0] == ase_config.pbc[1] == ase_config.pbc[2] == True:
        pass
    elif ase_config.pbc[0] == ase_config.pbc[1] == ase_config.pbc[2] == False:
        box = np.zeros((3,3))
    else:
        raise NotImplementedError("<setup_structure_ase> Partial periodicity not implemented.")
    structure.box = box    
    # PARTICLE PROPERTIES
    props = ['id', 'type', 'mass', 'pos']
    ids = [ i+1 for i in range(ase_config.get_number_of_atoms()) ]
    types = ase_config.get_atomic_numbers()
    positions = ase_config.get_positions()
    masses = ase_config.get_masses()
    names = ase_config.get_chemical_symbols()
    # DEFAULT TOPOLOGY
    if top == None:
        top = [('segment', 1, len(positions))]
    # VERIFY PARTICLE COUNT
    atom_count = 0
    for section in top:
        atom_count += section[1]*section[2]
    assert atom_count == len(positions) # Does topology match structure?
    # PARTITION: CREATE SEGMENTS, PARTICLES
    atom_idx = 0
    for section in top:
        seg_type = section[0]
        n_segs = section[1]
        n_atoms = section[2]
        for i in range(n_segs):
            segment = structure.addSegment()
            segment.name = seg_type
            segment.type = seg_type
            for j in range(n_atoms):
                particle = structure.addParticle(segment)
                particle.pos = positions[atom_idx]
                particle.mass = masses[atom_idx]
                particle.weight = 1.
                particle.sigma = 0.5
                particle.name = names[atom_idx]
                particle.type = names[atom_idx]
                particle.type_id = types[atom_idx]
                atom_idx += 1
    return structure 

def setup_structure_density(structure, radius_map, charge_map, anonymize=False):
    for particle in structure:
        particle.weight = charge_map[particle.type]
        particle.sigma = radius_map[particle.type]
        if anonymize:
            particle.type = "X"
    return

class AseConfigList(object):
    def __init__(self,
        config_files=[],
        log=None,
        grep_key='',
        read_fct=None,
        read_fct_args={}):
        if read_fct == None:
            import ase.io
            read_fct = ase.io.read
            read_fct_args = { 'index':':' }        
        self.configs = []
        config_idx = -1
        if log: log << "Reading configurations ..." << log.endl        
        for config_file in config_files:
            # Parse coordinate file
            try:
                if log: log << log.back << "..." << config_file << log.flush
                ase_configs = read_fct(config_file, **read_fct_args)
                if log: datastring = log >> log.catch >> 'cat %s | grep %s' % (config_file, grep_key)
                else: datastring = ''
            except KeyError:
                if log: log << endl << log.mr << "... Error when reading %s" % config_file << log.endl
                continue
            # Log & store configuration
            frame_idx = -1
            for ase_config in ase_configs:
                frame_idx += 1
                config_idx += 1
                config = AseConfig(ase_config=ase_config, 
                    config_idx=config_idx,
                    frame_idx=frame_idx,
                    config_file=config_file,
                    datastring=datastring)
                self.configs.append(config)
        if log: log << log.endl
    def __getitem__(self, idx):
        return self.configs[idx]
    def __iter__(self):
        return self.configs.__iter__()
    def __len__(self):
        return len(self.configs)

class AseConfig(object):
    def __init__(self,
        ase_config=None,
        config_idx=None, 
        frame_idx=None, 
        config_file=None,
        datastring=''):
        # COORDINATES
        self.atoms = ase_config
        # BOOK-KEEPING
        self.config_idx = config_idx
        self.frame_idx = frame_idx
        self.config_file = config_file
        # DATA [E.G., FROM 2nd LINE IN XYZ]
        self.datastring = datastring
        self.data = datastring.split()
        # Z-STATISTICS
        self.z_count = {}
        for z in self.atoms.get_atomic_numbers():
            try:
                self.z_count[z] += 1
            except KeyError:
                self.z_count[z] = 1
        return
    def InfoString(self):
        z_string = 'N= %-2d ' % len(self.config.z)
        for z in Z_ELEMENT_LIST:
            z_string += 'Z[%d]= %-2d ' % (z, self.z_count[z])
        return "'%s'   Frame= %-4d Idx= %-6d %-40s" % (\
            self.config_file, self.frame_idx, 
            self.config_idx, z_string)
    def InfoStringVerbose(self):
        appendix = ''
        for d in self.data:
            appendix += '%-12s' % d
        return '%s RG= %+1.7e %s' % (self.InfoString(), self.RadiusOfGyration(), appendix)
    def GetData(self, idx, to_type=float):
        return to_type(self.data[idx])
    def RadiusOfGyration(self):
        radius_g2 = 0.
        com = self.atoms.get_center_of_mass()
        for pos in self.atoms.positions:
            dr = pos-com
            r2 = np.dot(dr,dr)
            radius_g2 += r2
        radius_g2 = radius_g2/len(self.atoms.positions)
        return radius_g2**0.5
        





