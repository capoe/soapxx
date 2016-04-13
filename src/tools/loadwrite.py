import os
import numpy as np
from .. import _soapxx as soap


try:
    import ase
    import ase.io
except ImportError:
    print("Note: ase.io import failed. Install PYTHON-ASE to harvest full reader functionality.")

def write_xyz(xyz_file, structure):
    ofs = open(xyz_file, 'w')
    ofs.write('%d\n%s\n' % (len(list(structure.particles)), structure.label))
    for p in structure.particles:
        ofs.write('%2s %+1.7e %+1.7e %+1.7e\n' % (p.name, p.pos[0], p.pos[1], p.pos[2]))
    ofs.close()
    return    

def ase_load_all(folder, log=None):
    cwd = os.getcwd()
    os.chdir(folder)
    config_listdir = sorted(os.listdir('./'))
    config_files = []
    for config_file in config_listdir:
        if os.path.isfile(config_file):
            config_files.append(config_file)
    ase_config_list = AseConfigList(config_files, log=log)
    os.chdir(cwd)
    return ase_config_list

def ase_load_single(config_file, log=None):
    ase_config_list = AseConfigList([config_file], log=log)
    return ase_config_list[0]

def setup_structure_ase(label, ase_config):
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
    # CREATE SINGLE SEGMENT
    segment = structure.addSegment()    
    # CREATE PARTICLES
    props = ['id', 'type', 'mass', 'pos']
    ids = [ i+1 for i in range(ase_config.get_number_of_atoms()) ]
    types = ase_config.get_atomic_numbers()
    positions = ase_config.get_positions()
    masses = ase_config.get_masses()
    names = ase_config.get_chemical_symbols()
    for id, name, typ, pos, mass in zip(ids, names, types, positions, masses):
        particle = structure.addParticle(segment)
        particle.pos = pos
        particle.mass = mass
        particle.weight = 1.
        particle.sigma = 0.5
        particle.name = name
        particle.type = name
        particle.type_id = typ
    return structure


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
        datastring=None):
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
        





