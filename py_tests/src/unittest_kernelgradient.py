#! /usr/bin/env python
import soap
import soap.tools

import os
import sys
import numpy as np
import logging
import io
import unittest

from momo import osio, endl, flush
from kernel import KernelPotential, TrajectoryLogger
from kernel import perturb_positions, random_positions
from kernel import evaluate_energy, evaluate_energy_gradient

# LOGGING/OUTPUT
soap.silence()
logging.basicConfig(
    format='[%(asctime)s] %(message)s', 
    datefmt='%I:%M:%S', 
    level=logging.ERROR)
verbose = False

class TestGlobalGeneric(unittest.TestCase):
    def setUp(self):
        self.setUp_Options()
        self.setUp_Structure()
        self.setUp_Kernel(self.options, self.structure)
    def setUp_Options(self):
        # CONSTRAINTS, EXCLUSIONS
        exclude_center_pids = []
        # EXPANSIONS
        options = soap.Options()
        options.excludeCenters([])
        options.excludeTargets([])
        options.excludeCenterIds(exclude_center_pids)
        options.excludeTargetIds([])
        options.set('radialbasis.type', 'gaussian')
        options.set('radialbasis.mode', 'adaptive')
        options.set('radialbasis.N', 9) # 9
        options.set('radialbasis.sigma', 0.5) # 0.9
        options.set('radialbasis.integration_steps', 15)
        options.set('radialcutoff.Rc', 6.8)
        options.set('radialcutoff.Rc_width', 0.5)
        options.set('radialcutoff.type', 'heaviside')
        options.set('radialcutoff.center_weight', 0.5)
        options.set('angularbasis.type', 'spherical-harmonic')
        options.set('angularbasis.L', 6) # 6
        options.set('densitygrid.N', 20)
        options.set('densitygrid.dx', 0.15)
        options.set('spectrum.gradients', True)
        options.set('spectrum.2l1_norm', False) # <- pull here
        # KERNEL OPTIONS
        options.set('kernel.adaptor', 'global-generic') # <- pull here
        options.set('kernel.type', 'dot')
        options.set('kernel.delta', 1.)
        options.set('kernel.xi', 4.) # <- pull here
        # STORE
        self.options = options
        return
    def setUp_Structure(self):
        data_root = os.path.join(os.environ['SOAP_ROOT'], 'py_tests/data')
        # STRUCTURE
        xyzfile = os.path.join(data_root, 'config_simple_L.xyz')
        config = soap.tools.ase_load_single(xyzfile)
        structure = soap.tools.setup_structure_ase(config.config_file, config.atoms)
        positions_0 = [ part.pos for part in structure ]
        # STORE
        self.structure = structure
        self.positions_0 = positions_0
        return
    def setUp_Kernel(self, options, structure):
        # SETUP KERNEL        
        kernelpot = KernelPotential(options)
        kernelpot.acquire(structure, 1.)
        # STORE
        self.kernelpot = kernelpot
        return

class TestGlobalGenericSetup(TestGlobalGeneric):
    def test_Structure(self):
        # GENERATE
        tmp = io.StringIO()
        tmp.write(u'Positions at start\n')
        for part in self.structure.particles:
            pos = part.pos
            tmp.write(u'%d %s %+1.7e %+1.7e %+1.7e\n' % (part.id, part.type, pos[0], pos[1], pos[2]))
        output_str = tmp.getvalue()
        tmp.close()
        # COMPARE
        output_ref_str = '''
        Positions at start
        1 C +0.0000000e+00 +0.0000000e+00 +0.0000000e+00
        2 C +1.5000000e+00 +0.0000000e+00 +0.0000000e+00
        3 C +0.0000000e+00 +1.5000000e+00 +0.0000000e+00
        '''
        self.assertEqual(output_str.split(), output_ref_str.split())
        return

class TestGlobalGenericKernelGradient(TestGlobalGeneric):
    def runTest(self):
        exclude_perturb_pids = []
        exclude_random_pids = []
        constrain_pids = []
        perturb_initial = True
        random_initial = True
        opt_pids = [3]
        dx = 0.05
        idx = 1
        # EXTRACT POSITIONS
        opt_pids = np.array(opt_pids)
        opt_pidcs = opt_pids-1
        positions = np.array(self.positions_0)
        positions_short = positions[opt_pidcs]
        positions_short = positions_short.flatten()
        # ABBREVIATIONS
        f = evaluate_energy
        x0 = positions_short
        fprime = evaluate_energy_gradient
        args = (self.structure, self.kernelpot, opt_pids)

        # GENERATE OUTPUT
        tmp = io.StringIO()
        for i in range(20):
            # Set positions
            xi = i*dx
            pi = np.array([0.,0.,0.])
            pi[idx] = xi            
            e = evaluate_energy(pi, *args)
            g = evaluate_energy_gradient(pi, *args)
            # pos[idx] - energy - grad[idx]
            tmp.write(u'%+1.7e %+1.7e %+1.7e\n' % (pi[idx], e, g[idx]))
        output_str = tmp.getvalue()
        output = np.fromstring(output_str, sep=' ')
        tmp.close()
        
        # COMPARE TO REFERENCE
        output_ref_str = '''\
        +0.0000000e+00 +5.2071483e-01 +6.5344824e-18
        +5.0000000e-02 +5.2209009e-01 +5.4944827e-02
        +1.0000000e-01 +5.2619625e-01 +1.0910396e-01
        +1.5000000e-01 +5.3297438e-01 +1.6169339e-01
        +2.0000000e-01 +5.4232650e-01 +2.1193466e-01
        +2.5000000e-01 +5.5411612e-01 +2.5907080e-01
        +3.0000000e-01 +5.6816986e-01 +3.0238666e-01
        +3.5000000e-01 +5.8428034e-01 +3.4124182e-01
        +4.0000000e-01 +6.0221079e-01 +3.7510832e-01
        +4.5000000e-01 +6.2170159e-01 +4.0360591e-01
        +5.0000000e-01 +6.4247850e-01 +4.2653661e-01
        +5.5000000e-01 +6.6426241e-01 +4.4390393e-01
        +6.0000000e-01 +6.8677959e-01 +4.5591911e-01
        +6.5000000e-01 +7.0977171e-01 +4.6298273e-01
        +7.0000000e-01 +7.3300443e-01 +4.6564341e-01
        +7.5000000e-01 +7.5627317e-01 +4.6453147e-01
        +8.0000000e-01 +7.7940521e-01 +4.6027457e-01
        +8.5000000e-01 +8.0225719e-01 +4.5340524e-01
        +9.0000000e-01 +8.2470817e-01 +4.4427347e-01
        +9.5000000e-01 +8.4664866e-01 +4.3297924e-01
        '''
        output_ref = np.fromstring(output_ref_str, sep= ' ')
        self.assertEqual(output_str.split(), output_ref_str.split())
        return

if __name__ == "__main__":
    unittest.main()






















