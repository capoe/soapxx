#! /usr/bin/env python
from momo import osio, endl, flush, os, sys

l = sys.argv[1]

vmd_exe = '/usr/local/bin/vmd'
setup_tcl = 'vmdsetup.tcl'
cubefile = 'recon.id-7_center-C_type-C_l-%s.cube' % l

vmdsetup_tcl = '''
mol new {cubefile:s}
mol modselect 0 top all
mol modcolor 0 top Name
mol modstyle 0 top cpk 1 0.8
mol modmaterial 0 top Edgy
mol numperiodic top 0 1
mol addrep top
mol modselect 1 top all
mol modcolor 1 top ColorId 0
mol modstyle 1 top Isosurface -0.01 0 0 Solid
mol modmaterial 1 top Transparent
mol numperiodic top 1 1
mol addrep top
mol modselect 2 top all
mol modcolor 2 top ColorId 1
mol modstyle 2 top Isosurface 0.01 0 0 Solid
mol modmaterial 2 top Transparent
mol numperiodic top 2 1
display rendermode GLSL
axes location off
color Name F cyan3
color Name N iceblue
'''.format(cubefile=cubefile)

ofs = open(setup_tcl, 'w')
ofs.write(vmdsetup_tcl)
ofs.close()

os.system('%s -e %s' % (vmd_exe, setup_tcl))
