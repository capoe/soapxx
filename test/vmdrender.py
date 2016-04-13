#! /usr/bin/env python
from __pyosshell__ 	import *
from vmdtools import *

VMDEXE = '/home/cp605/packages/install_vmd/vmd/bin/vmd'
TCL = 'vmdsetup.tcl'
cube = 'density.expanded.cube'

# SETUP GRAPHICS
system = SystemVMD()
system.AddMol(cube)

system.AddRep(modselect = 'all', modcolor = 'Name', modstyle = 'cpk 1 0.8', modmaterial = 'Edgy')
for iso_val in [0.05, 0.15, 0.25, 0.45, 0.75]:
    system.AddRep(modselect = 'all', modcolor = 'Name', modstyle = 'Isosurface %1.1f 0 0 Solid' % iso_val, modmaterial = 'Transparent')
#system.AddRep(modselect = 'resname BPW CEN', modcolor = 'Name', modstyle = 'DynamicBonds 2.1 0.2', modmaterial = 'Edgy')
#system.AddMol(grofile)
#system.AddRep(modselect = 'resname BPY CEN', modcolor = 'Name', modstyle = 'Paperchain 0.1 16', modmaterial = 'Transparent')
#system.AddRep(modselect = 'resname BPY CEN', modcolor = 'Name', modstyle = 'DynamicBonds 2.1 0.2', modmaterial = 'Edgy')

ofs = open(TCL,'w')
# Write Reps
system.ToTCL(ofs)
# Write transformations
#ofs.write('rotate x by 90\n')
#ofs.write('rotate y by 115\n')
#ofs.write('scale to 0.018\n')
#ofs.write('translate to 0 0 0\n')
# Write display settings
#ofs.write('display projection perspective\n')
ofs.write('display rendermode GLSL\n')
ofs.write('axes location off\n')
ofs.write('color Name F cyan3\n')
ofs.write('color Name N iceblue\n')
# Render
#ofs.write('mol off 0\n')
#ofs.write('render Tachyon z75\n')
#ofs.write('mol on 0\n')
#ofs.write('mol off 1\n')
#ofs.write('render Tachyon f75\n')
#ofs.write('exit\n')
ofs.close()
os.system('%s -e %s' % (VMDEXE, TCL))
#os.system('/sw/linux/vmd-1.8.7/lib/vmd/tachyon_LINUXAMD64 -aasamples 12 %s -res %d %d -format TARGA -o %s.tga' % ('z75', res_x, res_y, 'z75'))
#os.system('/sw/linux/vmd-1.8.7/lib/vmd/tachyon_LINUXAMD64 -aasamples 12 %s -res %d %d -format TARGA -o %s.tga' % ('f75', res_x, res_y, 'f75'))
#os.system('convert z75.tga z75.png')
#os.system('convert f75.tga f75.png')
#os.system('rm z75.tga f75.tga')

sys.exit(0)


