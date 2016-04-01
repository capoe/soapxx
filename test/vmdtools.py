

class SystemVMD(object):
	def __init__(self):
		self.mols = []
		self.mols.append(MolVMD())
	def AddMol(self,
               coordfile):
		self.mols.append(MolVMD(coordfile, len(self.mols)))
		return
	def AddRep(self,
			   modselect = "all", 
	           modcolor = "name", 
	           modstyle = "lines",
	           modmaterial = "Opaque",
	           showperiodic="",
	           numperiodic=1):
		if len(self.mols[-1].reps): addrep = True
		else: addrep = False # Use default rep
		ID = len(self.mols[-1].reps)
		self.mols[-1].reps.append(RepVMD(ID, 
		                        modselect, 
		                        modcolor, 
		                        modstyle,
		                        modmaterial, 
		                        addrep,
		                        showperiodic,
		                        numperiodic))
		return
	def ToTCL(self,outt):
		for mol in self.mols:
			mol.ToTCL(outt)
		return
	def SetMaterial(self, name, amb, spec, diff, shin, opac):
		tcl = ''
		tcl += 'material change ambient {mat} {amb:+1.3f}\n'.format(mat=name, amb=amb)
		tcl += 'material change specular {mat} {spec:+1.3f}\n'.format(mat=name, spec=spec)
		tcl += 'material change diffuse {mat} {diff:+1.3f}\n'.format(mat=name, diff=diff)
		tcl += 'material change shininess {mat} {shin:+1.3f}\n'.format(mat=name, shin=shin)
		tcl += 'material change opacity {mat} {opac:+1.3f}\n'.format(mat=name, opac=opac)
		return tcl
		
		

class MolVMD(object):
	def __init__(self, coordfile=None, ID=0):
		self.ID = ID
		self.coordfile = coordfile
		self.reps = []
		return
	def ToTCL(self, outt):
		if self.coordfile != None:
			outt.write('mol new %s\n' % self.coordfile)
		for rep in self.reps:
			rep.ToTCL(outt)


class RepVMD(object):
	def __init__(self, ID = 0, modselect = "all", 
	                           modcolor = "name", # ColorID 0 Posz
	                           modstyle = "lines", 
	                           modmaterial = "Opaque",
	                           addrep = True,
	                           showperiodic="", # x xyz xX nxX
	                           numperiodic=1):
		self.ID				= ID
		self.modselect		= modselect
		self.modcolor		= modcolor
		self.modstyle		= modstyle
		self.modmaterial 	= modmaterial
		self.add			= addrep
		self.numperiodic	= numperiodic
		self.showperiodic	= showperiodic
	def ToTCL(self, outt):
		if self.add:
			outt.write('mol addrep top\n')
		outt.write('mol modselect %d top %s\n' % (self.ID, self.modselect))
		outt.write('mol modcolor %d top %s\n' % (self.ID, self.modcolor))
		outt.write('mol modstyle %d top %s\n' % (self.ID, self.modstyle))
		outt.write('mol modmaterial %d top %s\n' % (self.ID, self.modmaterial))
		outt.write('mol numperiodic top %d %s\n' % (self.ID, self.numperiodic))
		if self.showperiodic != "":
			outt.write('mol showperiodic top %d %s\n' % (self.ID, self.showperiodic))
		return


class ColorsVMD(object):
	def __init__(self):
		self.colors = []
		self.names = [\
			'blue', 'red', 'gray', 'orange', 'yellow', 
			'tan', 'silver', 'green',        'pink',    # ... white ...
			'cyan', 'purple', 'lime', 'mauve', 'ochre', 
			'iceblue', 'black', 'yellow2', 'yellow3', 'green2',
			'green3', 'cyan2', 'cyan3', 'blue2', 'blue3', 
			'violet', 'violet2', 'magenta', 'magenta2', 'red2',
			'red3', 'orange2', 'orange3' ]
		self.idcs = [\
			0, 1, 2, 3, 4,
			5, 6, 7,    9,  # ... 8=white ...
			10, 11, 12, 13, 14,
			15, 16, 17, 18, 19,
			20, 21, 22, 23, 24,
			25, 26, 27, 28, 29,
			30, 31, 32]
		self.current_color = -1
	def SetGrayScale(self, modulo=None):
		if modulo == None: modulo = len(self.names)
		for i in range(len(self.names)):
			imod = i % modulo
			bwi = 0+(float(modulo-1)-float(imod))/(modulo-1)
			self.colors.append('%1.3f %1.3f %1.3f' % (bwi, bwi, bwi))
		return
	def PrintToStream(self, ofs):
		for name, col in zip(self.names, self.colors):
			ofs.write('color change rgb %s %s\n' % (name, col))
		return
	def NextColor(self):
		self.current_color += 1
		self.current_color = self.current_color % len(self.idcs)
		return self.idcs[self.current_color]
		
		
#color Name C blue
#color scale method bwr
#color scale midpoint 0.5
#color scale min 0
#color change blue 0 0 1

#RND = "/sw/linux/vmd-1.8.7/lib/vmd/tachyon_LINUXAMD64 -aasamples 12 %s -format TARGA -o %s.tga" % (OUT[:-4],OUT[:-4])

# mol modstyle 0 top isosurface <isovalue> <volumeidx> <show> <draw> <step>
# mol modstyle 0 top cpk <sphereradius> <bondradius>
