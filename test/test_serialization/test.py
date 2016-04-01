#! /usr/bin/env python
import soap
import soap.tools

import os
import numpy as np
from momo import osio, endl, flush

archfile = 'config_000057.xyz.spectrum.arch'

spectrum = soap.Spectrum(archfile)
spectrum.writeDensityOnGrid(1, "C", "")
spectrum.save("%s-2" % archfile)
osio.okquit()


