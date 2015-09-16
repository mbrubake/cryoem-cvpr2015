#!/usr/bin/env python
usagestr = "Usage: %run -i run.py exp/1"

import sys
import os
if len(sys.argv) < 2:
	print usagestr
	raise Exception("No Experiment Directory Provided.")
expbase = sys.argv[1]
if not os.path.isdir(expbase):
	print usagestr
	raise Exception("Experiment Directory Does not Exist.")
kargs = {}
for a in sys.argv[2:]:
	key,val = a.split('=',1)
	kargs[key] = val

import optimizer

CO = optimizer.CryoOptimizer(expbase, kargs)

CO.run()

