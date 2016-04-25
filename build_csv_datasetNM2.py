import os
import subprocess

lines = []

#obtain positive samples
traindbdir = "./data/char/"

print("processing "+traindbdir);
for filename in os.listdir(traindbdir):
  out = subprocess.check_output(["./extract_featuresNM2",traindbdir+filename])

  if ("Non-integer" in out):
		print "ERROR: Non-integer Euler number"

  else:
		if (out != ''):
            lines += ["C,"+o for o in out.split('\n')]



#obtain negative samples
traindbdir = "./data/nonchar/"

print("processing "+traindbdir);
for filename in os.listdir(traindbdir):
  out = subprocess.check_output(["./extract_featuresNM2",traindbdir+filename])

  if ("Non-integer" in out):
		print "ERROR: Non-integer Euler number"

  else:
		if (out != ''):
            lines += ["C,"+o for o in out.split('\n')]

print "Total: %d " % len(lines)

with open('char_datasetNM2.csv', 'w') as f:
  f.writelines(lines)

