import os
import subprocess

lines = []

#obtain positive samples
traindbdir = "./data/char/"

print("processing "+traindbdir);
for filename in os.listdir(traindbdir):
  out = subprocess.check_output(["./extract_featuresNM1",traindbdir+filename])

  if ("Non-integer" in out):
		print "ERROR: Non-integer Euler number"

  else:
		if (out != ''):
			out = out.replace("\n","\nC,",out.count("\n")-1)
			lines.append("C,"+out)



#obtain negative samples
traindbdir = "./data/nonchar/"

print("processing "+traindbdir);
for filename in os.listdir(traindbdir):
  out = subprocess.check_output(["./extract_featuresNM1",traindbdir+filename])

  if ("Non-integer" in out):
		print "ERROR: Non-integer Euler number"

  else:
		if (out != ''):
			out = out.replace("\n","\nN,",out.count("\n")-1)
			lines.append("N,"+out)

print "Total: %d " % len(lines)

with open('char_datasetNM1.csv', 'w') as f:
  f.writelines(lines)

