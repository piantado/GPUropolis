xs = []
ys = []
for l in open(data, 'r'):
	l = l.strip()
	if re.match("\s*#",l): continue
	
	x,y = map(float, re.split("\t", l))
	xs.append(x)
	ys.append(y)
xs = numpy.array(xs)
ys = numpy.array(ys)


#http://pythonhosted.org/pygp/demo_gpr.html