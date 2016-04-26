import scipy.special

n = 10

for r in [0.,0.1,1.]:
    sph_in, sph_din = scipy.special.sph_in(n, r)
    print "r =", r

    for i in range(n):
        print "%2d %+1.7e %+1.7e" % (i, sph_in[i], sph_din[i])
