import time
import numpy as np
import soap
import scipy.special
np.random.seed(7)
log = soap.log

def assert_equal(z, target, eps=1e-5):
    if np.abs(z-target) > eps: raise ValueError(z)
    else: log << "+" << log.flush

def get_calc(scale):
    return soap.external.GylmCalculator(
        rcut=scale*3.5,
        rcut_width=scale*0.5,
        nmax=9,
        lmax=6,
        sigma=scale*0.5,
        part_sigma=scale*0.5,
        wconstant=True,
        wscale=scale*1.0,
        wcentre=scale*1.0,
        ldamp=4.,
        types="C,N,O,S,H,F,Cl,Br,I,B,P".split(","),
        #types="C,H".split(","),
        periodic=False)

def test_gylm_scaleinv():
    log << log.mg << "<test_scaleinv>" << log.endl
    calc0 = get_calc(scale=1.)
    configs = soap.tools.io.read('structures.xyz')
    for cidx, config in enumerate(configs):
        log << "Struct" << cidx << log.flush
        heavy = np.where(np.array(config.symbols) != "H")[0]
        x0 = calc0.evaluate(system=config, positions=config.positions[heavy], normalize=True)
        pos_orig = np.copy(config.positions)
        for scale in [ 0.5, 1.5, 2.5 ]:
            config.positions = scale*pos_orig
            calc1 = get_calc(scale=scale)
            x1 = calc1.evaluate(
                system=config, 
                positions=config.positions[heavy], 
                normalize=True)
            diff = np.max(np.abs(x0.dot(x0.T) - x1.dot(x1.T)))
            assert_equal(diff, 0.0, 1e-10)
        log << log.endl

def test_gylm_rotinv():
    log << log.mg << "<test_rotinv>" << log.endl
    gylm_calc = soap.external.GylmCalculator(
        rcut=4.0,
        rcut_width=0.5,
        nmax=9,
        lmax=6,
        sigma=0.5,
        types="C,N,O,S,H,F,Cl,Br,I,B,P".split(","),
        periodic=False)
    configs = soap.tools.io.read('structures.xyz')
    for cidx, config in enumerate(configs):
        log << "Struct" << cidx << log.flush
        heavy = np.where(np.array(config.symbols) != "H")[0]
        x0 = gylm_calc.evaluate(system=config, positions=config.positions[heavy])
        norm = 1./np.sum(x0**2, axis=1)**0.5
        x0 = (x0.T*norm).T
        for i in range(4):
            R = soap.tools.transformations.get_random_rotation_matrix()
            config.positions = config.positions.dot(R)
            x1 = gylm_calc.evaluate(system=config, positions=config.positions[heavy])
            norm = 1./np.sum(x1**2, axis=1)**0.5
            x1 = (x1.T*norm).T
            diff = np.max(np.abs(x0.dot(x0.T) - x1.dot(x1.T)))
            assert_equal(diff, 0.0, 1e-10)
        log << log.endl

def test_ylm(lmax=7):
    log << log.mg << "<test_ylm>" << log.endl
    xyz_orig = np.random.normal(0, 1., size=(3,))
    xyz = xyz_orig/np.dot(xyz_orig,xyz_orig)**0.5
    # Reference via scipy
    theta = np.arccos(xyz[2])
    phi = np.arctan2(xyz[1], xyz[0])
    if phi < 0.: phi += 2*np.pi
    ylm_ref = []
    ylm_ref_re = []
    for l in range(lmax+1):
        ylm_ref_l = []
        ylm_ref_re_l = []
        for m in range(-l, l+1):
            ylm_ref_l.append(scipy.special.sph_harm(m, l, phi, theta))
            ylm_ref_re_l.append(0)
        reco = 1./2**0.5
        imco = np.complex(0,1)/2**0.5
        ylm_ref_re_l[l] = ylm_ref_l[l].real
        for m in range(1, l+1):
            qlm = ylm_ref_l[l+m]
            ql_m = ylm_ref_l[l-m]
            s = (imco*(ql_m - (-1)**m*qlm)).real
            r = (reco*(ql_m + (-1)**m*qlm)).real
            ylm_ref_re_l[l-m] = s
            ylm_ref_re_l[l+m] = r
        ylm_ref.extend(ylm_ref_l)
        ylm_ref_re.extend(ylm_ref_re_l)
    ylm_ref = np.array(ylm_ref) 
    ylm_ref_re = np.array(ylm_ref_re)
    # soap::ylm
    S = 1000
    ylm_out = np.zeros((S*(lmax+1)**2,))
    xyz = np.tile(xyz_orig, (S,1))
    x = np.copy(xyz[:,0])
    y = np.copy(xyz[:,1])
    z = np.copy(xyz[:,2])
    t0 = time.time()
    soap.external.ylm(x, y, z, x.shape[0], lmax, ylm_out)
    t1 = time.time()
    log << "delta_t =" << t1-t0 << log.endl
    for j in np.random.randint(0, S, size=(3,)):
        ll = (lmax+1)**2
        dy = ylm_out[j*ll:(j+1)*ll] - ylm_ref_re
        for l in range(lmax+1):
            log << "l=%d" % l << log.flush
            for m in range(-l,l+1):
                lm = l**2+l+m
                assert_equal(np.abs(dy[lm]), 0.0, 1e-7)
            log << log.endl
        
if __name__ == "__main__":
    test_ylm()
    test_gylm_rotinv()
    test_gylm_scaleinv()

