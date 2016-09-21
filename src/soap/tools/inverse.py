import numpy as np
import numpy.linalg

from ..soapy import momo

osio = momo.osio
endl = momo.endl
flush = momo.flush

def compute_X(Qr):
    # Q is N x (2*L+1) real-valued matrix
    return Qr.dot(Qr.T)

def compute_usv(Q):
    U, s, V = np.linalg.svd(Q, full_matrices=True)
    S = np.zeros((U.shape[0], V.shape[0]))
    S[:s.shape[0], :s.shape[0]] = np.diag(s)
    
    s_inv = np.zeros(s.shape)
    for i in range(s_inv.shape[0]):
        s_inv[i] = s[i]
        if s_inv[i] < 1e-10: pass
        else: s_inv[i] = 1./s_inv[i]    
    S_inv = np.zeros((U.shape[0],V.shape[0]))
    S_inv[:s_inv.shape[0], :s_inv.shape[0]] = np.diag(s_inv)
    S_inv = S_inv.T
    
    return U, S, S_inv, V

def compute_rms_dist(X, Y, eps=1e-10):
    D = (X-Y)**2/(Y*Y+eps*eps)
    return (np.sum(D)/(D.shape[0]*D.shape[1]))**0.5

def create_random(N, M, lower=-0.1, upper=0.1):
    R = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            R[i][j] = np.random.uniform(lower, upper)
    return R

def load_X(tabfile):
    N = 9
    # LOAD XNKL
    X = np.zeros((N,N))
    ifs = open(tabfile, 'r')
    for ln in ifs.readlines():
        sp = ln.split()
        n = int(sp[0])
        k = int(sp[1])
        l = int(sp[2])
        xnkl = float(sp[3])
        X[n][k] = xnkl
    ifs.close()
    L1 = l+1
    L21 = 2*l+1
    return X, N, l, L1, L21
    
def setup_M(L1):
    M = np.zeros((L1,L1))
    M_inv = np.zeros((L1,L1))
    M[0][0] = 1.
    M_inv[0][0] = 1.
    for i in range(1, L1):
        M[i][i] = 2.
        M_inv[i][i] = 0.5
    return M, M_inv

def invert_xnkl_aa_fixed_l(X, Y, N, l, Q_return, P_return):
    assert X.shape == Y.shape == (N,N)    
    debug = False
    
    l1 = l+1
    l21 = 2*l+1
    
    # Real-valued coefficients (associated with real-valued Ylm)
    Qr_n = create_random(N,l21,-0.1,0.1)
    if l == 0:
        pass
    else:
        for m in np.arange(-l,l+1):
            if m != 0:
                Qr_n[:,l+m] = 0.
            else:
                pass
                #Qr_n[:,l+m] = 1.

    print "Starting l = %d, Qr_0 =" % l
    print Qr_n

    i = 0
    max_i = 4096
    converged = False
    omega_sor = 0.5
    tolerance = 1e-12
    while True:
        i += 1
        
        # Compute Q_n
        U, S, S_inv, V = compute_usv(Qr_n)
        Qr_n1 = V.T.dot(S_inv.dot(U.T.dot(X))).T
        Qr_n = omega_sor*(Qr_n+Qr_n1)

        # Check convergence
        X_n = compute_X(Qr_n)
        drms = compute_rms_dist(X_n, X)
        if drms < tolerance:
            print "Converged to precision after %d iterations." % (i)
            converged = True
            break
        if i >= max_i:
            print "Maximum iteration number reached, drms = %+1.7e" % drms
            converged = False
            break

    print "Finished l = %d with Qr =" % l
    print Qr_n

    if debug: print "X_n =\n", compute_X(Qr_n)
    if debug: print "X   =\n", X

    # Save to Q_return, P_return
    for n in range(N):
        m = 0
        Q_return[n][l*l+l+m] = Qr_n[n][l+m]
        P_return[n][l*l+l+m] = 0
        for m in np.arange(1,l+1):
            # Real qr m+/-
            qr_m_ = Qr_n[n][l-m] # l,-m
            qr_mx = Qr_n[n][l+m] # l,+m
            # Transform real qr m+/- to complex qc m+/-
            X__ = 0.5**0.5 * 1.j
            X_x = 0.5**0.5 * 1.j * (-1.)**(m+1)
            Xx_ = 0.5**0.5
            Xxx = 0.5**0.5 * (-1.)**(m)
            qc_m_ = qr_m_*X__ + qr_mx*Xx_
            qc_mx = qr_m_*X_x + qr_mx*Xxx
            # Store
            Q_return[n][l*l+l-m] = qc_m_.real
            P_return[n][l*l+l-m] = qc_m_.imag
            Q_return[n][l*l+l+m] = qc_mx.real
            P_return[n][l*l+l+m] = qc_mx.imag
                
    return Q_return, P_return, converged

def invert_xnkl_aa(xnkl_cmplx, basis, l_smaller=0, l_larger=-1):
    np.set_printoptions(precision=3, linewidth=120, threshold=10000)
    debug = False
    
    N = basis.N
    L = basis.L
    L1 = L+1
    L21 = 2*L+1
    
    # Sanity checks
    assert xnkl_cmplx.shape[0] == N*N and xnkl_cmplx.shape[1] == L+1
    
    # Split xnkl = X+i*Y onto real (X) and imaginary (Y) part, 
    Xnkl = np.copy(xnkl_cmplx.real)
    Ynkl = np.copy(xnkl_cmplx.imag)
    
    # Prepare qnlm = Q+i*P with real Q and imaginary P
    qnlm_cmplx = np.zeros((N, L21*L21), dtype='complex128')
    Qnlm = np.zeros((N, L1*L1))
    Pnlm = np.zeros((N, L1*L1))
    
    for l in range(L1):
        if l < l_smaller: continue
        if l and l == l_larger+1: break

        conv = (2.*l+1.)**0.5/(2.*2.**0.5*np.pi)
        
        Xnk = Xnkl[:,l]*conv
        Ynk = Ynkl[:,l]*conv
        Xnk = Xnk.reshape((N,N))
        Ynk = Ynk.reshape((N,N))
        
        Qnlm, Pnlm, converged = invert_xnkl_aa_fixed_l(Xnk, Ynk, N, l, Qnlm, Pnlm)
        
        if debug: osio << osio.my << Qnlm+1.j*Pnlm << endl
        if not converged: osio << osio.mr << "Not converged" << endl
        raw_input('...')       
    
    qnlm_complex = Qnlm + np.complex(0.,1.)*Pnlm
    if debug: print qnlm_complex

    return qnlm_complex
















