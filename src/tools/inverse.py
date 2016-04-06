import numpy as np
import numpy.linalg

from momo import osio, endl, flush


def compute_X(Q, P, M):
    # Q is N x (L+1) matrix
    # M = diag(1,2,2,...)
    return Q.dot(M.dot(Q.T)) + P.dot(M.dot(P.T))

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

def compute_rms_dist(X, Y):
    D = X-Y
    return (np.sum(D**2)/(D.shape[0]*D.shape[1]))**0.5

def create_random(N, M):
    R = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            R[i][j] = np.random.uniform(0.,1.)
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
    
    l1 = l+1
    
    # Real and imaginary l-expansions
    Q_n = create_random(N,l1)
    P_n = create_random(N,l1)
    P_n[:,0] = 0. # Yl0 is real for all l
    
    # Metric to reflect symmetries Y
    M, M_inv = setup_M(l1)

    with_P = False
    if not with_P: P_n[:,:] = 0.

    print "Starting l = %d, Q0+iP0 =" % (l1-1)
    for i in range(N):
        print Q_n[i], "+ i*", P_n[i]

    i = 0
    max_i = 512
    while True:
        i += 1

        # Compute Q_n
        U, S, S_inv, V = compute_usv(Q_n)
        #if i == 1:
        #    Q_n1 = M_inv.dot(V.T.dot(S_inv.dot(U.T.dot(X)))).T
        #else:
        Q_n1 = M_inv.dot(V.T.dot(S_inv.dot(U.T.dot(X-P_n.dot(M.dot(P_n.T)))))).T
        Q_n = 0.5*(Q_n+Q_n1)

        # Compute P_n
        if with_P:
            U, S, S_inv, V = compute_usv(P_n)
            P_n1 = M_inv.dot(V.T.dot(S_inv.dot(U.T.dot(X-Q_n.dot(M.dot(Q_n.T)))))).T
            P_n = 0.5*(P_n+P_n1)
            P_n[:,0] = 0.

        # Check convergence
        X_n = compute_X(Q_n, P_n, M)
        drms = compute_rms_dist(X_n, X)
        if drms < 1e-9:
            print "Converged to precision after %d iterations." % (i)
            break
        if i >= max_i:
            print "Maximum iteration number reached, drms = ", drms
            break

        #print Q_n_plus_1.dot(Q_n_plus_1.T)

    #print "Q_n \n", Q_n
    print "Finished with Q+iP ="
    for i in range(N):
        print Q_n[i], "+ i*", P_n[i]

    print "X_n \n", compute_X(Q_n, P_n, M)
    print "X   \n", X

    print "Y_n (should be zero) \n", P_n.dot(M.dot(Q_n.T)) - Q_n.dot(M.dot(P_n.T))

    # Save to Q_return, P_return
    for n in range(N):
        m = 0
        Q_return[n][l*l+l+m] = Q_n[n][m]
        P_return[n][l*l+l+m] = P_n[n][m]
        for m in range(1, l1):
            Q_return[n][l*l+l+m] = Q_n[n][m]
            Q_return[n][l*l+l-m] = Q_n[n][m]*(-1)**(m)
            P_return[n][l*l+l+m] = P_n[n][m]
            P_return[n][l*l+l-m] = P_n[n][m]*(-1)**(m+1)
                
    return Q_return, P_return

def invert_xnkl_aa(xnkl_cmplx, basis, l_smaller=0, l_larger=-1):
    np.set_printoptions(precision=3, linewidth=120, threshold=10000)
    
    N = basis.N
    L = basis.L
    L1 = L+1
    L21 = 2*L+1
    
    # Sanity checks
    assert xnkl_cmplx.shape[0] == N*N and xnkl_cmplx.shape[1] == L+1
    
    # Split xnkl = X+i*Y onto real (X) and imaginary (Y) part, 
    Xnkl = np.zeros((N*N, L1))
    Ynkl = np.zeros((N*N, L1))
    for n in range(N):
        for k in range(N):
            for l in range(L1):
                xy = xnkl_cmplx[n*N+k][l]
                Xnkl[n*N+k][l] = xy.real
                Ynkl[n*N+k][l] = xy.imag
    
    # Prepare qnlm = Q+i*P with real Q and imaginary P
    qnlm_cmplx = np.zeros((N, L21*L21), dtype='complex128')
    Qnlm = np.zeros((N, L1*L1))
    Pnlm = np.zeros((N, L1*L1))
    
    for l in range(L1):
        if l < l_smaller: continue
        
        Xnk = Xnkl[:,l]
        Ynk = Ynkl[:,l]
        Xnk = Xnk.reshape((N,N))
        Ynk = Ynk.reshape((N,N))
        
        Qnlm, Pnlm = invert_xnkl_aa_fixed_l(Xnk, Ynk, N, l, Qnlm, Pnlm)
        
        osio << osio.my << Qnlm << endl
        raw_input('...')
        if l == l_larger: break
    
    qnlm_complex = Qnlm + np.complex(0,1)*Pnlm
    print qnlm_complex

    return qnlm_complex
















