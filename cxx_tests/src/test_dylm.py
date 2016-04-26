import scipy.special
import numpy as np


def theta_hat(theta, phi):
    x = np.cos(theta)*np.cos(phi)
    y = np.cos(theta)*np.sin(phi)
    z = -np.sin(theta)
    return np.array([x,y,z])
    
def phi_hat(theta, phi):
    x = -np.sin(phi)
    y = +np.cos(phi)
    z = 0.
    return np.array([x,y,z])

def R_theta_phi(r):
    r = np.array(r)
    R = np.dot(r,r)**0.5
    theta = np.arccos(r[2]/R)
    
    if r[0] == 0.:
        if r[1] <= 0.:
            phi = -0.5*np.pi
        else:
            phi = +0.5*np.pi
    else:
        phi = np.arctan2(r[1],r[0])
    return R, theta, phi
    

def psi_10(r):
    R, theta, phi = R_theta_phi(r)    
    pre = - (3./(4.*np.pi))**0.5 * np.sin(theta)    
    direction = theta_hat(theta, phi)
    return pre*direction

def psi_11(r):
    R, theta, phi = R_theta_phi(r)    
    pre = - (3./(8.*np.pi))**0.5 * np.exp(phi*1.j)
    direction = np.cos(theta)*theta_hat(theta, phi) + 1.j*phi_hat(theta, phi)
    return pre*direction
    
def psi_20(r):
    R, theta, phi = R_theta_phi(r)
    return -1.5*(5./np.pi)**0.5 * np.sin(theta)*np.cos(theta)*theta_hat(theta, phi)

def psi_21(r):
    R, theta, phi = R_theta_phi(r)
    return -1.*(15./8./np.pi)**0.5*np.exp(phi*1.j)*( np.cos(2*theta)*theta_hat(theta,phi) + 1.j*np.cos(theta)*phi_hat(theta,phi) )
    
def psi_2_1(r):
    dylm_21 = psi_21(r)
    return -1*np.conjugate(dylm_21)
    
def psi_22(r):
    R, theta, phi = R_theta_phi(r)
    return (15./8./np.pi)**0.5*np.sin(theta)*np.exp(2.j*phi)*( np.cos(theta)*theta_hat(theta,phi) + 1.j*phi_hat(theta, phi) )
    
def psi_2_2(r):
    dylm_22 = psi_22(r)
    return np.conjugate(dylm_22)
    
r_list = []
r_list.append([2.,0.,0.])
r_list.append([0.,1.,0.])
r_list.append([0.,0.,1.])
r_list.append([0.5**0.5,0.,0.5**0.5])
r_list.append([-0.2,0.3,0.7])
r_list.append([-0.2,-0.4,-0.1])


results = []

print "==== THETA, PHI ===="
for r in r_list:
    R, theta, phi = R_theta_phi(r)
    print r, "=>", theta, phi

print "==== 10 ===="
for r in r_list:
    print r, "=>", " => unnorm", psi_10(r)/np.dot(r,r)**0.5
    results.append(psi_10(r)/np.dot(r,r)**0.5) 
    
print "==== 11 ===="  
for r in r_list:
    print r, "=>", " => unnorm", psi_11(r)/np.dot(r,r)**0.5
    results.append(psi_11(r)/np.dot(r,r)**0.5)
    
print "==== 20 ===="  
for r in r_list:
    print r, "=>", " => unnorm", psi_20(r)/np.dot(r,r)**0.5
    results.append(psi_20(r)/np.dot(r,r)**0.5)
    
print "==== 21 ===="  
for r in r_list:
    print r, "=>", " => unnorm", psi_21(r)/np.dot(r,r)**0.5
    results.append(psi_21(r)/np.dot(r,r)**0.5)
    
for r in r_list:
    results.append(psi_2_1(r)/np.dot(r,r)**0.5)

print "==== 22 ===="  
for r in r_list:
    print r, "=>", " => unnorm", psi_22(r)/np.dot(r,r)**0.5
    results.append(psi_22(r)/np.dot(r,r)**0.5)
    
for r in r_list:
    results.append(psi_2_2(r)/np.dot(r,r)**0.5)
    
for r in results:
    print "results.push_back(std::complex<double>(%+1.7e, %+1.7e));" % (r[0].real, r[0].imag)
    print "results.push_back(std::complex<double>(%+1.7e, %+1.7e));" % (r[1].real, r[1].imag)
    print "results.push_back(std::complex<double>(%+1.7e, %+1.7e));" % (r[2].real, r[2].imag)
    

