import numpy as np
import soap
log = soap.log

def assert_equal(z, target, eps=1e-5, verbose=False):
    if np.abs(z-target) > eps: raise ValueError(z)
    elif verbose: log << "+" << log.flush

def setup(configs, soap_options, verbose=True):
    # Returns average normalized SOAP vectors (DMap's) for all configs, along with a dummy weight vector
    dset = soap.DMapMatrixSet()
    for idx, config in enumerate(configs):
        if verbose: log << log.back << "SOAP for structure" << idx << log.flush
        if idx == 0: soap_options["spectrum.gradients"] = True
        else: soap_options["spectrum.gradients"] = False
        dset.append(soap.soapy.PowerSpectrum(config, soap_options).exportDMapMatrix())
    if verbose: log << log.endl
    for i in range(len(dset)):
        dset[i].sum()
        dset[i].normalize()
    return dset[0][0], [ dset[i][0] for i in range(1, len(dset)) ], np.ones(shape=(len(dset)-1,))/(len(dset)-1)

def predict_with_grad(x_test, x_train_list, w_train, xi=2., verbose=True):
    y_test_with_grad = soap.DMap()
    for i in range(len(x_train_list)):
        if verbose: log << log.back << "Kernel contribution from training structure" << i << log.flush
        x_test.dotGradLeft(x_train_list[i], w_train[i], xi, y_test_with_grad)
    if verbose: log << log.endl
    y_test = y_test_with_grad.val()
    y_test_grad = { g.pid: np.array([ g.x, g.y, g.z ]) for g in y_test_with_grad.gradients }
    return y_test, y_test_grad

if __name__ == "__main__":
    # Analytical gradients
    configs = soap.tools.io.read('structures.xyz')
    soap_options = soap.soapy.configure_default()
    soap.soapy.PowerSpectrum.verbose = False
    x_test, x_train_list, w_train = setup(configs, soap_options)
    y_test, y_test_grad = predict_with_grad(x_test, x_train_list, w_train)
    log << "Prediction y=k.w= %+1.4f" % y_test << log.endl
    for pid in sorted(y_test_grad):
        log << " dy/dr_%-2d= %+1.4f %+1.4f %+1.4f" % (pid, y_test_grad[pid][0], 
            y_test_grad[pid][1], y_test_grad[pid][2]) << log.endl

    # Numerical gradients
    log << "Numerical checks" << log.endl
    R = np.copy(configs[0].positions)
    for pid_check in sorted(y_test_grad):
        h = 0.00001
        y_check_grad = []
        for dim in range(3):
            configs[0].positions = np.copy(R)
            configs[0].positions[pid_check-1][dim] += h
            x_test_h, _, _ = setup(configs[0:1], soap_options, verbose=False)
            y_test_h, _ = predict_with_grad(x_test_h, x_train_list, w_train, verbose=False)
            y_check_grad.append((y_test_h-y_test)/h)
        log << " dy/dr_%-2d= %+1.4f %+1.4f %+1.4f (numerical)" % (pid_check, y_check_grad[0], 
            y_check_grad[1], y_check_grad[2]) << log.endl
        for dim in range(3):
            assert_equal(y_check_grad[dim]-y_test_grad[pid_check][dim], 0., 1e-5)

