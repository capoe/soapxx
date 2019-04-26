import librmt as rmt
import soap
import sys
import numpy as np
log = soap.log
soap.silence()
np.random.seed(971872)

def assert_equal(z, target, eps=1e-5):
    if np.abs(z-target) > eps: raise ValueError(z)
    else: log << "+" << log.flush

def test_linear():
    log << log.mg << "%-17s" % "<test_linear>" << log.flush
    w = np.array([1.5,-2.5,0.3,0.1])
    X = np.random.uniform(size=(15,3))
    cgraph = soap.CGraph()
    c1 = cgraph.addInput()
    c2 = cgraph.addInput()
    c3 = cgraph.addInput()
    c4 = cgraph.addNode("linear", [c1,c2,c3])
    c4.params().set(w, "float64") 
    cgraph.evaluate(X, str(X.dtype))
    y = c4.vals("float64")
    y_check = X.dot(w[0:3])+w[3]
    assert_equal(np.max(np.abs(y-y_check)), 0.0, 1e-10)
    log << log.endl

def test_operators():
    log << log.mg << "%-17s" % "<test_operators>" << log.flush
    cgraph = soap.CGraph()
    c1 = cgraph.addInput()
    c2 = cgraph.addInput()
    c3 = cgraph.addNode("exp", [c1])
    c4 = cgraph.addNode("log", [c1])
    c5 = cgraph.addNode("mod", [c2])
    c6 = cgraph.addNode("pow", [c1])
    c7 = cgraph.addNode("mult", [c1,c2])
    c8 = cgraph.addNode("div", [c1,c2])
    X1 = np.random.uniform(size=(15,1))
    X2 = np.random.uniform(-1,1,size=(15,1))
    X = np.concatenate([X1,X2], axis=1)
    for n in range(3):
        for params in cgraph.params:
            rnd_vals = np.random.uniform(-1, 1, size=(params.size,))
            params.set(rnd_vals, str(rnd_vals.dtype))
        cgraph.evaluate(X, str(X.dtype))
        v3 = c3.vals("float64")
        v4 = c4.vals("float64")
        v5 = c5.vals("float64")
        v6 = c6.vals("float64")
        v7 = c7.vals("float64")
        v8 = c8.vals("float64")
        v3c = np.exp(c3.params().vals("float64")[0]*X1[:,0])
        v4c = np.log(X1[:,0])
        v5c = np.abs(X2[:,0])
        v6c = X1[:,0]**c6.params().vals("float64")[0]
        v7c = X1[:,0]*X2[:,0]
        v8c = X1[:,0]/X2[:,0]
        assert_equal(np.max(np.abs(v3-v3c)), 0.0, 1e-10)
        assert_equal(np.max(np.abs(v4-v4c)), 0.0, 1e-10)
        assert_equal(np.max(np.abs(v5-v5c)), 0.0, 1e-10)
        assert_equal(np.max(np.abs(v6-v6c)), 0.0, 1e-10)
        assert_equal(np.max(np.abs(v7-v7c)), 0.0, 1e-10)
        assert_equal(np.max(np.abs(v8-v8c)), 0.0, 1e-10)
    log << log.endl

def test_operators_grad():
    log << log.mg << "%-17s" % "<test_operators_grad>" << log.endl
    w = np.array([0.5,-1.,0.2,0.1])
    X = np.random.uniform(size=(15,3))
    Y = np.random.uniform(size=(15,1))
    for operator in ["exp", "log", "mod", "pow", "mult", "div"]:
        cgraph = soap.CGraph()
        c1 = cgraph.addInput()
        c2 = cgraph.addInput()
        c3 = cgraph.addInput()
        c4 = cgraph.addNode("linear", [c1,c2,c3])
        c4abs = cgraph.addNode("mod", [c4])
        c5 = cgraph.addNode("sigmoid", [c2,c3,c4])
        if operator in {"mult","div"}:
            deps = [c4,c5]
        elif operator in {"log","pow"}:
            deps = [c4abs]
        else:
            deps = [c4]
        cu = cgraph.addNode(operator, deps)
        c6 = cgraph.addNode("linear", [cu,c5])
        t1 = cgraph.addTarget()
        o1 = cgraph.addObjective("mse", [c6,t1])
        for node in cgraph.nodes():
            node.params().set(
                np.random.uniform(size=(node.params().size,)), 
                "float64")
        for i in range(2):
            if cu.params().size > 0:
                cgrad = cu if i == 0 else c5
                n_params = 1 if i == 0 else 4
            else:
                cgrad = c4 if i == 0 else c5
                n_params = 4 
            # Evaluate graph 
            cgrad.params().set(w, "float64") 
            cgraph.feed(X, Y, str(X.dtype), str(Y.dtype))
            mse = o1.vals("float64")[0]
            # Analytical gradients
            p4_id = cgrad.params().id
            o1.grads().listParamSets()
            g4 = o1.grads().vals(p4_id, "float64");
            # Numerical gradients
            h = 1e-8
            for i in range(n_params):
                wh = np.copy(w)
                wh[i] += h
                cgrad.params().set(wh, "float64")
                cgraph.feed(X, Y, str(X.dtype), str(Y.dtype))
                mseh = o1.vals("float64")[0]
                g4i_num = (mseh-mse)/h
                log << "  %-4s / %-7s %+1.4e == %+1.4e ?" % (operator, cgrad.op, g4i_num, g4[i,0]) << log.flush
                assert_equal(g4[i,0]-g4i_num, 0.0, 1e-4)
                log << log.endl

def test_sigmoid():
    log << log.mg << "%-17s" % "<test_sigmoid>" << log.flush
    w = np.array([1.,-2.,0.5,0.4])
    X = np.random.uniform(size=(15,3))
    cgraph = soap.CGraph()
    c1 = cgraph.addInput()
    c2 = cgraph.addInput()
    c3 = cgraph.addInput()
    c4 = cgraph.addNode("sigmoid", [c1,c2,c3])
    c4.params().set(w, "float64") 
    cgraph.evaluate(X, str(X.dtype))
    y = c4.vals("float64")
    y_check = 1./(1. + np.exp(-X.dot(w[0:3])-w[3]))
    assert_equal(np.max(np.abs(y-y_check)), 0.0, 1e-10)
    log << log.endl

def test_misc():
    log << log.mg << "%-17s" % "<test_misc>" << log.flush
    cgraph = soap.CGraph()
    c1 = cgraph.addInput()
    c2 = cgraph.addInput()
    c3 = cgraph.addInput()
    c4 = cgraph.addNode("sigmoid", [c1,c2,c3])
    c5 = cgraph.addNode("linear", [c1,c4])
    X = np.random.uniform(size=(15,3))
    w4 = np.array([0.,-1.,0.2,0.1])
    w5 = np.array([-0.1,0.5,1.2])
    c4.params().set(w4, "float64")
    c5.params().set(w5, "float64")
    cgraph.evaluate(X, "float64")
    y4 = c4.vals("float64")
    y5 = c5.vals("float64")
    y4_check = 1./(1. + np.exp(-X.dot(w4[0:3])-w4[3]))
    y5_check = w5[0]*X[:,0] + w5[1]*y4_check + w5[2]
    assert_equal(np.max(np.abs(y4-y4_check)), 0.0, 1e-10)
    assert_equal(np.max(np.abs(y5-y5_check)), 0.0, 1e-10)
    log << log.endl

def test_mse():
    log << log.mg << "%-17s" % "<test_mse>" << log.flush
    w = np.array([0.,-1.,0.2,0.1])
    X = np.random.uniform(size=(15,3))
    Y = np.random.uniform(size=(15,1))
    cgraph = soap.CGraph()
    c1 = cgraph.addInput()
    c2 = cgraph.addInput()
    c3 = cgraph.addInput()
    c4 = cgraph.addNode("sigmoid", [c1,c2,c3])
    t1 = cgraph.addTarget()
    o1 = cgraph.addObjective("mse", [c4,t1])
    c4.params().set(w, "float64") 
    cgraph.feed(X, Y, str(X.dtype), str(Y.dtype))
    y = c4.vals("float64")
    y_check = 1./(1. + np.exp(-X.dot(w[0:3])-w[3]))
    mse = o1.vals("float64")
    mse_check = np.sum((y_check-Y[:,0])**2)/Y.shape[0]
    assert_equal(np.max(np.abs(y-y_check)), 0.0, 1e-10)
    assert_equal(mse_check-mse, 0.0, 1e-10)
    log << log.endl

def test_xent():
    log << log.mg << "%-17s" % "<test_xent>" << log.flush
    w = np.array([0.,-1.,0.2,0.1])
    X = np.random.uniform(size=(15,3))
    Y = 1.*np.random.randint(0, 2, size=(15,1))
    cgraph = soap.CGraph()
    c1 = cgraph.addInput()
    c2 = cgraph.addInput()
    c3 = cgraph.addInput()
    c4 = cgraph.addNode("sigmoid", [c1,c2,c3])
    t1 = cgraph.addTarget()
    o1 = cgraph.addObjective("xent", [c4,t1])
    c4.params().set(w, "float64") 
    cgraph.feed(X, Y, str(X.dtype), str(Y.dtype))
    y = c4.vals("float64")
    y_check = 1./(1. + np.exp(-X.dot(w[0:3])-w[3]))
    xent = o1.vals("float64")
    xent_check = -1./Y.shape[0]*np.sum(Y[:,0]*np.log(y_check) + (1.-Y[:,0])*np.log(1.-y_check))
    assert_equal(np.max(np.abs(y-y_check)), 0.0, 1e-10)
    assert_equal(xent_check-xent, 0.0, 1e-10)
    log << log.endl

def test_xent_grad():
    log << log.mg << "%-17s" % "<test_xent_grad>" << log.endl
    w = np.array([0.,-1.,0.2,0.1])
    X = np.random.uniform(size=(15,3))
    Y = np.random.uniform(size=(15,1))
    cgraph = soap.CGraph()
    c1 = cgraph.addInput()
    c2 = cgraph.addInput()
    c3 = cgraph.addInput()
    c4 = cgraph.addNode("linear", [c1,c2,c3])
    c5 = cgraph.addNode("sigmoid", [c2,c3,c4])
    c6 = cgraph.addNode("sigmoid", [c4,c5])
    t1 = cgraph.addTarget()
    o1 = cgraph.addObjective("xent", [c6,t1])
    for node in cgraph.nodes():
        node.params().set(
            np.random.uniform(size=(node.params().size,)), 
            "float64")
    cgrads = [c4,c5]
    for i in range(2):  
        cgrad = cgrads[i]
        # Evaluate graph 
        cgrad.params().set(w, "float64") 
        cgraph.feed(X, Y, str(X.dtype), str(Y.dtype))
        xent = o1.vals("float64")[0]
        # Analytical gradients
        p4_id = cgrad.params().id
        o1.grads().listParamSets()
        g4 = o1.grads().vals(p4_id, "float64");
        # Numerical gradients
        h = 1e-8
        for i in range(len(w)):
            wh = np.copy(w)
            wh[i] += h
            cgrad.params().set(wh, "float64")
            cgraph.feed(X, Y, str(X.dtype), str(Y.dtype))
            xenth = o1.vals("float64")[0]
            g4i_num = (xenth-xent)/h
            log << "  %-7s %+1.4e == %+1.4e ?" % (cgrad.op, g4i_num, g4[i,0]) << log.flush
            assert_equal(g4[i][0]-g4i_num, 0.0, 1e-7)
            log << log.endl

def test_mse_grad():
    log << log.mg << "%-17s" % "<test_mse_grad>" << log.endl
    w = np.array([0.,-1.,0.2,0.1])
    X = np.random.uniform(size=(15,3))
    Y = np.random.uniform(size=(15,1))
    cgraph = soap.CGraph()
    c1 = cgraph.addInput()
    c2 = cgraph.addInput()
    c3 = cgraph.addInput()
    c4 = cgraph.addNode("linear", [c1,c2,c3])
    c5 = cgraph.addNode("sigmoid", [c2,c3,c4])
    c6 = cgraph.addNode("linear", [c4,c5])
    t1 = cgraph.addTarget()
    o1 = cgraph.addObjective("mse", [c6,t1])
    for node in cgraph.nodes():
        node.params().set(
            np.random.uniform(size=(node.params().size,)), 
            "float64")
    cgrads = [c4,c5]
    for i in range(2):  
        cgrad = cgrads[i]
        # Evaluate graph 
        cgrad.params().set(w, "float64") 
        cgraph.feed(X, Y, str(X.dtype), str(Y.dtype))
        y = cgrad.vals("float64")
        y_check = 1./(1. + np.exp(-X.dot(w[0:3])-w[3]))
        mse = o1.vals("float64")[0]
        # Analytical gradients
        p4_id = cgrad.params().id
        o1.grads().listParamSets()
        g4 = o1.grads().vals(p4_id, "float64");
        # Numerical gradients
        h = 1e-8
        for i in range(len(w)):
            wh = np.copy(w)
            wh[i] += h
            cgrad.params().set(wh, "float64")
            cgraph.feed(X, Y, str(X.dtype), str(Y.dtype))
            mseh = o1.vals("float64")[0]
            g4i_num = (mseh-mse)/h
            log << "  %-7s %+1.4e == %+1.4e ?" % (cgrad.op, g4i_num, g4[i,0]) << log.flush
            assert_equal(g4[i][0]-g4i_num, 0.0, 1e-7)
            log << log.endl

def test_input_grads():
    log << log.mg << "%-17s" % "<test_input_grads>" << log.endl
    X = np.random.uniform(size=(15,3))
    Y = np.random.uniform(size=(15,1))
    for operator in ["exp", "log", "mod", "pow", "mult", "div"]:
        cgraph = soap.CGraph()
        c1 = cgraph.addInput()
        c2 = cgraph.addInput()
        c3 = cgraph.addInput()
        if operator in {"mult","div"}:
            deps = [c2,c3]
        elif operator in {"log","pow"}:
            deps = [c1]
        else:
            deps = [c3]
        cu = cgraph.addNode(operator, deps)
        c4 = cgraph.addNode("linear", [c1,c2,c3])
        c5 = cgraph.addNode("sigmoid", [c1,c2,c3,cu])
        c6 = cgraph.addNode("mult", [c4,c5])
        for node in cgraph.nodes():
            node.params().set(
                np.random.uniform(size=(node.params().size,)), 
                "float64")
        h = 1e-8
        for cgrad in [ c4, c5, c6 ]:
            # Analytical
            cgraph.evaluateInputGrads(X, str(X.dtype))
            x = cgrad.vals("float64")
            g = np.concatenate(
                [ cgrad.input_grads().vals(j+1, "float64") for j in range(len(cgraph.inputs())) ],
                axis=0).T # n_samples x n_inputs
            for j in range(X.shape[1]):
                # Numerical
                Xh = np.copy(X)
                Xh[:,j] = Xh[:,j] + h
                cgraph.evaluate(Xh, str(Xh.dtype))
                xh = cgrad.vals("float64")
                g_num = (xh-x)/h
                mean_rel_abserr = np.average(np.abs((g_num-g[:,j])/g_num))
                log << "  %-7s MRAE = %+1.4e" % (cgrad.op, mean_rel_abserr) << log.flush
                for i in range(X.shape[0]):
                    assert_equal(g[i,j]-g_num[i], 0.0, 1e-6)
                log << log.endl

if __name__ == "__main__":
    test_linear()
    test_sigmoid()
    test_misc()
    test_mse()
    test_xent()
    test_operators()
    test_mse_grad()
    test_xent_grad()
    test_operators_grad()
    test_input_grads()
    log << "All passed." << log.endl

