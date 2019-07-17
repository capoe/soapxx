import numpy as np
import soap
log = soap.log

def assert_equal(z, target, eps):
    if np.abs(z-target) > eps: 
        print z, target
        raw_input('...')
        #raise ValueError("%1.7e != %1.7e within %1.2e" % (z, target, eps))
    else: log << "+" << log.flush

def test_dot():
    log << log.mg << "<test_dot_grad>" << log.endl
    np.random.seed(971341)
    # BUILD GRAPH
    graph = soap.soapy.nn.PyGraph()
    X1 = graph.addNode(op="input", props={ "dim": 10, "tag": "X1" })
    X2 = graph.addNode(op="input", props={ "dim": 20, "tag": "X2" })
    Y1 = graph.addNode(op="input", props={ "dim": 1, "tag": "Y1" })
    L1 = graph.addNode(op="linear", parents=[X1,X2],    props={ "dim": 5, "params": "C12" })
    L2 = graph.addNode(op="sigmoid", parents=[X1,X1,X1], props={ "dim": 5, "params": "C12" })
    L5 = graph.addNode(op="scalar", parents=[L2])
    L3 = graph.addNode(op="dot", parents=[L1,L5], props={ "dim": 1 })
    L4 = graph.addNode(op="sigmoid", parents=[L3], props={ "dim": 1 })
    L6 = graph.addNode(op="exp", parents=[L4])
    L7 = graph.addNode(op="sigmoid", parents=[L6])
    xent = graph.addNode(op="xent", parents=[L7,Y1])
    graph.printInfo()
    # INITIALIZE & FEED
    for par in graph.params:
        par.randomize()
    X1_in = np.random.uniform(size=(10,10))
    X2_in = np.random.uniform(size=(10,20))
    Y1_in = np.random.uniform(size=(10,1))
    graph.propagate(feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
    graph.backpropagate(log=log)

    # CHECK DOT
    dot_12_chk = L1.val().dot(L5.val().T).diagonal()
    dot_12 = L3.val().flatten()
    log << log.mg << "Test dot" << log.flush
    assert_equal(np.max(np.abs(dot_12-dot_12_chk)), 0.0, 1e-6)
    log << log.endl
    sc_12_chk = L2.val()*L5.params.C
    sc_12 = L5.val()
    log << log.mg << "Test scalar" << log.flush
    assert_equal(np.max(np.abs(sc_12-sc_12_chk)), 0.0, 1e-6)
    log << log.endl
    exp_chk = np.exp(L4.val())
    exp = L6.val()
    log << log.mg << "Test exp" << log.flush
    assert_equal(np.max(np.abs(exp-exp_chk)), 0.0, 1e-6)
    log << log.endl

    # CHECK GRADIENTS: PARAMETERS
    h = 1e-9
    for node in [L4,L2,L1]:
        log << log.mg << "Test node" << node.idx << node.op << log.endl
        g_ref = np.copy(node.params.grad)
        obj = xent
        C0 = np.copy(node.params.C)
        graph.evaluate(obj, feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
        obj0 = obj.val()
        for i in range(C0.shape[0]):
            if i > 0 and i % 3 == 0: log << log.endl
            log << "Row %2d" % i << log.flush
            for j in range(C0.shape[1]):
                node.params.C = np.copy(C0)
                node.params.C[i,j] = C0[i,j]+h
                graph.evaluate(obj, feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
                obj1 = obj.val()
                g_num = (obj1-obj0)/h
                err = (g_num-g_ref[i,j])/g_num
                assert_equal(g_num, g_ref[i,j], 1e-5)
            log << "  " << log.flush
        log << log.endl
        node.params.C = C0
    graph.propagate(feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
    graph.backpropagate()
    obj0 = obj.val()
    for node in [L5]:
        log << log.mg << "Test node" << node.idx << node.op << log.endl
        g_ref = np.copy(node.params.grad)
        obj = xent
        C0 = np.copy(node.params.C)
        graph.evaluate(obj, feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
        obj0 = obj.val()
        for i in range(C0.shape[0]):
            log << "Row %2d" % i << log.flush
            node.params.C = np.copy(C0)
            node.params.C[i] = C0[i]+h
            graph.evaluate(obj, feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
            obj1 = obj.val()
            g_num = (obj1-obj0)/h
            log << "%+1.7e == %+1.7e" % (g_num, g_ref[i]) << log.flush
            assert_equal(g_num, g_ref[i], 1e-5)
            log << "  " << log.endl
        node.params.C = C0

    # CHECK GRADIENTS: INPUTS
    graph.propagate(feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
    graph.backpropagate()
    obj0 = obj.val()
    for key, node, X_in in zip(["X1", "X2", "Y1"], [ X1, X2, Y1 ], [ X1_in, X2_in, Y1_in ]):
        log << log.mg << "Test input" << key << node.op << log.endl
        X_in_ref = np.copy(X_in)
        g_ref = np.copy(node.grad)
        for i in range(X_in.shape[0]):
            log << "Sample %2d" % i << log.flush
            for j in range(X_in.shape[1]):
                X_in = np.copy(X_in_ref)
                X_in[i,j] = X_in[i,j] + h
                feed = {"X1": X1_in, "X2": X2_in, "Y1": Y1_in}
                feed[key] = X_in
                graph.propagate(feed=feed, log=None)
                obj1 = obj.val()
                g_num = (obj1-obj0)/h
                err = (g_num-g_ref[i,j])
                assert_equal(err, 0., 2e-5)
            log << log.endl
        X_in = np.copy(X_in_ref)

def test_xent_grad():
    log << log.mg << "<test_xent_grad>" << log.endl
    np.random.seed(971341)
    # BUILD GRAPH
    graph = soap.soapy.nn.PyGraph()
    X1 = graph.addNode(op="input", props={ "dim": 10, "tag": "X1" })
    X2 = graph.addNode(op="input", props={ "dim": 20, "tag": "X2" })
    Y1 = graph.addNode(op="input", props={ "dim": 1, "tag": "Y1" })
    L1 = graph.addNode(op="linear", parents=[X1,X2],    props={ "dim": 5, "params": "C12" })
    L2 = graph.addNode(op="sigmoid", parents=[X1,X1,X1], props={ "dim": 5, "params": "C12" })
    L4 = graph.addNode(op="mult", parents=[L1,L2])
    L3 = graph.addNode(op="sigmoid", parents=[L1,L4],    props={ "dim": 1 })
    xent = graph.addNode(op="xent", parents=[L3,Y1])
    graph.printInfo()
    # INITIALIZE & FEED
    for par in graph.params:
        par.randomize()
    X1_in = np.random.uniform(size=(10,10))
    X2_in = np.random.uniform(size=(10,20))
    Y1_in = np.random.uniform(size=(10,1))
    graph.propagate(feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
    graph.backpropagate(log=log)
    # CHECK GRADIENTS: PARAMETERS
    h = 1e-7
    for node in [L3,L2,L1]:
        log << log.mg << "Test node" << node.idx << node.op << log.endl
        g_ref = np.copy(node.params.grad)
        obj = xent
        C0 = np.copy(node.params.C)
        graph.evaluate(obj, feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
        obj0 = obj.val()
        for i in range(C0.shape[0]):
            if i > 0 and i % 3 == 0: log << log.endl
            log << "Row %2d" % i << log.flush
            for j in range(C0.shape[1]):
                node.params.C = np.copy(C0)
                node.params.C[i,j] = C0[i,j]+h
                graph.evaluate(obj, feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
                obj1 = obj.val()
                g_num = (obj1-obj0)/h
                err = (g_num-g_ref[i,j])/g_num
                assert_equal(err, 0., 2e-5)
            log << "  " << log.flush
        log << log.endl
        node.params.C = C0

    # CHECK GRADIENTS: INPUTS
    graph.propagate(feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
    graph.backpropagate()
    obj0 = obj.val()
    for key, node, X_in in zip(["X1", "X2", "Y1"], [ X1, X2, Y1 ], [ X1_in, X2_in, Y1_in ]):
        log << log.mg << "Test input" << key << node.op << log.endl
        X_in_ref = np.copy(X_in)
        g_ref = np.copy(node.grad)
        for i in range(X_in.shape[0]):
            log << "Sample %2d" % i << log.flush
            for j in range(X_in.shape[1]):
                X_in = np.copy(X_in_ref)
                X_in[i,j] = X_in[i,j] + h
                feed = {"X1": X1_in, "X2": X2_in, "Y1": Y1_in}
                feed[key] = X_in
                graph.propagate(feed=feed, log=None)
                obj1 = obj.val()
                g_num = (obj1-obj0)/h
                err = (g_num-g_ref[i,j])
                assert_equal(err, 0., 2e-5)
            log << log.endl
        X_in = np.copy(X_in_ref)

def test_functions_grad():
    log << log.mg << "<test_sigmoid>" << log.endl
    np.random.seed(971341)
    # BUILD GRAPH
    graph = soap.soapy.nn.PyGraph()
    X1 = graph.addNode(op="input", props={ "dim": 10, "tag": "X1" })
    X2 = graph.addNode(op="input", props={ "dim": 20, "tag": "X2" })
    Y1 = graph.addNode(op="input", props={ "dim": 1, "tag": "Y1" })
    L1 = graph.addNode(op="linear", parents=[X1,X2],    props={ "dim": 5, "params": "C12" })
    L2 = graph.addNode(op="sigmoid", parents=[X1,X1,X1], props={ "dim": 5, "params": "C12" })
    L3 = graph.addNode(op="sigmoid", parents=[L1,L2],    props={ "dim": 1 })
    mse = graph.addNode(op="mse", parents=[L3,Y1])
    graph.printInfo()
    # INITIALIZE & FEED
    for par in graph.params:
        par.randomize()
    X1_in = np.random.uniform(size=(10,10))
    X2_in = np.random.uniform(size=(10,20))
    Y1_in = np.random.uniform(size=(10,1))
    graph.propagate(feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
    graph.backpropagate(log=log)
    # CHECK GRADIENTS: PARAMETERS
    h = 1e-7
    for node in [L3,L2,L1]:
        log << log.mg << "Test node" << node.idx << node.op << log.endl
        g_ref = np.copy(node.params.grad)
        obj = mse
        C0 = np.copy(node.params.C)
        graph.evaluate(obj, feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
        obj0 = obj.val()
        for i in range(C0.shape[0]):
            if i > 0 and i % 3 == 0: log << log.endl
            log << "Row %2d" % i << log.flush
            for j in range(C0.shape[1]):
                node.params.C = np.copy(C0)
                node.params.C[i,j] = C0[i,j]+h
                graph.evaluate(obj, feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
                obj1 = obj.val()
                g_num = (obj1-obj0)/h
                err = (g_num-g_ref[i,j])/g_num
                assert_equal(err, 0., 2e-5)
            log << "  " << log.flush
        log << log.endl
        node.params.C = C0

    # CHECK GRADIENTS: INPUTS
    graph.propagate(feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
    graph.backpropagate()
    obj0 = obj.val()
    for key, node, X_in in zip(["X1", "X2", "Y1"], [ X1, X2, Y1 ], [ X1_in, X2_in, Y1_in ]):
        log << log.mg << "Test input" << key << node.op << log.endl
        X_in_ref = np.copy(X_in)
        g_ref = np.copy(node.grad)
        for i in range(X_in.shape[0]):
            log << "Sample %2d" % i << log.flush
            for j in range(X_in.shape[1]):
                X_in = np.copy(X_in_ref)
                X_in[i,j] = X_in[i,j] + h
                feed = {"X1": X1_in, "X2": X2_in, "Y1": Y1_in}
                feed[key] = X_in
                graph.propagate(feed=feed, log=None)
                obj1 = obj.val()
                g_num = (obj1-obj0)/h
                err = (g_num-g_ref[i,j])
                assert_equal(err, 0., 2e-5)
            log << log.endl
        X_in = np.copy(X_in_ref)

def test_functions():
    log << log.mg << "<test_functions>" << log.endl
    np.random.seed(971341)
    graph = soap.soapy.nn.PyGraph()
    X1 = graph.addNode(op="input", props={ "dim": 10, "tag": "X1" })
    X2 = graph.addNode(op="input", props={ "dim": 20, "tag": "X2" })
    Y1 = graph.addNode(op="input", props={ "dim": 1, "tag": "Y1" })
    L1 = graph.addNode(op="linear", parents=[X1,X2], props={ "dim": 5 })
    L2 = graph.addNode(op="linear", parents=[X1], props={ "dim": 5 })
    L3 = graph.addNode(op="sigmoid", parents=[L1,L2],    props={ "dim": 1 })
    mse = graph.addNode(op="mse", parents=[L3,Y1])
    for par in graph.params: par.randomize()
    X1_in = np.random.uniform(size=(10,10))
    X2_in = np.random.uniform(size=(10,20))
    Y1_in = np.random.uniform(size=(10,1))
    graph.propagate(feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
    log << "[linear]" << log.flush
    v1_chk = np.concatenate([X1_in,X2_in], axis=1).dot(L1.params.C[0:-1])+L1.params.C[-1]
    v1 = L1.val()
    assert_equal(np.max(np.abs(v1_chk-v1)), 0., 1e-7)
    v2_chk = X1_in.dot(L2.params.C[0:-1])+L2.params.C[-1]
    v2 = L2.val()
    assert_equal(np.max(np.abs(v2_chk-v2)), 0., 1e-7)
    log << "[sigmoid]" << log.flush
    v3_chk = np.concatenate([v1_chk, v2_chk], axis=1).dot(L3.params.C[0:-1])+L3.params.C[-1]
    v3_chk = 1./(1.+np.exp(v3_chk))
    v3 = L3.val()
    assert_equal(np.max(np.abs(v3_chk-v3)), 0., 1e-7)
    log << "[mse]" << log.flush
    mse_chk = np.sum((v3_chk-Y1_in)**2)/v3_chk.shape[0]
    assert_equal(mse_chk, mse.val(), 1e-7)
    log << log.endl
    assert np.max(np.abs(v3_chk)) > 0.1 # guard against zeros

def test_adagrad():
    log << log.mg << "<test_adagrad>" << log.endl
    np.random.seed(971341)
   
    # BUILD GRAPH
    graph = soap.soapy.nn.PyGraph()
    X1 = graph.addNode(op="input", props={ "dim": 10, "tag": "X1" })
    X2 = graph.addNode(op="input", props={ "dim": 20, "tag": "X2" })
    Y1 = graph.addNode(op="input", props={ "dim": 1, "tag": "Y1" })
    L1 = graph.addNode(op="sigmoid", parents=[X1,X2],    props={ "dim": 5, "params": "C12" })
    L2 = graph.addNode(op="sigmoid", parents=[X1,X1,X1], props={ "dim": 5, "params": "C12" })
    L3 = graph.addNode(op="sigmoid", parents=[L1,L2],    props={ "dim": 1 })
    mse = graph.addNode(op="mse", parents=[L3,Y1])
    graph.printInfo()

    # INITIALIZE & FEED
    for par in graph.params:
        par.randomize()
    X1_in = np.random.uniform(size=(10,10))
    X2_in = np.random.uniform(size=(10,20))
    Y1_in = np.random.uniform(size=(10,1))

    # OPTIMIZE
    opt = soap.soapy.nn.OptAdaGrad(props={"rate": 0.1})
    opt.initialize(graph)
    opt.step(graph=graph, n_steps=1000, report_every=100, obj=mse,
        feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in}, log=log)

def test_softmax_grad():
    log << log.mg << "<test_softmax>" << log.endl
    np.random.seed(971341)
   
    # BUILD GRAPH
    graph = soap.soapy.nn.PyGraph()
    X1 = graph.addNode(op="input", props={ "dim": 10, "tag": "X1" })
    X2 = graph.addNode(op="input", props={ "dim": 20, "tag": "X2" })
    Y1 = graph.addNode(op="input", props={ "dim": 1, "tag": "Y1" })
    L1 = graph.addNode(op="sigmoid", parents=[X1,X2],    props={ "dim": 5, "params": "C12" })
    L2 = graph.addNode(op="sigmoid", parents=[X1,X1,X1], props={ "dim": 5, "params": "C12" })
    L3 = graph.addNode(op="softmax", parents=[L1])
    L4 = graph.addNode(op="sigmoid", parents=[L1,L2,L3],    props={ "dim": 1 })
    mse = graph.addNode(op="mse", parents=[L4,Y1])
    graph.printInfo()

    # INITIALIZE & FEED
    for par in graph.params:
        par.randomize()
    X1_in = np.random.uniform(size=(10,10))
    X2_in = np.random.uniform(size=(10,20))
    Y1_in = np.random.uniform(size=(10,1))
    graph.propagate(feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
    graph.backpropagate(log=log)

    # CHECK GRADIENTS: PARAMETERS
    h = 1e-7
    for node in [L1]:
        log << log.mg << "Test node" << node.idx << node.op << log.endl
        g_ref = np.copy(node.params.grad)
        obj = mse
        C0 = np.copy(node.params.C)
        graph.evaluate(obj, feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
        obj0 = obj.val()
        for i in range(C0.shape[0]):
            if i > 0 and i % 3 == 0: log << log.endl
            log << "Row %2d" % i << log.flush
            for j in range(C0.shape[1]):
                node.params.C = np.copy(C0)
                node.params.C[i,j] = C0[i,j]+h
                graph.evaluate(obj, feed={"X1": X1_in, "X2": X2_in, "Y1": Y1_in})
                obj1 = obj.val()
                g_num = (obj1-obj0)/h
                assert_equal(g_num, g_ref[i,j], 1e-7)
            log << "  " << log.flush
        log << log.endl
        node.params.C = C0

if __name__ == "__main__":
    test_functions()
    test_functions_grad()
    test_softmax_grad()
    test_adagrad()
    test_xent_grad()
    test_dot()

