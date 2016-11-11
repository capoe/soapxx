import numpy as np

def subsample_array(array, n_select, method='stride', stride_shift=0):
    # Number of data items; selected, discarded items
    if type(array) == list:
        n_data = len(array)
    else:
        n_data = array.shape[0]
    # Treat border cases
    if n_select >= n_data:
        return array, []
    elif n_select <= 0:
        return [], array
    n_discard = n_data - n_select
    # Subsample
    if method == 'random':
        # Random access indices
        idcs = np.arange(0, n_data)
        np.random.shuffle(idcs)
        # Create selections
        if type(array) == list:
            array_select = []
            array_discard = []
            for i in range(n_data):
                if i < n_select:
                    array_select.append(array[idcs[i]])
                else:
                    array_discard.append(array[idcs[i]])
        else:
            raise NotImplementedError("<subsample_data_array> Array type other than list")
    elif method == 'stride':
        # Construct index sets
        idcs = np.arange(0, n_data)
        idcs_sel = [ int(float(idcs.shape[0])/n_select*i) for i in range(n_select) ]
        idcs_sel = np.array(idcs_sel)
        # Take into account periodic shift
        if stride_shift != 0:
            idcs_sel = np.sort((idcs_sel+stride_shift) % idcs.shape[0])
        # Index complement
        mask = np.zeros(idcs.shape[0], dtype=bool)
        mask[idcs_sel] = True
        idcs_disc = idcs[~mask]
        # Create selections
        if type(array) == list:
            array_select = []
            array_discard = []
            for i in idcs_sel:
                array_select.append(array[i])
            for i in idcs_disc:
                array_discard.append(array[i])
        else:
            raise NotImplementedError("<subsample_data_array> Array type other than list")
    return array_select, array_discard

if __name__ == "__main__":
    print "Data array"
    a = range(20)
    print a
    print "Select with stride, shift=0"
    a1, a2 = subsample_array(a, 5, 'stride', 0)
    print a1, a2
    print "Select with stride, shift=6"
    a1, a2 = subsample_array(a, 5, 'stride', 6)
    print a1, a2
    print "Select random subsample"
    a1, a2 = subsample_array(a, 5, 'random')
    print a1, a2



