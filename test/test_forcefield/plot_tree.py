#! /usr/bin/env python
import numpy as np

pid_chid = np.loadtxt('out.tree.txt')
pos_list = np.loadtxt('tmp.txt')
pid_list = set(pid_chid[:,0])
dim = pos_list.shape[1]

def write_pos(ofs, pos, dim):
    if dim == 2:
        ofs.write('%+1.7e %+1.7e\n' % (pos[0], pos[1]))
    elif dim == 3:
        ofs.write('%+1.7e %+1.7e %+1.7e\n' % (pos[0], pos[1], pos[2]))
    else:
        print "Truncating position dimension"
        ofs.write('%+1.7e %+1.7e %+1.7e\n' % (pos[0], pos[1], pos[2]))
    return

ofs = open('tmp.connect.txt', 'w')
for pid in pid_list:
    pos1 = pos_list[0]
    pos2 = pos_list[pid-1]
    #write_pos(ofs, pos1, dim)
    #write_pos(ofs, pos2, dim)
    #ofs.write('\n')

for i in range(pid_chid.shape[0]):
    pid = pid_chid[i][0]
    chid = pid_chid[i][1]
    pos1 = pos_list[pid-1]
    pos2 = pos_list[chid-1]
    write_pos(ofs, pos1, dim)
    write_pos(ofs, pos2, dim)
    ofs.write('nan nan nan\n\n')
ofs.close()








