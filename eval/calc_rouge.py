'''
This script is for calculating the rouge by 
giving reference file (grounding document file) 
and hypothesis file (generated response file)

'''

import sys
from rouge import Rouge

with open(sys.argv[1]) as f_r, open(sys.argv[2]) as f_h:
    r_lines = f_r.readlines()
    h_lines = f_h.readlines()
    rs = []
    hs = []
    for i in range(len(r_lines)):
        r = r_lines[i].strip(' \r\n')
        h = h_lines[i].split('\t')[-1].strip(' \r\n')
        rs += [r]
        hs += [h]
rouge = Rouge()
scores = rouge.get_scores(hs, rs, avg=True)
print(scores)

