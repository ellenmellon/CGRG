# optional use
# used to process generation output file with ref included

import sys

with open(sys.argv[1]) as fin, open('db/dstc_ref.txt', 'w') as fr, open('db/dstc_gen.txt', 'w') as fg:
  for line in fin:
    src, tgt, gen = line.strip(' \r\n').split('\t')
    fr.write(tgt+'\n')
    fg.write(gen+'\n')
