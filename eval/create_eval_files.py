import sys

with open(sys.argv[1]) as fin, open('ref.txt', 'w') as fr, open('pred.txt', 'w') as fg:
  for line in fin:
    src, tgt, gen = line.strip(' \r\n').split('\t')
    fr.write(tgt+'\n')
    fg.write(gen+'\n')
