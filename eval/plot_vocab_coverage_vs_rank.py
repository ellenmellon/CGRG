'''
Read files named as "vocab_coverage_vs_rank_{input_setting}.txt" to plot lines of 
vocab coverage versus rank for different input settings

'''

import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def read_line_from_file(fname, label):
  line = {'x': [], 'y': [], 'label': label}
  with open(fname) as fin:
    for l in fin.readlines()[1:]:
      if l.strip(' \r\n') == '':
        continue
      y = l.strip(' \r\n').split(',')[1]
      x = l.strip(' \r\n').split(',')[0]
      line['x'] += [float(x)]
      line['y'] += [float(y)]
  return line

def plot_lines(lines, xlabel, ylabel, title):
  for line in lines:
    plt.plot(line['x'][:10], line['y'][:10], label=line['label'])
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend(loc='upper right')
  plt.savefig(title)
  return

lines = []
folder = '.'

for fname in sys.argv[1:]:
  lines += [read_line_from_file(fname, fname.split('_')[-1].replace('.txt', ''))]
  if '/' in fname:
    folder = '/'.join(fname.split('/')[:-1])
plot_lines(lines, 'rank', 'vocab_coverage', '{}/vocab_coverage_vs_rank.png'.format(folder))
