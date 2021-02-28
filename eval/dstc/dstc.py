# author: Xiang Gao @ Microsoft Research, Oct 2018
# evaluate DSTC-task2 submissions. https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling

from util import *
from metrics import *
from tokenizers import *


def eval_one_system(submitted, ref, n_refs=1, n_lines=None, clean=False, vshuman=-1, PRINT=True):

	print('evaluating %s' % submitted)

	fld_out = submitted.replace('.txt','')
	if clean:
		fld_out += '_cleaned'
	path_hyp = submitted
	path_refs = [ref+str(i) for i in range(n_refs)]
	nist, bleu, meteor, entropy, div, avg_len = nlp_metrics(path_refs, path_hyp, fld_out, n_lines=n_lines)
	
	if n_lines is None:
		n_lines = len(open(path_hyp, encoding='utf-8').readlines())

	if PRINT:
		print('n_lines = '+str(n_lines))
		print('NIST = '+str(nist))
		print('BLEU = '+str(bleu))
		print('METEOR = '+str(meteor))
		print('entropy = '+str(entropy))
		print('diversity = ' + str(div))
		print('avg_len = '+str(avg_len))

	return [n_lines] + nist + bleu + [meteor] + entropy + div + [avg_len]


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('submitted')	# if 'all' or '*', eval all teams listed in dstc/teams.txt
	                                    # elif endswith '.txt', eval this single file
	                                    # else, eval all *.txt in folder `submitted_fld`
	parser.add_argument('--ref_file_prefix', '-rf', type=str, default='/data2/ellen/experiments/dstc/dstc_test_ref.txt.')      
	parser.add_argument('--clean', '-c', action='store_true')     # whether to clean ref and hyp before eval
	parser.add_argument('--n_lines', '-n', type=int, default=-1)  # eval all lines (default) or top n_lines (e.g., for fast debugging)
	parser.add_argument('--n_refs', '-r', type=int, default=3)    # number of references
	parser.add_argument('--vshuman', '-v', type=int, default='1') # when evaluating against human performance (N in refN.txt that should be removed) 
	                                                                      # in which case we need to remove human output from refs
	parser.add_argument('--teams', '-i', type=str, default='dstc/teams.txt')
	parser.add_argument('--report', '-o', type=str, default=None)
	args = parser.parse_args()
	print('Args: %s\n' % str(args), file=sys.stderr)

	if args.n_lines < 0:
		n_lines = None	# eval all lines
	else:
		n_lines = args.n_lines	# just eval top n_lines

	eval_one_system(args.submitted, args.ref_file_prefix, clean=args.clean, n_lines=n_lines, n_refs=args.n_refs, vshuman=args.vshuman)
