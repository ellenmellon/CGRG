'''
This is also an optional use. 

Use it only if you want to do a second round evaluation for masking out 
the constraint words because you think it is sort of cheating.

run `python replace_constraits.py input_file (in decoded text) generated_response_file`

It outputs a file, which puts some placeholder for constraints in the generated responses.
The output file will be at the same place as the generation file, with additional suffix to be '.hide_cstr' 

'''


import sys

# get constraints

with open(sys.argv[1]) as f_src, open(sys.argv[2]) as f_pred, open(sys.argv[2]+'.hide_cstr', 'w') as f_pred_new:
    src_lines = f_src.readlines()
    pred_lines = f_pred.readlines()
    n_lines = len(src_lines)
    for i in range(n_lines):
        src_line = src_lines[i]
        pred_line = pred_lines[i]
        src = src_line.strip(' \r\n')
        pred = pred_line.strip(' \r\n').split('\t')[-1]

        doc_and_cstr = src.split('<|endoftext|>')[-1].strip()
        if len(doc_and_cstr) == 0:
            cstr_tokens = set([])
        else:
            cstr_tokens = set(doc_and_cstr.split('<s>')[-1].strip().split())
        pred_tokens = pred.strip().split()
        new_pred_tokens = []
        for p_t in pred_tokens:
            if p_t not in cstr_tokens:
            #    new_pred_tokens += ['<CSTR>']
            #else:
                new_pred_tokens += [p_t]
        f_pred_new.write(' '.join(new_pred_tokens)+'\n')


