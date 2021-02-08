'''
* @Date: 2019-04-05 16:50:50 
'''
import torch
import torch.nn.functional as F


EOS_ID = 50256
EXCLUDE = [66, 220, 29, 1279] # c, [space], >, [space]<
TOPK = 8

def generate_next_token(model, input_ids, position_ids=None, token_type_ids=None, attn_masks=None, prev=None, temperature=1, top_k=0, sample=False, past=None, bonus_indices=None, bonus=0.0, beam_size=1):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    with torch.no_grad():
        if not past:
            # print(prev.device)
            # print(token_type_ids.device)
            hidden_states, past = model.transformer(prev, position_ids, token_type_ids, attn_masks, past=past)
        else:
            # print(past)
            # print(past.device)            
            hidden_states, past = model.transformer(prev, position_ids, token_type_ids, attn_masks, past=past)    # position embedding might be wrong?
        logits = model.lm_head(hidden_states)
        logits = logits[:, -1, :] / temperature
        if bonus_indices is not None or sample:
            logits = top_k_logits(logits, k=top_k)
        else:
            logits = top_k_logits(logits, k=beam_size)
        log_probs = F.softmax(logits, dim=-1)

        if bonus_indices is not None:
            # now it only work when shape[0] = 1
            _, top_ind = torch.topk(log_probs, k=top_k, dim=-1)
            extra_score = torch.zeros(log_probs.shape).to(log_probs.device)
            #print(bonus_indices)
            for b in range(top_ind.size(0)):
                cur_bonus_indices = torch.LongTensor(list(set(bonus_indices[b].view(-1).tolist()).intersection(set(top_ind[b].view(-1).tolist())))).to(bonus_indices[b].device)
                extra_score[b].index_fill_(0, cur_bonus_indices, bonus)
            #print(bonus_indices)
            #print(log_probs)
            #print(torch.index_select(extra_score, 1, bonus_indices.view(-1)))
            log_probs = log_probs + extra_score
        if sample:
            prev = torch.multinomial(log_probs, num_samples=1)
            # import pdb; pdb.set_trace()
            return prev, torch.gather(log_probs, 1, prev), past
        else:
            # this is where the beam search bug begins : no matter what beam size you take, it only take the top 1 (= beam size always 1)
            log_probs_sel, prev = torch.topk(log_probs, k=beam_size, dim=-1)
            #print(log_probs_sel, torch.sum(log_probs), torch.topk(log_probs, k=5, dim=-1))
            return prev, log_probs_sel, past

def get_bonus_indices(prev_path, bonus_indices, device):
    bonus_indices = list(set(bonus_indices).difference(set(prev_path)).difference(EXCLUDE))
    return torch.LongTensor(bonus_indices).to(device)


def generate_sequence(model, input_ids, position_ids=None, token_type_ids=None, attn_masks=None, start_token=None, temperature=1, top_k=0, length = 20, sample=False, use_bonus=False, bonus=0.0, enc=None, past=None, device='cuda'):
    output = input_ids.new_zeros([input_ids.size(0),0])
    # tgt_start_index = torch.sum(input_ids != 0, dim = 1)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    prev = input_ids
    if use_bonus:
        bonus_indices_tensor = []
        for b in range(input_ids.size(0)):
            in_string = enc.decode(input_ids[b:b+1].view(-1).tolist())
            if '<s>' in in_string:
                in_string = '<c>'.join(in_string.split('<c>')[:-1])
                bonus_indices = enc.encode(in_string.split('<s>')[-1])
            else:
                bonus_indices = []
            bonus_indices_tensor.append(bonus_indices)
    for i in range(length):
        if use_bonus:
            new_bonus_indices = []
            for b in range(input_ids.size(0)):
                new_bonus_indices.append(get_bonus_indices(output[b].tolist(), bonus_indices_tensor[b], device).view(1, -1))
        else:
            new_bonus_indices = None
        prev, log_probs, past = generate_next_token(model, input_ids, position_ids, token_type_ids, attn_masks, prev, temperature, top_k, sample, past, new_bonus_indices, bonus, 1)
        output = torch.cat((output, prev), dim=1)
        position_ids = position_ids[:,-1:]+1
        token_type_ids = token_type_ids[:,-1:]
        # TODO

        attn_masks = torch.cat((attn_masks, attn_masks[:,-1:,:]), dim=1)
        tmp = torch.zeros(attn_masks.shape[0], attn_masks.shape[1], 1).to(device)
        tmp[:,-1,-1] = 1
        attn_masks = torch.cat((attn_masks, tmp), dim=2)
    return output


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for s in sentence:
        if s in remove_id:
            continue
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent


def torch_vec_to_str(x, tokenizer):
    xx = x.cpu().numpy()
    decode_str = [tokenizer.decode(cut_seq_to_eos(s)).encode('utf-8').decode('utf-8') for s in xx]
    return decode_str


###########################################################################
# Beam search based on ottokart/beam_search
###########################################################################
class Node(object):
    def __init__(self, parent, state, value, cost, extras, path):
        super(Node, self).__init__()
        self.value = value
        self.parent = parent  # parent Node, None for root
        self.state = state
        self.cum_cost = parent.cum_cost + cost if parent else cost
        self.length = 1 if parent is None else parent.length + 1
        self.extras = extras  # can hold, for example, attention weights
        self._sequence = None
        self.path = path

    def __repr__(self):
        return f'value = {self.value}, parent = {self.parent.value}, cost = {self.cum_cost}'


#def get_bonus_indices(prev_path, bonus_indices, device):
#    bonus_indices = list(set(bonus_indices).difference(set(prev_path)).difference(EXCLUDE))
#    return torch.LongTensor(bonus_indices).to(device)

def beam_search_naive(model, input_ids, position_ids=None, token_type_ids=None, attn_masks=None, length=20, beam_width=3, device='cuda', use_bonus=False, bonus=0.0, enc=None):
    """
    currently it does NOT support batch parabllel
    """

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    all_decode, all_decode_losses = [], []
    for b in range(input_ids.shape[0]):
        next_fringe = [Node(parent=None, state=None, value=-1, cost=0.0, extras=input_ids[b:b+1], path=[])]
        results = []
        token_type_sliced = None if token_type_ids is None else token_type_ids[b:b+1]
        position_sliced = None if position_ids is None else position_ids[b:b+1]
        attn_masks_sliced = None if attn_masks is None else attn_masks[b:b+1]

        # bonus indices
        if use_bonus:
            #mask = token_type_ids[b:b+1].ge(21) & token_type_ids[b:b+1].le(30)
            in_string = enc.decode(input_ids[b:b+1].view(-1).tolist())
            if '<s>' in in_string:
                in_string = '<c>'.join(in_string.split('<c>')[:-1])
                bonus_indices = enc.encode(in_string.split('<s>')[-1])
            else:
                bonus_indices = []
            #bonus_indices = enc.encode(' ' + enc.decode(torch.masked_select(input_ids[b:b+1].view(-1), mask.view(-1)).tolist()))
        for i in range(length):
            fringe, all_prev, all_probs, all_past = [], torch.Tensor(0).long().to(device), [], []
            for n in next_fringe:
                if n.value == EOS_ID:
                    results.append(n)
                else:
                    fringe.extend([n]*beam_width)

                if not fringe:
                    break
                if use_bonus:
                    new_bonus_indices = get_bonus_indices(n.path, bonus_indices, device).view(1, -1)
                else:
                    new_bonus_indices = None
                # NOTICE that beam_width is not used, thus this is basically greedy search 
                prev, probs, past = generate_next_token(model, input_ids[b:b+1], position_sliced, token_type_sliced, attn_masks_sliced,
                                                        n.extras, 1, TOPK, False, n.state, new_bonus_indices, bonus, beam_width)
                if position_sliced is not None:
                    position_sliced = position_sliced[:,-1:]+1
                if token_type_sliced is not None:
                    token_type_sliced = token_type_sliced[:,-1:]
                if attn_masks_sliced is not None:
                    # TODO
                    attn_masks_sliced = torch.cat((attn_masks_sliced, attn_masks_sliced[:,-1:,:]), dim=1)
                    tmp = torch.zeros(attn_masks_sliced.shape[0], attn_masks_sliced.shape[1], 1).to(device)
                    tmp[:,-1,-1] = 1
                    attn_masks_sliced = torch.cat((attn_masks_sliced, tmp), dim=2)


                log_probs = torch.log(probs)[0]
                all_prev = torch.cat((all_prev, prev[0]))
                all_probs.extend(log_probs.tolist())
                all_past.extend([past]*len(log_probs))

            next_fringe = []
            for prev, log_probs, past, n in zip(all_prev, all_probs, all_past, fringe):
                new_node = Node(parent=n, state=past, value=prev.item(), cost=log_probs, extras=prev.expand(1, 1), path=n.path+[prev.item()])
                next_fringe.append(new_node)

            next_fringe = sorted(next_fringe, key=lambda n: n.cum_cost, reverse=True)[:beam_width] # may move this into loop to save memory

        results.extend(next_fringe)
        results.sort(key=lambda n : n.cum_cost, reverse=True)
        best_result = results[0]
        #print(enc.decode(best_result.path), best_result.path)
        #print(enc.decode(bonus_indices), bonus_indices)
        #print(enc.decode(get_bonus_indices(best_result.path, bonus_indices, device).tolist()), get_bonus_indices(best_result.path, bonus_indices, device).tolist())
        decode, decode_loss = [], []
        while best_result.value != -1:
            decode.append(best_result.value)
            decode_loss.append(best_result.cum_cost)
            best_result = best_result.parent
        decode, decode_loss = decode[::-1], decode_loss[::-1]
        all_decode.append(decode)
        all_decode_losses.append(decode_loss)

    output = torch.nn.utils.rnn.pad_sequence([torch.tensor(f, dtype=torch.long) for f in all_decode],
                                             batch_first=True, padding_value=EOS_ID)
    return output

