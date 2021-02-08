"""
data loader taking python shelve DB
NOTE: only works for training
"""
import gzip
import json
import math
import random
import shelve

import torch
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from gpt2_training.train_utils import InputFeatures_v4



class BucketSampler(Sampler):
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)


class GPT2FeatureDataset(Dataset):
    def __init__(self, features, max_len=None):
        self.features = features
        self.max_len = max_len  # this max_len do truncate

    def __getitem__(self, i):
        feat_dict = self.features[i]
        if self.max_len is not None and feat_dict['input_len'] > self.max_len:
            # tuncate on the left side (context)
            feat_dict['input_ids'] = feat_dict['input_ids'][-self.max_len:]
            feat_dict['position_ids'] = feat_dict['position_ids'][
                -self.max_len:]  # FIXME what is the correct behavior here?
            feat_dict['token_type_ids'] = feat_dict['token_type_ids'][
                -self.max_len:]
            feat_dict['lm_labels'] = feat_dict['lm_labels'][-self.max_len:]
            feat_dict['attn_masks'] = feat_dict['attn_masks'][-self.max_len:]
        # import pdb; pdb.set_trace()
        feat = InputFeatures_v4(**feat_dict)
        return feat

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features):
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        position_ids = pad_sequence([torch.tensor(f.position_ids,
                                                  dtype=torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids,
                                                    dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)

        attn_masks = torch.zeros(input_ids.shape[0], input_ids.shape[1], input_ids.shape[1])
        for i in range(len(features)):
            f = features[i]
            # seq_l * seq_l
            f_attn_masks = pad_sequence([torch.tensor(mask, dtype=torch.float)
                               for mask in f.attn_masks],
                              batch_first=True, padding_value=0)
            attn_masks[i, :f_attn_masks.shape[0], :f_attn_masks.shape[0]] = f_attn_masks

        return (input_ids, position_ids, token_type_ids, labels, attn_masks)


class BucketingDataLoader(object):
    def __init__(self, db_name, batch_size, max_seq_len,
                 bucket=100, shuffle=True):
        self.db = shelve.open(f'{db_name}/db', 'r')
        self.batch_size = batch_size
        self.max_len = max_seq_len
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle

    def _get_keys(self):
        keys = list(self.db.keys())
        return keys

    def __iter__(self):
        keys = self._get_keys()
        if self.shuffle:
            random.shuffle(keys)
        for key in keys:
            chunk = json.loads(gzip.decompress(self.db[key]).decode('utf-8'))
            # discard long examples
            trunc_chunk = []
            lens = []
            for feat in chunk:
                if feat['input_len'] > self.max_len:
                    continue
                trunc_chunk.append(feat)
                lens.append(feat['input_len'])

            dataset = GPT2FeatureDataset(trunc_chunk, self.max_len)
            sampler = BucketSampler(lens, self.bucket_size, self.batch_size,
                                    droplast=True, shuffle=self.shuffle)
            loader = DataLoader(dataset, batch_sampler=sampler,
                                num_workers=0,  # can test multi-worker
                                collate_fn=GPT2FeatureDataset.collate)
            yield from loader

    def __len__(self):
        raise NotImplementedError()

    def __del__(self):
        self.db.close()


class DistributedBucketingDataLoader(BucketingDataLoader):
    def __init__(self, rank, num_replica, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.num_replica = num_replica

    def _get_keys(self):
        keys = list(self.db.keys())[self.rank::self.num_replica]
        return keys


def test(db_path):
    import ipdb
    device = torch.device('cuda')
    loader = BucketingDataLoader(db_path, 32, 256)
    for batch in loader:
        ipdb.set_trace()


if __name__ == '__main__':
    import sys
    db_path = sys.argv[1]
    test(db_path)
