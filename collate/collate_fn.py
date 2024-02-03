from utils.utils import normalize_query, extract_query_or, extract_query_for_embs
from dataclasses import dataclass
from torch import Tensor as T
import random
import torch


@dataclass
class BiencoderInput():
    query_ids: T
    query_attn_mask: T
    query_token_type_ids: T
    passage_ids: T
    passage_attn_mask: T
    passage_token_type_ids: T

def collate_multi_fn(samples,tokenizer,args,stage):
    # batch input for biencoder
    # each sample: 
    #  {1 query, 10 positive docs, 10 (hard) negative docs}
    #  take mean of the 10 positive docs as the positive doc
    # prepare query input
    queries = [normalize_query(x['query']) for x in samples] # list of queries
    query_inputs = tokenizer(
        queries,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    # prepare document input
    ## select the first positive document
    ## passage = title + document
    # [[pos_doc1_for_q1, pos_doc2_for_q1,...], [pos_doc1_for_q2, pos_doc2_for_q2], ...]
    positive_passages = [x['positive_docs'] for x in samples]
    positive_titles = [x['title'] for xx in positive_passages for x in xx]
    positive_docs = [x['text'] for xx in positive_passages for x in xx]

    pos_doc_idxs = []
    cur_label = 1
    ## random choose one negative document per one positive document
    negative_passages_sample = []
    for x in samples:
        num_pos_doc = min(len(x['positive_docs']), args.num_pos_ctx)
        num_neg_doc = min(len(x['negative_docs']), args.num_other_negative_ctx)
        num_hard_neg_doc = min(len(x['hard_negative_docs']), args.num_hard_negative_ctx)
        num_neg_passages_per_sample = min(
            num_pos_doc,
            num_neg_doc,
            num_hard_neg_doc,
        )

        hard_neg_ctxs = x['hard_negative_docs'][ :num_hard_neg_doc]
        neg_ctx = x['negative_docs'][ :num_neg_doc]
        negative_passages = hard_neg_ctxs + neg_ctx
        
        assert num_neg_passages_per_sample <= len(negative_passages), \
            f"num_neg_passages_per_doc: {num_neg_passages_per_sample}, len(negative_passages): {len(negative_passages)}"

        negative_passages = random.sample(
            negative_passages,
            num_neg_passages_per_sample,
        )
        negative_passages_sample.append(negative_passages)
        pos_doc_idxs.append(
            [cur_label] * num_pos_doc + [0] * (args.num_docs_per_sample - num_pos_doc)
        )
        cur_label += 1

    negative_titles = [x["title"] for xx in negative_passages_sample for x in xx]
    negative_docs = [x["text"] for xx in negative_passages_sample for x in xx]
    titles = positive_titles + negative_titles
    docs = positive_docs + negative_docs
    
    doc_inputs = tokenizer(
        titles,
        docs,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    # print(f"len: {[len(p) for p in pos_doc_idxs]}")
    biencoder_input = {
        'query_ids':query_inputs.input_ids,
        'query_attn_mask':query_inputs.attention_mask,
        'query_token_type_ids':query_inputs.token_type_ids,

        "passage_ids":doc_inputs.input_ids,
        "passage_attn_mask":doc_inputs.attention_mask,
        "passage_token_type_ids":doc_inputs.token_type_ids,     
    }
    print(pos_doc_idxs)
    labels = torch.cat(torch.tensor(pos_doc_idxs), dim=0)
    print(labels)
    print(labels.shape)
    return biencoder_input, labels


def collate_multi123_fn(samples,tokenizer,args,stage):
    # batch input for biencoder
    # each sample: 
    #  {1 query, 10 positive docs, 10 (hard) negative docs}
    #  take mean of the 10 positive docs as the positive doc
    # prepare query input
    queries = [normalize_query(x['query']) for x in samples] # list of queries
    query_inputs = tokenizer(
        queries,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    # prepare document input
    ## select the first positive document
    ## passage = title + document
    # [[pos_doc1_for_q1, pos_doc2_for_q1,...], [pos_doc1_for_q2, pos_doc2_for_q2], ...]
    positive_passages = [x['positive_docs'][: args.num_pos_ctx] for x in samples]
    positive_titles = [x['title'] for xx in positive_passages for x in xx]
    positive_docs = [x['text'] for xx in positive_passages for x in xx]

    ## random choose one negative document per one positive document
    negative_passages_sample = []
    for x in samples:
        num_pos_doc = min(len(x['positive_docs']), args.num_pos_ctx)
        num_neg_doc = min(len(x['negative_docs']), args.num_other_negative_ctx)
        num_hard_neg_doc = min(len(x['hard_negative_docs']), args.num_hard_negative_ctx)
        num_neg_passages_per_sample = min(
            num_pos_doc,
            num_neg_doc,
            num_hard_neg_doc,
        )

        hard_neg_ctxs = x['hard_negative_docs'][ :num_hard_neg_doc]
        neg_ctx = x['negative_docs'][ :num_neg_doc]
        negative_passages = hard_neg_ctxs + neg_ctx
        
        assert num_neg_passages_per_sample <= len(negative_passages), \
            f"num_neg_passages_per_doc: {num_neg_passages_per_sample}, len(negative_passages): {len(negative_passages)}"

        negative_passages = random.sample(
            negative_passages,
            args.num_hard_negative_ctx,
        )
        negative_passages_sample.append(negative_passages)

    negative_titles = [x["title"] for xx in negative_passages_sample for x in xx]
    negative_docs = [x["text"] for xx in negative_passages_sample for x in xx]
    # titles = positive_titles + negative_titles
    # docs = positive_docs + negative_docs
    
    neg_inputs = tokenizer(
        negative_titles,
        negative_docs,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    pos_inputs = tokenizer(
        positive_titles,
        positive_docs,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    # print(f"len: {[len(p) for p in pos_doc_idxs]}")
    biencoder_input = {
        'query_ids':query_inputs.input_ids,
        'query_attn_mask':query_inputs.attention_mask,
        'query_token_type_ids':query_inputs.token_type_ids,

        "passage_ids":pos_inputs.input_ids,
        "passage_attn_mask":pos_inputs.attention_mask,
        "passage_token_type_ids":pos_inputs.token_type_ids,

        "neg_passage_ids":neg_inputs.input_ids,
        "neg_passage_attn_mask":neg_inputs.attention_mask,
        "neg_passage_token_type_ids":neg_inputs.token_type_ids,     

    }

    return biencoder_input


def collate_extract_query_fn(samples,tokenizer,args,stage):
    # batch input for biencoder
    # each sample: 
    #  {1 query, 10 positive docs, 10 (hard) negative docs}
    #  take mean of the 10 positive docs as the positive doc
    # prepare query input
    queries = [normalize_query(x['query']) for x in samples] # list of queries
    original_queries = [x['original_query'] for x in samples]
    templates = [x['template'] for x in samples]
    query_extracted = [extract_query_for_embs(oq) for oq in original_queries]

    query_inputs = [tokenizer(
        qe,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ) for qe in query_extracted]
    
    # prepare document input
    ## select the first positive document
    ## passage = title + document
    # [[pos_doc1_for_q1, pos_doc2_for_q1,...], [pos_doc1_for_q2, pos_doc2_for_q2], ...]
    positive_passages = [x['positive_docs'][: args.num_pos_ctx] for x in samples]
    positive_titles = [x['title'] for xx in positive_passages for x in xx]
    positive_docs = [x['text'] for xx in positive_passages for x in xx]

    ## random choose one negative document per one positive document
    negative_passages_sample = []
    for x in samples:
        num_pos_doc = min(len(x['positive_docs']), args.num_pos_ctx)
        num_neg_doc = min(len(x['negative_docs']), args.num_other_negative_ctx)
        num_hard_neg_doc = min(len(x['hard_negative_docs']), args.num_hard_negative_ctx)
        num_neg_passages_per_sample = min(
            num_pos_doc,
            num_neg_doc,
            num_hard_neg_doc,
        )

        hard_neg_ctxs = x['hard_negative_docs'][ :num_hard_neg_doc]
        neg_ctx = x['negative_docs'][ :num_neg_doc]
        negative_passages = hard_neg_ctxs + neg_ctx
        
        assert num_neg_passages_per_sample <= len(negative_passages), \
            f"num_neg_passages_per_doc: {num_neg_passages_per_sample}, len(negative_passages): {len(negative_passages)}"

        negative_passages = random.sample(
            negative_passages,
            args.num_hard_negative_ctx,
        )
        negative_passages_sample.append(negative_passages)

    negative_titles = [x["title"] for xx in negative_passages_sample for x in xx]
    negative_docs = [x["text"] for xx in negative_passages_sample for x in xx]
    # titles = positive_titles + negative_titles
    # docs = positive_docs + negative_docs
    
    neg_inputs = tokenizer(
        negative_titles,
        negative_docs,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    pos_inputs = tokenizer(
        positive_titles,
        positive_docs,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    # print(f"len: {[len(p) for p in pos_doc_idxs]}")
    biencoder_input = {
        'query_ids':[q.input_ids for q in query_inputs],
        'query_attn_mask':[q.attention_mask for q in query_inputs],
        'query_token_type_ids':[q.token_type_ids for q in query_inputs],

        "passage_ids":pos_inputs.input_ids,
        "passage_attn_mask":pos_inputs.attention_mask,
        "passage_token_type_ids":pos_inputs.token_type_ids,

        "neg_passage_ids":neg_inputs.input_ids,
        "neg_passage_attn_mask":neg_inputs.attention_mask,
        "neg_passage_token_type_ids":neg_inputs.token_type_ids,     

        "templates": templates,
    }

    return biencoder_input