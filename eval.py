from utils.utils import normalize_query, extract_query_or, extract_query_for_embs
import csv, jsonlines
import faiss,pickle        
import numpy as np 
from tqdm import tqdm
from transformers import DPRQuestionEncoder,DPRQuestionEncoderTokenizer,BertModel,BertTokenizer
import torch
import unicodedata
import time
import transformers
from utils.example_utils import read_documents
transformers.logging.set_verbosity_error()
from sentence_transformers import SentenceTransformer
from dpr import get_query_embedding

def get_num_sub_queries(q):
    pass


def normalize(text):
    return unicodedata.normalize("NFD", text)

def further_search(q, d, k):
    # assert (q @ d.T).all() == (np.dot(q, d.T)).all()
    similarity = q @ d.T
    sorted_idx = np.argsort(similarity)[::-1][:k]
    return sorted_idx


def _eval(gold, prediction, idx_2_query: dict):
    # for i, g in enumerate(gold):
    #     assert idx_2_query[i] == g["query"], f"{idx_2_query[i]} != {g['query']}"

    p_vals = []
    r_vals = []
    f1_vals = []

    gold_documents = [(g["docs"]) for g in gold]
    gold_query = [(g["query"]) for g in gold]
    count = 0
    for i, (gold_docs, pred) in enumerate(zip(gold_documents, prediction)):
        predicted_docs = set(pred)
        gold_docs = set(gold_docs)
        tp = len(gold_docs.intersection(predicted_docs))
        fp = len(predicted_docs.difference(gold_docs))
        fn = len(gold_docs.difference(predicted_docs))
        if tp:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            count += 1
            # print(f'no tp for {gold_query[i]}')
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        p_vals.append(precision)
        r_vals.append(recall)
        f1_vals.append(f1)
    print(f'no tp for {count}/ {len(gold_query)} queries')
    return p_vals,r_vals,f1_vals

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_file",default="data/documents.jsonl")
    parser.add_argument("--test_file",default="data/test.jsonl")
    parser.add_argument("--encoding_batch_size",type=int,default=32)
    parser.add_argument("--num_shards",type=int,default=2)
    parser.add_argument("--num_docs",type=int,default=325505)
    parser.add_argument("--topk",type=int,default=10)
    parser.add_argument("--embedding_dir",required=True)
    parser.add_argument("--pretrained_model_path",required=True)
    parser.add_argument("--retriever", type=str, default="", choices=["multi_stage", "learned_query", ""])
    parser.add_argument("--sbert", action="store_true", default=False)
    args = parser.parse_args()

    # doc title to idx
    documents = read_documents(args.documents_file)
    doc_2_idx = {doc.title: idx for idx, doc in enumerate(documents)}
    idx_2_doc = {idx: doc.title for idx, doc in enumerate(documents)}

    ## load dataset
    queries,relevant_docs = [],[]
    if args.retriever == "multi_stage":
        multi_queries = []
        num_atom_queries = []
    if args.retriever == "learned_query":
        num_atom_queries = []
        templates = []
    num_queries = 0
    with jsonlines.open(args.test_file) as reader:
        for line in reader:
            num_queries += 1
            relevant_docs.append(
                {
                    "query": line['query'],
                    "docs": [doc_2_idx[doc] for doc in line['docs']]
                }
            )
            if args.retriever == "learned_query":
                queries.append(
                    extract_query_for_embs(line['original_query'])
                )
                num_atom_queries.append(len(queries[-1]))
                templates.append(line['metadata']['template'])
            else:
                queries.append(normalize_query(line['query']))

            if args.retriever == "multi_stage":
                multi_queries.append(
                    extract_query_or(
                        line['query'],
                        line['original_query'], 
                        line['metadata']['template'],
                    )
                )
                num_atom_queries.append(len(multi_queries[-1]))

    idx_2_query = {idx: query for idx, query in enumerate(queries)}
    
    queries = [
        queries[
            idx:idx+args.encoding_batch_size
        ] for idx in range(0,len(queries),args.encoding_batch_size)
    ]
    if args.retriever == "multi_stage":
        idx_2_multi_query = {idx: multi_query for idx, multi_query in enumerate(multi_queries)}
        multi_queries = [
            multi_queries[
                idx:idx+args.encoding_batch_size
            ] for idx in range(0,len(multi_queries),args.encoding_batch_size)
        ]
    if args.retriever == "learned_query":
        templates = [
            templates[
                idx:idx+args.encoding_batch_size
            ] for idx in range(0,len(templates),args.encoding_batch_size)
        ]
    
    # make faiss index
    embedding_dimension = 768 
    index = faiss.IndexFlatIP(embedding_dimension)
    for idx in tqdm(range(args.num_shards),desc='building index from embedding...'):
        data = np.load(f"{args.embedding_dir}/documents_shard_{idx}.npy")
        index.add(data)
    
    ## load query encoder
    if args.pretrained_model_path == 'facebook/dpr-question_encoder-single-nq-base':
        query_encoder = DPRQuestionEncoder.from_pretrained(args.pretrained_model_path)
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.pretrained_model_path)
    else:
        query_encoder = BertModel.from_pretrained(args.pretrained_model_path,add_pooling_layer=False)
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_encoder.to(device).eval()

    ## embed multi-queries
    if args.retriever == "multi_stage":
        multi_query_embeddings = []
        if args.sbert:
            atom_query_encoder = SentenceTransformer("multi-qa-mpnet-base-dot-v1").to(device).eval()
            # atom_query_encoder = SentenceTransformer(
            #     "multi-qa-mpnet-base-dot-v1-finetuned",
            # ).to(device).eval()
            for query in tqdm(multi_queries,desc='encoding multi queries...'):
                multi_query_embedding = []
                with torch.no_grad():
                    for single_q in query:
                        num_sub_queries = len(single_q)
                        single_qemb = [
                            torch.tensor(
                                atom_query_encoder.encode(single_q)
                            ).to(device)
                        ] + [torch.zeros(1, embedding_dimension).to(device)] * (3 - num_sub_queries)
                        single_qemb = torch.concatenate(single_qemb, dim=0)
                        assert single_qemb.shape == (3, embedding_dimension)
                        multi_query_embedding.append(single_qemb.cpu().detach().numpy())
                        assert len(multi_query_embedding) <= args.encoding_batch_size
                multi_query_embeddings.append(multi_query_embedding)
            multi_query_embeddings = np.concatenate(multi_query_embeddings,axis=0)
        else:
            for query in tqdm(multi_queries,desc='encoding multi queries...'):
                multi_query_embedding = []
                with torch.no_grad():
                    for single_q in query:
                        num_sub_queries = len(single_q)
                        single_qemb = [
                            query_encoder(
                                **tokenizer(
                                    q,
                                    max_length=256,
                                    truncation=True,
                                    padding='max_length',
                                    return_tensors='pt'
                                ).to(device)
                            ).last_hidden_state[:,0,:] for q in single_q
                        ] + [torch.zeros(1, embedding_dimension).to(device)] * (3 - num_sub_queries)
                        single_qemb = torch.concatenate(single_qemb, dim=0)
                        multi_query_embedding.append(single_qemb.cpu().detach().numpy())
                multi_query_embeddings.append(multi_query_embedding)
            multi_query_embeddings = np.concatenate(multi_query_embeddings,axis=0)
        print(f"{multi_query_embeddings.shape=}")

    # get query embeddings
    query_embeddings = []    
    if args.retriever == "learned_query":
        for query,template in tqdm(zip(queries,templates),desc='encoding queries...'):
            with torch.no_grad():
                qembs = [query_encoder(
                    **tokenizer(
                        q,
                        max_length=256,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    ).to(device)
                ).last_hidden_state[:,0,:] for q in query]
                query_embedding = [get_query_embedding(t, q).unsqueeze(0) for t, q in zip(template, qembs)]
                query_embedding = torch.cat(query_embedding, dim=0)
            query_embeddings.append(query_embedding.cpu().detach().numpy())
        query_embeddings = np.concatenate(query_embeddings,axis=0)
        print(f"{query_embeddings.shape=}")
    else:
        for query in tqdm(queries,desc='encoding queries...'):
            with torch.no_grad():
                query_embedding = query_encoder(
                    **tokenizer(
                    query,
                    max_length=256,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                    ).to(device)
                )
            if isinstance(query_encoder,DPRQuestionEncoder):
                query_embedding = query_embedding.pooler_output
            else:
                query_embedding = query_embedding.last_hidden_state[:,0,:]
            query_embeddings.append(query_embedding.cpu().detach().numpy())
        query_embeddings = np.concatenate(query_embeddings,axis=0)
        print(f"{query_embeddings.shape=}")
    

    ## retrieve top-k documents
    # print("searching index ", end=' ')
    start_time = time.time()
    _, I = index.search(query_embeddings, args.topk) # I: (num_queries, topk)

    p_vals,r_vals,f1_vals = _eval(relevant_docs, I, idx_2_query)
    print(f"Evaluation results for k = {args.topk}")
    print(f"Avg. Precision: {np.mean(p_vals)}")
    print(f"Avg. Recall: {np.mean(r_vals)}")
    print(f"Avg. F1: {np.mean(f1_vals)}")


    if args.retriever == "multi_stage":
        SINGLE_DOC = np.concatenate(
            [np.load(f"{args.embedding_dir}/documents_shard_{idx}.npy") for idx in range(args.num_shards)], 
            axis=0
        )

        # first search uses query embeddings
        multi_retriever_I = []
        _, I = index.search(query_embeddings, 4 * args.topk) # I: (num_queries, topk)
        for qidx, neighbors in enumerate(I):
            retrieved_docs = np.array([SINGLE_DOC[idx] for idx in neighbors])
            multi_queries_embs = multi_query_embeddings[qidx]
            n = num_atom_queries[qidx]
            for i in range(1):
                topk = args.topk# * (n - i)
                retrieved_doc_idx = further_search(
                    multi_queries_embs[i],
                    retrieved_docs,
                    topk,
                )
                if i == n - 1:
                    assert topk == args.topk
                retrieved_docs = retrieved_docs[retrieved_doc_idx]
            multi_retriever_I.append(neighbors[retrieved_doc_idx])
        
        p_vals,r_vals,f1_vals = _eval(relevant_docs, multi_retriever_I, idx_2_query)
        print(f"Evaluation results for multi-stage k = {args.topk}")
        print(f"Avg. Precision: {np.mean(p_vals)}")
        print(f"Avg. Recall: {np.mean(r_vals)}")
        print(f"Avg. F1: {np.mean(f1_vals)}")


        # TODO: using ONLY first atom query of multi query-embeddings
        # This should be the upper bound of multi-stage retrieval
        multi_retriever_I = []
        first_atom_query = multi_query_embeddings[:,0]
        _, I = index.search(first_atom_query, args.topk)
        
        p_vals,r_vals,f1_vals = _eval(relevant_docs, I, idx_2_query)
        print(f"Evaluation results for multi-stage, querying only first sub-query = {args.topk}")
        print(f"Avg. Precision: {np.mean(p_vals)}")
        print(f"Avg. Recall: {np.mean(r_vals)}")
        print(f"Avg. F1: {np.mean(f1_vals)}")

        # # TODO: first search using first atom query of multi query-embeddings
        # multi_retriever_I = []
        # first_atom_query = multi_query_embeddings[:,0]
        # _, I = index.search(first_atom_query, 3 * args.topk)
        # for qidx, neighbors in enumerate(I):
        #     n = num_atom_queries[qidx]
        #     if n == 1:
        #         multi_retriever_I.append(neighbors[:args.topk])
        #         continue
        #     multi_queries_embs = multi_query_embeddings[qidx]
        #     retrieved_docs = np.array([SINGLE_DOC[idx] for idx in neighbors])
        #     for i in range(1, n):
        #         topk = args.topk * (n - i)
        #         retrieved_doc_idx = further_search(
        #             multi_queries_embs[i],
        #             retrieved_docs,
        #             topk,
        #         )
        #         if i == n - 1:
        #             assert topk == args.topk
        #         retrieved_docs = retrieved_docs[retrieved_doc_idx]
        #     multi_retriever_I.append(neighbors[retrieved_doc_idx])
        
        # p_vals,r_vals,f1_vals = _eval(relevant_docs, multi_retriever_I, idx_2_query)
        # print(f"Evaluation results for multi-stage k = {args.topk}")
        # print(f"Avg. Precision: {np.mean(p_vals)}")
        # print(f"Avg. Recall: {np.mean(r_vals)}")
        # print(f"Avg. F1: {np.mean(f1_vals)}")






    

