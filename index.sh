#!/bin/sh

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ./dataset \
  --index indexes \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw