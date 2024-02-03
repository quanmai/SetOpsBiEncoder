import jsonlines
from utils.utils import segment_query, extract_query


# read jsonl file, return original query and template
def main():
    with jsonlines.open('data/val.jsonl') as reader, open('output.txt', 'w') as writer:
        for obj in reader:
            query = obj['original_query']
            template = obj['metadata']['template']
            writer.write(f"Q: {query}\n")
            writer.write(f"T: {template}\n")
            writer.write(f"R: {extract_query(query, template)}\n")
            writer.write('-------------------------\n')

if __name__ == '__main__':
    main()