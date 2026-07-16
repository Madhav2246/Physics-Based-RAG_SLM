import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        # Skip .git and some cache directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'node_modules', 'hf_cache']]
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

if __name__ == '__main__':
    list_files(r'd:\\S6\\NLP\\Physics_Based_RAG_SLM')
