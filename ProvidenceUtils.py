import os

def getFileIds(prov_path, by_child=False):
    """Given the root directory of the XML version of the Providence Corpus,
    returns all the fileids of the corpus to be used by ProvidenceCorpusReader.
    """
    subdirs = [subdir for subdir in os.listdir(prov_path)
               if not subdir.startswith('.')]
    
    if by_child:
        fileids = {}
        for subdir in subdirs:
            fileids[subdir] = [os.path.join(subdir, filename) 
                            for filename in os.listdir(os.path.join(prov_path, subdir))]
            fileids[subdir].sort()
            
    else:
        fileids = []
        for subdir in subdirs:
            fileids.extend([os.path.join(subdir, filename) 
                            for filename in os.listdir(os.path.join(prov_path, subdir))])
            
        fileids.sort()
    return fileids