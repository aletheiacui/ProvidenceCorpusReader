import ProvidenceUtils
from ProvidenceCorpusReader import ProvidenceCorpusReader
import os

prov_path = "Providence" # the root directory of the Providence Corpus
fileids = ProvidenceUtils.getFileIds(prov_path, True)

prov = ProvidenceCorpusReader(prov_path, fileids)

csv_dir = 'ProvidenceCSV'

for child in fileids.keys():
    prov = ProvidenceCorpusReader(prov_path, fileids[child])

    child_file = open(os.path.join(csv_dir, '%s_child.csv' % child), 'w', encoding='utf-8')
    child_file.write('child,fileid,age,orthography,stem,model,actual,pos,start_time,end_time\n')
    parent_file = open(os.path.join(csv_dir, '%s_parentt.csv' % child), 'w',encoding='utf-8')
    parent_file.write('child,fileid,age,orthography,stem,phonemes,pos,start_time,end_time\n')


    child_words = prov.words_info(fileids[child],speaker=["CHI"])
    mot_words = prov.words_info(fileids[child],speaker=["MOT"])

    for word in mot_words:
        parent_file.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % (child,word.fileid, word.age, word.orthography, 
                                                      word.stem, '.'.join(word.transcription), 
                                                      word.pos, ','.join(str(t) for t in word.media_times)))
    for word in child_words:
        child_file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (child,word.fileid, word.age,word.orthography, word.stem, 
                                                        '.'.join(word.transcription.model), 
                                                        '.'.join(word.transcription.actual), 
                                                        word.pos, ','.join(str(t) for t in word.media_times)))


print('Done.')
