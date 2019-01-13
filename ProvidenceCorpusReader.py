"""
Corpus reader for the XML version of the Providence corpus. This parser
is based on and extends CHILDESCorpusReader from NLTK:
    http://www.nltk.org/_modules/nltk/corpus/reader/childes.html
This parser was created to facilitate phonetic or phonological studies
of language acquisition using the Providence Corpus. It has additional 
functionalities, including 
    -extract the start and end times of each utterance in the corresponding 
        media file
    -parse and return the model pronunciation and actual pronunciation
        by the children
    -return the phonemic transcription of parental speech in Arpabet or IPA
"""

from __future__ import print_function, division

from collections import namedtuple

import nltk
from nltk.corpus.reader.childes import CHILDESCorpusReader
from nltk.util import LazyMap, LazyConcatenation
from six import string_types

import xml.etree.ElementTree as ElementTree

__docformat__ = 'epytext en'

NS = 'http://www.talkbank.org/ns/talkbank'

cDigraphs = {
    'ɪ': ['a', 'ɑ', 'o', 'e', 'ɔ'],
    'ʊ': ['a', 'ɑ', 'o', 'ɔ'], 
    'ʃ': ['t'],
    'ʒ': ['d'],
    'r': ['ʌ','ə', 'ɜ˞'],
    'ɹ': ['ʌ','ə'],
    '̩': ['n', 'l', 'm'],
    '̃': ['i', 'ɪ', 'e', 'ɛ', 'æ', 'ʌ', 'ə', 'a', 'ɑ', 'ɔ', 'o', 'ʊ', 'u'],
    '˞': ['ɜ'],
}

cReplacement = {
    'tʃ': 'ʧ',
    'əl': 'l̩',
    'ʌɹ': 'ɚ',
    'əɹ': 'ɚ',
    'ɜ˞r': 'ɚ',
    'ɜ˞': 'ɚ',
    'ər': 'ɚ',
    'oɪ':'ɔɪ',
    'oʊ': 'o',
    'dʒ': 'ʤ',
    'a': 'ɑ'
}

cArpaToIPA = {
    'AO': 'ɔ',
    'AA': 'ɑ',
    'IY': 'i',
    'UW': 'u',
    'EH': 'ɛ',
    'IH': 'ɪ',
    'UH': 'ʊ',
    'AH': 'ʌ',
    'AH0': 'ə',
    'AE': 'æ',
    'AX': 'ə',
    'EY': 'eɪ',
    'AY': 'aɪ',
    'OW': 'oʊ',
    'OW': 'o',
    'AW': 'aʊ',
    'OY': 'ɔɪ',
    'ER': 'ɚ',
    'P': 'p',
    'B': 'b',
    'T': 't',
    'D': 'd',
    'K': 'k',
    'G': 'g',
    'CH': 'ʧ',
    'JH': 'ʤ',
    'F': 'f',
    'V': 'v',
    'TH': 'θ',
    'DH': 'ð',
    'S': 's',
    'Z': 'z',
    'SH': 'ʃ',
    'ZH': 'ʒ',
    'HH': 'h',
    'M': 'm',
    'N': 'n',
    'NG': 'ŋ',
    'L': 'l',
    'R': 'r',
    'W': 'w',
    'Y': 'j',
    'Q': 'ʔ'
}

cVowels = {'ɔ', 'ɑ', 'i', 'u', 'ɛ', 'ɪ', 'ʊ', 'ʌ', 'ə', 'æ', 'ə', 'e', 'eɪ', 
           'aɪ', 'oʊ', 'o', 'aʊ', 'ɔɪ', 'ɚ'}

cConsonants = {'P', 'B', 'T', 'D', 'K', 'G', 'CH', 'JH', 'F', 'V', 'TH', 'DH', 
               'S', 'Z', 'SH', 'ZH', 'HH', 'M', 'N', 'NG', 'L', 'R',' Q'}


# Download the CMU Dictionary
try:
    cmu = nltk.corpus.cmudict.dict()
except LookupError:
    print('CMU Dictionary not found. Downloading the CMU Dictionary.')
    nltk.download('cmudict')
    cmu = nltk.corpus.cmudict.dict()


class ProvidenceCorpusReader(CHILDESCorpusReader):
    
    def __init__(self, root, fileids, lazy=True):

        CHILDESCorpusReader.__init__(self, root, fileids, lazy)
        
        # use namedtuple to more easily access information needed
        self.word_info = namedtuple('word_info', ['fileid', 'age','orthography', 'stem', 'transcription', 
                                       'pos', 'media_times'])
        self.child_transcription = namedtuple('transcription', ['model', 'actual'])
     
     
    def words_info(self, fileids=None, speaker='ALL', sent=False, stem=True,
            relation=False, strip_space=True, replace=False, pos=True,
             utt_times=True, transcription=True, word_info=True):
        """
        :return: the given file(s) as a list of words with their relevant phonological
        and linguistic annotations from the corpus, 
            encoded as namedtuples
            ``namedtuple(fileid, age,orthography, stem, transcription, pos, media_times)``.
        :rtype: list(namedtuple(str, str, str, str, namedtuple, str, tuple))
        """
        if not self._lazy:
            return [self._get_words(fileid, speaker, False, stem, relation,
                pos, strip_space, replace, utt_times =True, transcription=True, word_info = True) 
                                    for fileid in self.abspaths(fileids)]

        get_words = lambda fileid: self._get_words(fileid, speaker, False, stem, relation,
            pos, strip_space, replace, utt_times=True, transcription=True, word_info = True)
#         print(word_info)
        return LazyConcatenation(LazyMap(get_words, self.abspaths(fileids)))
     
      
    def words_times(self, fileids=None, speaker='ALL', sent=True, stem=False,
            relation=False, strip_space=True, replace=False, pos=False):
        """
        :return: the given file(s) as a list of words and the position of their
            corresponding sentence in the audio file, 
            encoded as tuples
            ``(word, (start_time, end_time))``.
        :rtype: list(tuple(str, tuple))
        """
        if not self._lazy:
            return [self._get_words(fileid, speaker, sent, stem, relation,
                True, strip_space, replace, utt_times=True) 
                                    for fileid in self.abspaths(fileids)]

        get_words = lambda fileid: self._get_words(fileid, speaker, sent, stem, relation,
            True, strip_space, replace, utt_times=True)
        return LazyConcatenation(LazyMap(get_words, self.abspaths(fileids)))
    
    
    def words_transcription(self, fileids=None, speaker='ALL', sent=True, stem=False,
            relation=False, strip_space=True, replace=False, pos=False):
        """
        :return: the given file(s) as a list of words and their phonemic transription, 
            encoded as tuples
            ``(word, transcription)``.
            If the speaker is a child, the transcription is returned as a namedtuple, 
            in the form of
            ``(model_pronunciation, actual_pronunciation)``
        :rtype: list(tuple(str, list))
        """
        if not self._lazy:
            return [self._get_words(fileid, speaker, sent, stem, relation,
                pos, strip_space, replace, transcription=True) 
                                    for fileid in self.abspaths(fileids)]

        get_words = lambda fileid: self._get_words(fileid, speaker, sent, stem, relation,
            pos, strip_space, replace, transcription=True)
        return LazyConcatenation(LazyMap(get_words, self.abspaths(fileids)))
    
 
    def _get_words(self, fileid, speaker, sent, stem, relation, pos,
                   strip_space, replace, ipa=True, 
                   transcription = False, 
                   utt_times = False,
                   word_info = False):
        
        # Modified from the _get_words() function from CHILDESCorpusReader
        # with added functionalities
        
        print('Processing file ', fileid)
        filename = str(fileid).rsplit('/', maxsplit=1)[-1]
        age_month = self.age(fileid, month=True)[0]
        
        # processing each xml doc
        results = [] 
        
        # ensure we have a list of speakers
        if isinstance(speaker, string_types) and speaker != 'ALL':  
            speaker = [speaker]
            
        xmldoc = ElementTree.parse(fileid).getroot()
 
        # iterates through each sentence <u></u>
        for xmlsent in xmldoc.findall('.//{%s}u' % NS):
            
            sents = []
            # get the media times for each sentence
            # has to be done here since the media tags are attributes
            # of utterances
            if utt_times:
                media_times = self._get_media_times(xmlsent)
                
            # select speakers
            if speaker == 'ALL' or xmlsent.get('who') in speaker:
                # iterates through all word elements <w></w> in an utterance
                
                if speaker == ['CHI']:
                    xmlwords = xmlsent.findall('.//{%s}pg' % NS)
                else:
                    xmlwords = xmlsent.findall('.//{%s}w' % NS)

                for xmlword in xmlwords:
                    
                    suffixStem = None
                    
                    if replace:
                        xmlword = self._get_replaced_words(xmlsent)
                        
                    word = self._get_word_text(xmlword, strip_space).lower()
                    
                    # skip entries not found in the CMU dictionary
                    if word not in cmu:
                        continue
                        
                    # save the text of the word because the word can be
                    # changed to stem+suffix later
                    orthography = word
                    # stem
                    if relation or stem:
                        stem = self._get_word_stem(xmlword, word).lower()
                        word = word
                            
                    # pos
                    if relation or pos:
                        pos_tag = self._get_word_pos(xmlword)
                        word = (word, pos_tag)
                        
                    # relation
                    if relation:
                        word, suffixStem = self._get_word_relation(xmlword, word, suffixStem)
                    
                    # get transcription since parental speech
                    # aren't transcribed
                    if transcription:
                        transcription = self._get_transcription(xmlword, orthography, speaker, ipa)
                        word = (orthography, transcription)
                    
                    if utt_times:
                        word = (orthography, media_times)
                    
                    
                    if word_info:
                        # Discard word if there's no transcription. Not ideal, so
                        # future development should let the user decide what to do when the
                        # transcription isn't found.
                        try:
                            word = self.word_info(filename, age_month, orthography, stem, transcription, 
                                                  pos_tag, media_times)
                        except IndexError:
                            continue
                      
                    sents.append(word)
                    
                if sent or relation:
                    results.append(sents)
                else:
                    results.extend(sents)
                    
        return results
    
    
    def _get_media_times(self, xmlsent):
        """
        Finds the location of the sentence within the audio file.
        """
        sent_times = xmlsent.find('.//{%s}media' % (NS))
        try:
            media_times = (float(sent_times.attrib['start']), float(sent_times.attrib['end']))
        except AttributeError:
            media_times = (0.0, 0.0)
        return media_times
    
    
    def _get_replaced_word(self, xmlsent):

        if xmlsent.find('.//{%s}w/{%s}replacement'
                                            % (NS, NS)):
            xmlword = xmlsent.find('.//{%s}w/{%s}replacement/{%s}w'
                                   % (NS, NS, NS))
        elif xmlsent.find('.//{%s}w/{%s}wk' % (NS, NS)):
            xmlword = xmlsent.find('.//{%s}w/{%s}wk' % (NS, NS))
        
        return xmlword
    
    
    def _get_word_text(self, xmlword, strip_space):
        """
        Get the text of the word
        """
        # If the speaker is a 'CHI', get the child element <w>
        if xmlword.tag == '{%s}pg' % NS:
            xmlword = xmlword.find('.//{%s}w' % NS)
        if xmlword.text:
            word = xmlword.text
        else:
            word = ''
             
        # strip tailing space
        if strip_space:
            word = word.strip()
            
        return word
    
    
    def _get_word_stem(self, xmlword, word):
                        
        try:
            xmlstem = xmlword.find('.//{%s}stem' % NS)
            word = xmlstem.text
        except AttributeError:
            pass
        
        # if there is an inflection
        try:
            xmlinfl = xmlword.find('.//{%s}mor/{%s}mw/{%s}mk'
                                   % (NS, NS, NS))
            word += '-' + xmlinfl.text
        except:
            pass
        
        # if there is a suffix
        try:
            xmlsuffix = xmlword.find('.//{%s}mor/{%s}mor-post/{%s}mw/{%s}stem'
                                     % (NS, NS, NS, NS))
            suffixStem = xmlsuffix.text
        except AttributeError:
            suffixStem = ""
            
        if suffixStem:
            word += "~" + suffixStem
            
        return word
    
    
    def _get_word_pos(self, xmlword):
        suffixTag = None

        try:
            xmlpos = xmlword.findall(".//{%s}c" % NS)
            xmlpos2 = xmlword.findall(".//{%s}s" % NS)
            if xmlpos2 != []:
                tag = xmlpos[0].text + ":" + xmlpos2[0].text
            else:
                tag = xmlpos[0].text
        except (AttributeError, IndexError):
            tag = ""
        try:
            xmlsuffixpos = xmlword.findall('.//{%s}mor/{%s}mor-post/{%s}mw/{%s}pos/{%s}c'
                                           % (NS, NS, NS, NS, NS))
            xmlsuffixpos2 = xmlword.findall('.//{%s}mor/{%s}mor-post/{%s}mw/{%s}pos/{%s}s'
                                            % (NS, NS, NS, NS, NS))
            if xmlsuffixpos2:
                suffixTag = xmlsuffixpos[0].text + ":" + xmlsuffixpos2[0].text
            else:
                suffixTag = xmlsuffixpos[0].text
        except:
            pass
        
        if suffixTag:
            tag += "~" + suffixTag
            
        return tag
    
    def _get_word_relation(self, xmlword, word, suffixStem):
        for xmlstem_rel in xmlword.findall('.//{%s}mor/{%s}gra'
                                                   % (NS, NS)):
            if not xmlstem_rel.get('type') == 'grt':
                word = (word[0], word[1],
                        xmlstem_rel.get('index')
                        + "|" + xmlstem_rel.get('head')
                        + "|" + xmlstem_rel.get('relation'))
            else:
                word = (word[0], word[1], word[2],
                        word[0], word[1],
                        xmlstem_rel.get('index')
                        + "|" + xmlstem_rel.get('head')
                        + "|" + xmlstem_rel.get('relation'))
        
        try:
            for xmlpost_rel in xmlword.findall('.//{%s}mor/{%s}mor-post/{%s}gra'
                                                       % (NS, NS, NS)):
                if not xmlpost_rel.get('type') == 'grt':
                    suffixStem = (suffixStem[0],
                                  suffixStem[1],
                                  xmlpost_rel.get('index')
                                  + "|" + xmlpost_rel.get('head')
                                  + "|" + xmlpost_rel.get('relation'))
                else:
                    suffixStem = (suffixStem[0], suffixStem[1],
                                  suffixStem[2], suffixStem[0],
                                  suffixStem[1],
                                  xmlpost_rel.get('index')
                                  + "|" + xmlpost_rel.get('head')
                                  + "|" + xmlpost_rel.get('relation'))
        except:
            pass
        
        return word, suffixStem
    
    
    def _get_transcription(self, xmlword, orthography, speaker, ipa):
        """
        Get the phonetic/phonemic transcription of a word. 
        """
        if speaker == ['CHI']:
            model_phones = self._get_child_phones(xmlword, False)
            actual_phones = self._get_child_phones(xmlword, True)
            transcription = self.child_transcription(model_phones, actual_phones)
        else:
            try:
                transcription = cmu[orthography.strip()][0]
            except KeyError:
                transcription = ''
            
            if ipa:
                transcription = self._arpa_to_ipa(transcription)
        
        return transcription

    def _arpa_to_ipa(self, transcription):
        """Turn Arpabet transcription into IPA transcription.
        """
        ipa_transcription = []
        for i, p in enumerate(transcription):
            if p != 'AH0':
                this_phone = cArpaToIPA[p.strip('012')]
            else:
                this_phone = cArpaToIPA[p]
            
            if (i == len(transcription) - 1 or transcription[i+1] in cConsonants) and this_phone == 'l' and ipa_transcription[-1] == 'ə':
                ipa_transcription[-1] = ipa_transcription[-1] + this_phone
            else:
                ipa_transcription.append(this_phone)
                
        return ipa_transcription
    
    def _get_child_phones(self, xmlword, actual = True):
        """Given a word uttered by a child, get the transcription of the model production
        and the child's actual production.
        """
        phones = []
        last_phone = ''
        
        if actual:
            xmlphones = xmlword.findall('.//{%s}actual/{%s}pw/{%s}ph' % (NS, NS, NS))
        else: 
            xmlphones = xmlword.findall('.//{%s}model/{%s}pw/{%s}ph' % (NS, NS, NS))
                    
        for i, xmlphone in enumerate(xmlphones):
            this_phone = xmlphone.text
            if this_phone == 'ː':
                continue
            
            if this_phone == '(':
                break
            
            if (this_phone in cDigraphs.keys() 
                and last_phone in cDigraphs[this_phone]):
                phones[-1] = phones[-1] + this_phone
            elif (i == len(xmlphones) - 1 or xmlphones[i+1].text in cVowels) and this_phone == 'l' and last_phone == 'ə':
                phones[-1] = phones[-1] + this_phone
            else:
                phones.append(this_phone)
            
            if phones[-1] in cReplacement.keys():
                phones[-1] = cReplacement[phones[-1]]
                
            last_phone = this_phone
            
        return phones
