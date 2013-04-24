import sys
import os.path
import gzip
from glob import iglob
from itertools import izip
from math import log
from collections import Counter
import re
import cPickle as marshal

def serialize_data(data, fname):
  """
  Writes `data` to a file named `fname`
  """
  with open(fname, 'wb') as f:
    marshal.dump(data, f)
 
def read_edit1s():
  """
  Returns the edit1s data
  It's a list of tuples, structured as [ .. , (misspelled query, correct query), .. ]
  """
  edit1s = []
  with gzip.open(edit1s_loc) as f:
    # the .rstrip() is needed to remove the \n that is stupidly included in the line
    edit1s = [ line.rstrip().split('\t') for line in f if line.rstrip() ]
  return edit1s

def scan_corpus(training_corpus_loc,lam = 0.2):
  """
  Scans through the training corpus and counts how many lines of text there are
  """
  # Unigram counts
  word_counter = Counter()
  # Unigram probabilities
  word_log_prob_dict = {}
  
  # Bigram counts
  biword_counter = Counter()
  # Bigram probabilities. If bigram is "w1 w2", the key for the dict is (w2,w1), representing P(w2|w1)
  biword_log_prob_dict = {}
  
  for block_fname in iglob( os.path.join( training_corpus_loc, '*.txt' ) ):
    print >> sys.stderr, 'processing dir: ' + block_fname
    with open( block_fname ) as f:
      words = re.findall(r'\w+', f.read().lower())
      print >> sys.stderr, 'Number of words in ' + block_fname + ' is ' + str(len(words))
      
      # Update Unigram counts
      for word in words:
        word_counter[word] += 1
        
      # Update Bigram counts
      for biword in izip(words[1:], words[:-1]):
        biword_counter[biword] += 1
  
  
  # Finished counts, now calculate probabilities   
  total_terms = float(sum(word_counter.values()))     
  for word in word_counter:
    try:
      word_log_prob_dict[word] = log(word_counter[word]/total_terms)
    except ValueError:
      print >> sys.stderr, word, word_counter[word],total_terms
      
  
  # Calculate interpolated bigram probability   
  for biword in biword_counter:
    w2,w1 = biword
    biword_log_prob_dict[biword] = log(lam*word_counter[w2]/total_terms
                                       + (1.0-lam)*biword_counter[biword]/word_counter[w1])
   
  
  # Save language models using marshal
  print >> sys.stderr, "Serializing language models and counters"
  serialize_data(word_log_prob_dict,"word_language_model.mrshl")
  serialize_data(biword_log_prob_dict,"biword_language_model.mrshl")
  serialize_data(word_counter,"word_counter.mrshl")
  serialize_data(biword_counter,"biword_counter.mrshl")

  return (word_log_prob_dict,biword_log_prob_dict)

def create_ngram_index(word_dict):
  word_index = {}
  bigram_index = {}
  trigram_index = {}
  
  counter_u = 1
  for word in word_dict:
    word_index[counter_u] = word
    
    bigrams = set([(t1+t2) for t1,t2 in zip(word[:-1],word[1:])])
    for cb in bigrams:
      if cb not in bigram_index:
        bigram_index[cb] = []
      bigram_index[cb].append(counter_u)
          
    trigrams = set([(t1+t2+t3) for t1,t2,t3 in zip(word[:-2],word[1:-1],word[2:])])
    for ct in trigrams:
      if ct not in trigram_index:
        trigram_index[ct] = []
      trigram_index[ct].append(counter_u)
          
    counter_u += 1
    
  print >>sys.stderr,[word_index[i]  for i in bigram_index['th']]
  # Save kgram index using marshal
  print >> sys.stderr, "Serializing character ngram index"
  serialize_data(word_index,"word_index.mrshl")
  serialize_data(bigram_index,"bigram_index.mrshl")
  serialize_data(trigram_index,"trigram_index.mrshl")


  
if __name__ == '__main__':
  u,b = scan_corpus(sys.argv[1],lam=0.2)
  print  >> sys.stderr,u['the']
  print  >> sys.stderr,b[('people','the')]
  
  create_ngram_index(u)
  
