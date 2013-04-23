import marshal
import sys
import os.path
import gzip
from glob import iglob
from itertools import izip
from math import log
from collections import Counter
import re

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
  unigram_counter = Counter()
  # Unigram probabilities
  unigram_log_prob_dict = {}
  
  # Bigram counts
  bigram_counter = Counter()
  # Bigram probabilities. If bigram is "w1 w2", the key for the dict is (w2,w1), representing P(w2|w1)
  bigram_log_prob_dict = {}
  
  for block_fname in iglob( os.path.join( training_corpus_loc, '*.txt' ) ):
    print >> sys.stderr, 'processing dir: ' + block_fname
    with open( block_fname ) as f:
      words = re.findall(r'\w+', f.read().lower())
      print >> sys.stderr, 'Number of words in ' + block_fname + ' is ' + str(len(words))
      
      # Update Unigram counts
      for word in words:
        unigram_counter[word] += 1
        
      # Update Bigram counts
      for bigram in izip(words[1:], words[:-1]):
        bigram_counter[bigram] += 1
  
  
  # Finished counts, now calculate probabilities   
  total_terms = float(sum(unigram_counter.values()))     
  for unigram in unigram_counter:
    try:
      unigram_log_prob_dict[unigram] = log(unigram_counter[unigram]/total_terms)
    except ValueError:
      print >> sys.stderr, unigram, unigram_counter[unigram],total_terms
      
  
  # Calculate interpolated bigram probability   
  for bigram in bigram_counter:
    w2,w1 = bigram
    bigram_log_prob_dict[bigram] = log(lam*unigram_counter[w2]/total_terms
                                       + (1.0-lam)*bigram_counter[bigram]/unigram_counter[w1])
   
  
  # Save language models using marshal
  print >> sys.stderr, "Serializing language models and counters"
  serialize_data(unigram_log_prob_dict,"unigram_language_model.mrshl")
  serialize_data(bigram_log_prob_dict,"bigram_language_model.mrshl")
  serialize_data(unigram_counter,"unigram_counter.mrshl")
  serialize_data(bigram_counter,"bigram_counter.mrshl")

  return (unigram_log_prob_dict,bigram_log_prob_dict)

def create_2gram_index(unigram_dict):
  unigram_index = {}
  char_bigram_index = {}
  
  counter_u = 1
  for unigram in unigram_dict:
    unigram_index[counter_u] = unigram
    
    char_bigrams = set([(t1+t2) for t1,t2 in zip(unigram[:-1],unigram[1:])])
  
    for cb in char_bigrams:
      if cb not in char_bigram_index:
        char_bigram_index[cb] = []
      char_bigram_index[cb].append(counter_u)
      
    counter_u += 1
  
  print >>sys.stderr,[unigram_index[i]  for i in char_bigram_index['th']]
  # Save kgram index using marshal
  print >> sys.stderr, "Serializing character bigram index"
  serialize_data(unigram_index,"unigram_index.mrshl")
  serialize_data(char_bigram_index,"char_bigram_index.mrshl")
  
if __name__ == '__main__':
  u,b = scan_corpus(sys.argv[1],lam=0.2)
  print  >> sys.stderr,u['the']
  print  >> sys.stderr,b[('people','the')]
  
  create_2gram_index(u)
  
