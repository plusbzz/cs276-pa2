
import sys
from collections import OrderedDict
import marshal

queries_loc = 'data/queries.txt'
gold_loc = 'data/gold.txt'
google_loc = 'data/google.txt'

alphabet = "abcdefghijklmnopqrstuvwxyz0123546789&$+_' "

bigram_counter = unserialize_data('bigram_counter.mrshl')
unigram_counter = unserialize_data('unigram_counter.mrshl')
unigram_index = unserialize_data('unigram_index.mrshl')
char_bigram_index = unserialize_data('char_bigram_index.mrshl')


def unserialize_data(fname):
  """
  Reads a pickled data structure from a file named `fname` and returns it
  IMPORTANT: Only call marshal.load( .. ) on a file that was written to using marshal.dump( .. )
  marshal has a whole bunch of brittle caveats you can take a look at in teh documentation
  It is faster than everything else by several orders of magnitude though
  """
  with open(fname, 'rb') as f:
    return marshal.load(f)

def jaccard_coeff(s1,s2):
  s1 = set([(t1+t2) for t1,t2 in zip(s1[:-1],s1[1:])])
  s2 = set([(t1+t2) for t1,t2 in zip(s2[:-1],s2[1:])])
  
  return (1.0*len(s1.intersection(s2)))/len(s1.union(s2))

def generate_word_candidates(word):
  candidates = set()
  # Split word into char-bigrams
  # Take union of postings list for each bigram
  # Perform some basic filtering
  # return candidates for word
  return

def generate_biword_candidates(biword, candidate_dict):
  pass

def is_rare_biword(biword):
  return False

def generate_candidate_queries(candidate_dict):
  pass

def parse_query(query):
  candidate_dict = OrderedDict()
  # Split query into biwords after converting to lowercase
  words = query.lower().split()
  
  # Decide if biword is rare enough
  
  # if rare, take each word of biword and generate isolated candidates from character-k-gram index

def read_query_data():
  """
  all three files match with corresponding queries on each line
  """
  queries = []
  gold = []
  google = []
  with open(queries_loc) as f:
    for line in f:
      queries.append(line.rstrip())
  with open(gold_loc) as f:
    for line in f:
      gold.append(line.rstrip())
  with open(google_loc) as f:
    for line in f:
      google.append(line.rstrip())
  assert( len(queries) == len(gold) and len(gold) == len(google) )
  return (queries, gold, google)

if __name__ == '__main__':
  print(sys.argv)
