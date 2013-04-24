import sys
from collections import deque
from itertools import izip
import cPickle as marshal

queries_loc = 'data/queries.txt'
gold_loc = 'data/gold.txt'
google_loc = 'data/google.txt'

alphabet = "abcdefghijklmnopqrstuvwxyz0123546789&$+_' "

def unserialize_data(fname):
  """
  Reads a pickled data structure from a file named `fname` and returns it
  IMPORTANT: Only call marshal.load( .. ) on a file that was written to using marshal.dump( .. )
  marshal has a whole bunch of brittle caveats you can take a look at in teh documentation
  It is faster than everything else by several orders of magnitude though
  """
  with open(fname, 'rb') as f:
    return marshal.load(f)


biword_counter = unserialize_data('biword_counter.mrshl')
word_counter = unserialize_data('word_counter.mrshl')
word_index = unserialize_data('word_index.mrshl')
bigram_index = unserialize_data('bigram_index.mrshl')



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

# take each word of biword and generate isolated candidates from character-k-gram index
def generate_word_candidates(word):
  return set(["test"])

def is_rare_biword(biword):
  return (biword not in biword_counter)

def generate_candidate_queries(candidate_dict):
  pass

def parse_singleword_query(query):
  pass

def parse_multiword_query(query):
  candidate_list = deque([])
  empty_set = set()
  
  # Split query into biwords after converting to lowercase
  words = query.lower().split()
  
  if len(words) == 1:
    return parse_singleword_query(words[0])

  # Update Bigram counts
  for biword in izip(words[1:], words[:-1]):
    # Decide if biword is rare enough
    if is_rare_biword(biword):
      for word in reversed(biword):
        candidates = generate_word_candidates(word)
        candidate_list.append(candidates)
    else:
      for word in reversed(biword):
        candidate_list.append(empty_set)

  final_list = []
  final_list.append(candidate_list.popleft())  
  for i in xrange(0,len(candidate_list)-1,2):
    e1 = candidate_list.popleft()
    e2 = candidate_list.popleft()
    final_list.append(e1.union(e2))
  final_list.append(candidate_list.popleft())  
   
  print >> sys.stderr, candidate_list
  print >> sys.stderr, final_list
  candidate_queries = generate_candidate_queries(final_list)
  
  return candidate_queries

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
