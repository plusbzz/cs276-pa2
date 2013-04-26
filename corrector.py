import sys
from collections import deque
from itertools import izip,product
import cPickle as marshal
from math import exp, log

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

word_log_prob = unserialize_data('word_language_model.mrshl')
biword_counter = unserialize_data('biword_counter.mrshl')
word_counter = unserialize_data('word_counter.mrshl')
word_index = unserialize_data('word_index.mrshl')
bigram_index = unserialize_data('bigram_index.mrshl')
trigram_index = unserialize_data('trigram_index.mrshl')

def sigmoid(z): return 1.0/(1+exp(-z))

def jaccard_coeff(s1,s2):
  if len(s1) <= 6 or len(s2) <= 6:
    s1 = set([(t1+t2) for t1,t2 in zip(s1[:-1],s1[1:])])
    s2 = set([(t1+t2) for t1,t2 in zip(s2[:-1],s2[1:])])
  else:
    s1 = set([(t1+t2+t3) for t1,t2,t3 in zip(s1[:-2],s1[1:-1],s1[2:])])
    s2 = set([(t1+t2+t3) for t1,t2,t3 in zip(s2[:-2],s2[1:-1],s2[2:])])
  
  return (1.0*len(s1.intersection(s2)))/len(s1.union(s2))

def edit_distance(a,b,cutoff=sys.maxint):
  m = len(a)
  n = len(b)
  
  d = [[0 for x in xrange(n+1)] for x in xrange(m+1)]
  
  for i in range(1,m+1):
    d[i][0] = i
    
  for j in range(1,n+1):
    d[0][j] = j

  for j in range(1,n+1):
    for i in range(1,m+1):
      if a[i-1] == b[j-1]:
        d[i][j] = d[i-1][j-1]
      else:
        d[i][j] = min( [d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + 1] )
        
      if i == j and d[i][j] >= cutoff:
        return d[i][j]     
      
  return d[m][n]

def uniform_cost_edit_distance(r,q,cost):
  """
  Estimates the probability of writting 'r' when meaning 'q'.
  Any single edit using an operator defined in the Damerau-Levenshtein distance
  has unifor probability defined by 'cost'
  
  Returns P(r|q) = (cost^edit_distance(r,q) * P(q)
  """
  
  if q not in word_log_prob:
    return 0
  else:
    d = edit_distance(r,q)
    log_prob_q = word_log_prob[q]
    log_prob_r_q = d * log(cost) + log_prob_q
    
    return exp(log_prob_r_q)

def findEditOperation(finalWord,intendedWord):
  """
    Takes two words separated by 1 edit distance
    Returns a tuple with the edit operation applied and the characters involved
    e.g. findEditOperation(hpello,hello) returns (4,'h','p')
    
    match         = 0
    deletion      = 1
    substitution  = 2
    transposition = 3
    insertion     = 4
    
  """
  match         = 0
  deletion      = 1
  substitution  = 2
  transposition = 3
  insertion     = 4
  
  result = (match,'','')
  
  if edit_distance(finalWord,intendedWord) != 1:
    return result

  editFound = False  
  lastIntendedLetter="#" 
  finalWordLength = len(finalWord)
  intendedWordLength = len(intendedWord)
  
  finalWordIdx = 0
  intendedWordIdx = 0
  while (finalWordIdx < finalWordLength) and (intendedWordIdx < intendedWordLength):
    
    if finalWord[finalWordIdx] == intendedWord[intendedWordIdx]:
      lastIntendedLetter = intendedWord[intendedWordIdx]      
      finalWordIdx += 1
      intendedWordIdx +=1
    else:
      finalWordNextIdx = finalWordIdx + 1;
      intendedWordNextIdx = intendedWordIdx + 1;
     
      # Deletion
      if (finalWordLength == intendedWordLength - 1) and (intendedWordNextIdx < intendedWordLength) and finalWord[finalWordIdx] == intendedWord[intendedWordNextIdx]:
        result = (deletion,lastIntendedLetter,intendedWord[intendedWordIdx])
        editFound = True
        break

      # Transposition      
      if (finalWordLength == intendedWordLength) and (finalWordNextIdx < finalWordLength) and (intendedWordNextIdx < intendedWordLength):
        if (intendedWord[intendedWordIdx] == finalWord[finalWordNextIdx] and intendedWord[intendedWordNextIdx] == finalWord[finalWordIdx]):
          result = (transposition, intendedWord[intendedWordIdx], intendedWord[intendedWordNextIdx])
          editFound = True
          break
      
      # Substitutions
      if (finalWordLength == intendedWordLength) and (finalWord[finalWordIdx] != intendedWord[intendedWordIdx]):
        result = (substitution,intendedWord[intendedWordIdx],finalWord[finalWordIdx])
        editFound = True
        break
        
      # Insertion
      if (finalWordLength == intendedWordLength + 1) and (finalWordNextIdx < finalWordLength) and intendedWord[intendedWordIdx] == finalWord[finalWordNextIdx]:
        result = (insertion,lastIntendedLetter,finalWord[finalWordIdx])
        editFound = True
        break
        
  if not editFound and (intendedWordIdx == intendedWordLength) and (finalWordIdx < finalWordLength):
    result = (insertion,lastIntendedLetter,finalWord[finalWordIdx])
    editFound = True

  if not editFound and (finalWordIdx == finalWordLength) and (intendedWordIdx < intendedWordLength):
    result = (deletion,lastIntendedLetter,intendedWord[intendedWordIdx])
    editFound = True
    
  return result  

# Filters
def is_good_candidate(candidate,word,jaccard_cutoff = 0.4, edit_cutoff = 3):
  # Candidate should start with same letter
  if word[0] != candidate[0]: return False
  
  # Candidate should have length within edit_cutoff of word
  if abs(len(candidate) - len(word)) >= edit_cutoff: return False
  
  # Jaccard overlap
  if jaccard_coeff(candidate,word) <= jaccard_cutoff: return False
  
  #Edit distance should be <= 2
  if edit_distance(candidate,word) >= edit_cutoff: return False
  
  return True

def generate_word_candidates_from_ngrams(word,candidates,jaccard_cutoff = 0.4, edit_cutoff = 3):
  # For each bigram in word
  if len(word) <= 6:
    bigrams = set([(t1+t2) for t1,t2 in zip(word[:-1],word[1:])])
    for cb in bigrams:
      if cb in bigram_index:
        postings = bigram_index[cb]
        for candidate_id in postings:
          candidate = word_index[candidate_id]
          if (candidate not in candidates):
            if is_good_candidate(word,candidate,jaccard_cutoff,edit_cutoff):
              candidates.add(candidate)
  else:
    trigrams = set([(t1+t2+t3) for t1,t2,t3 in zip(word[:-2],word[1:-1],word[2:])])
    for ct in trigrams:
      if ct in trigram_index:
        postings = trigram_index[ct]
        for candidate_id in postings:
          candidate = word_index[candidate_id]
          if (candidate not in candidates):
            if is_good_candidate(word,candidate,jaccard_cutoff,edit_cutoff):
              candidates.add(candidate)
    
  return candidates
 
def generate_candidates_with_spaces(word,candidates):  
  space_candidates = set()
  for i in xrange(1,len(word)):
    w1 = word[:i]
    w2 = word[i:]
    
    # TODO Use biword probability here?
    
    #print >> sys.stderr, w1,w2
    if (w1 in word_counter) and (w2 in word_counter):
      #print >> sys.stderr, w1,w2,"both in index"
      space_candidates.add(w1 + " " + w2)
    elif (w1 not in word_counter) and (w2 not in word_counter):
      #print >> sys.stderr, w1,w2,"both not in index"
      pass
    elif (w1 not in word_counter) and (w2 in word_counter):
      #print >> sys.stderr, w1,w2,"w1 not in index"
      w1_cands = generate_word_candidates_from_ngrams(w1,set(),edit_cutoff = 1)
      space_candidates.update([_w1 + " " + w2 for _w1 in w1_cands])
    elif (w1 in word_counter) and (w2 not in word_counter):
      #print >> sys.stderr, w1,w2,"w2 not in index"
      w2_cands = generate_word_candidates_from_ngrams(w2,set(),edit_cutoff = 1)
      space_candidates.update([w1 + " " + _w2 for _w2 in w2_cands])
  
  for sc in space_candidates:
    if is_good_candidate(sc,word):
      candidates.add(sc)
      
  return candidates

# take each word of biword and generate isolated candidates from character-k-gram index
def generate_word_candidates(word):
  '''Accept a word, return a set of strings, each representing a candidate for that word'''
  candidates = set()
  # if word is in corpus then it's a candidate
  if word in word_counter:
    candidates.add(word)
  
  # TODO: if word is common enough, then should we generate fewer candidates?
  # TODO: Is there a way to make candidate generation more strict or loose?
  
  # special handling for spaces 
  candidates = generate_candidates_with_spaces(word,candidates)
  candidates = generate_word_candidates_from_ngrams(word,candidates)
  
  return candidates

def is_rare_word(word):
  return (word not in word_counter)
  
def is_rare_biword(biword):
  return (biword not in biword_counter)

def parse_singleword_query(query):
  return generate_word_candidates(query)

def parse_query(query):
  '''Process a multiword query'''
  candidate_list = deque([])
  empty_set = set()
  
  # Split query into biwords after converting to lowercase
  query = query.lower()
  words = query.split()
  
  if len(words) == 1:
    return parse_singleword_query(words[0])

  # Biword counts
  for biword in izip(words[1:], words[:-1]):
    # Decide if biword is rare enough
    if is_rare_biword(biword):
      for word in reversed(biword):
        candidates = generate_word_candidates(word)
        candidate_list.append((word,candidates))
    else:
      for word in reversed(biword):
        candidate_list.append((word,empty_set))

  final_query_list = []
  final_query_list.append(candidate_list.popleft()[1])  
  for i in xrange(0,len(candidate_list)-1,2):
    e1 = candidate_list.popleft()
    e2 = candidate_list.popleft()
    final_query_list.append(e1[1].union(e2[1]))
  final_query_list.append(candidate_list.popleft()[1])  
   
  #print >> sys.stderr, final_list 
    
  return [" ".join(q) for q in product(*final_query_list)]

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
