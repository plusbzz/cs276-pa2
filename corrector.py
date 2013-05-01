import sys
from collections import deque
from itertools import izip,product,islice
import cPickle as marshal
from math import exp, log
from collections import Counter
import operator
from edit_operations import findEditOperation, edit_distance

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

print >>sys.stderr, "Reading language models"
word_log_prob = unserialize_data('word_language_model.mrshl')
biword_log_prob = unserialize_data('biword_language_model.mrshl')
biword_counter = unserialize_data('biword_counter.mrshl')
word_counter = unserialize_data('word_counter.mrshl')

print >>sys.stderr, "Reading n-gram indices"
word_index = unserialize_data('word_index.mrshl')
bigram_index = unserialize_data('bigram_index.mrshl')
trigram_index = unserialize_data('trigram_index.mrshl')

print >>sys.stderr, "Reading edit probabilities"
edits_del_counter = unserialize_data("edits_del_counter.mrshl")
edits_sub_counter = unserialize_data("edits_sub_counter.mrshl")
edits_tra_counter = unserialize_data("edits_tra_counter.mrshl")
edits_ins_counter = unserialize_data("edits_ins_counter.mrshl")
edits_char_counter = unserialize_data("edits_char_counter.mrshl")
edits_bichar_counter = unserialize_data("edits_bichar_counter.mrshl")


def sigmoid(z): return 1.0/(1+exp(-z))

def jaccard_coeff(s1,s2):
  '''Use bigrams or trigrams to calculate jaccard similarity of two strings'''
  if len(s1) <= 10 or len(s2) <= 10:
    s1 = set([(t1+t2) for t1,t2 in zip(s1[:-1],s1[1:])])
    s2 = set([(t1+t2) for t1,t2 in zip(s2[:-1],s2[1:])])
  else:
    s1 = set([(t1+t2+t3) for t1,t2,t3 in zip(s1[:-2],s1[1:-1],s1[2:])])
    s2 = set([(t1+t2+t3) for t1,t2,t3 in zip(s2[:-2],s2[1:-1],s2[2:])])
  
  return (1.0*len(s1.intersection(s2)))/len(s1.union(s2))

def calculate_biword_log_prob_sb(biword,alpha=0.4):
  '''Calculate biword prior log-probability with stupid backoff'''
  w2,w1 = biword
  bprob = 0
  if biword in biword_log_prob:
    bprob = biword_log_prob[biword]
  elif w2 in word_log_prob:
    bprob = log(alpha) + word_log_prob[w2]
  
  if bprob == 0: return -100
  return bprob

def calculate_biword_log_prob(biword,lam=0.2):
  '''Calculate biword prior log-probability with interpolation'''
  
  llam = log(lam)
  llam_c = log(1-lam)
  w2,w1 = biword
  bprob = 0
  if biword in biword_log_prob:
    bprob += exp(llam_c+biword_log_prob[biword])
  if w2 in word_log_prob:
    bprob += exp(llam + word_log_prob[w2])
  
  if bprob == 0: return -100
  return log(bprob)

def calculate_log_prob(query,lam=0.2):
  '''
  Calculate prior log-probability of a (multi-word) query Q = (w1,w2,...,wn)
  P(Q) = P(w1)P(w2|w1)... and so on
  '''
  words = query.split() 
  prob = 0
  # Product of biword conditionals
  for biword in izip(words[1:], words[:-1]):
    prob += calculate_biword_log_prob(biword,lam)
    
  w = words[0]
  if w in word_log_prob:
    prob += word_log_prob[w]
    
  if prob == 0: return -100
  return prob
  
def uniform_cost_edit_distance(r,q,cost=0.1,p_r_qr=0.9):
  """
  Estimates the probability of writing 'r' when meaning 'q'.
  Any single edit using an operator defined in the Damerau-Levenshtein distance
  has uniform probability defined by 'cost'
  
  Returns log( P(r|q) ) if r != q where P(r|q) = (cost^edit_distance(r,q) * P(q))
          log( p_r_qr ) if r == q  
  """
  if r==q:
    return log(p_r_qr)
  else:
    d = edit_distance(r,q)
    log_prob_q = calculate_log_prob(q)
    log_prob_r_q = d * log(cost) + log_prob_q
    return log_prob_r_q


def empirical_cost_edit_distance(r,q,uniform_cost=0.1,p_r_qr=0.9):
  """
  Estimates the probability of writing 'r' when meaning 'q'
  The cost of a single edit in the Damerau-Levenshtein distance is calculated from a noisy chanel model

  if editDistance(r,q) == 1 then P(r|q) is taken from the empirical noisy model
  if editDistance(r,q) > 1 then P(r|q) = P_empirical(r|q) * P_uniform(r|q)^(distance-1)
  
  Returns log( P(r|q) ) if r != q where P(r|q) = (cost^edit_distance(r,q) * P(q))
          log( p_r_qr ) if r == q 
  
  """
  
  if r==q:
    return log(p_r_qr) 
  else: 
    # if len(editOperation) > 0:
    d                  = edit_distance(r,q)  
    editOperation      = findEditOperation(r,q)
    log_prob_q         = calculate_log_prob(q)
    confusion_matrices = [edits_del_counter,edits_sub_counter,edits_tra_counter,edits_ins_counter]
    
    # editOperation e.g. [0, ('#','s')]  from: actual = un; intended = sun
    editName      = editOperation[0]
    editArguments = editOperation[1]
    
    # How many such edits were found on the training file for the noisy model
    numerator = confusion_matrices[editName][editArguments]
    
    if editName == 0: # deletion
        denominator = edits_bichar_counter[editArguments]
    elif editName == 1: # substitution
        denominator = edits_char_counter[editArguments[1]]
    elif editName == 2: # transposition
        denominator = edits_bichar_counter[editArguments]
    elif editName == 3: # insertion
        denominator = edits_char_counter[editArguments[0]]
    
    # Add-1 smoothing
    numberOfCharsInAlphabet = len(edits_char_counter)
    prob_r_q = float(numerator + 1) / float(denominator + numberOfCharsInAlphabet) 

    log_prob_r_q = log(prob_r_q) + (d-1)*log(uniform_cost) + log_prob_q  #code
    return log_prob_r_q

def is_good_candidate(candidate,word,jaccard_cutoff = 0.2, edit_cutoff = 3):
  '''Test if a candidate is good enough to a word with some heuristics'''

  # Candidate should start with same letter
  if word[0] != candidate[0]: return False
  
  # Candidate should have length within edit_cutoff of word
  if abs(len(candidate) - len(word)) >= edit_cutoff: return False
  
  # Jaccard overlap
  if len(word) > 10: jaccard_cutoff = max(jaccard_cutoff,0.5)
  if jaccard_coeff(candidate,word) <= jaccard_cutoff: return False
  
  #Edit distance should be <= 2
  if edit_distance(candidate,word) >= edit_cutoff: return False
  
  return True

def generate_word_candidates_from_ngrams(word,candidates,jaccard_cutoff = 0.2, edit_cutoff = 3):
  '''Generate candidates by concatenating postings lists from shared bigrams or trigrams (depending on length)'''

  if len(word) < 10:
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
  '''Generate candidates for a word with spaces inserted'''

  space_candidates = set()
  for i in xrange(1,len(word)):
    w1 = word[:i]
    w2 = word[i:]
    
    # TODO Use biword probability here?
    
    if (w1 in word_counter) and (w2 in word_counter):
      space_candidates.add(w1 + " " + w2)
    elif (w1 not in word_counter) and (w2 not in word_counter):
      pass
    elif (w1 not in word_counter) and (w2 in word_counter):
      w1_cands = generate_word_candidates_from_ngrams(w1,set(),edit_cutoff = 2)
      space_candidates.update([_w1 + " " + w2 for _w1 in w1_cands])
    elif (w1 in word_counter) and (w2 not in word_counter):
      w2_cands = generate_word_candidates_from_ngrams(w2,set(),edit_cutoff = 2)
      space_candidates.update([w1 + " " + _w2 for _w2 in w2_cands])
  
  for sc in space_candidates:
    if is_good_candidate(sc,word):
      candidates.add(sc)
      
  return candidates

def rank_candidates(candidates,word,max_c):
  '''Rank candidates for a word using cost function and return at most top max_c candidates'''

  scored_candidates = {}
  for cand in candidates:
    scored_candidates[cand] = edit_cost_func(word,cand)
    
  ranked_candidates = sorted(scored_candidates.iteritems(), key=operator.itemgetter(1),reverse=True)
  #print >> sys.stderr, ranked_candidates
  
  ranked_candidates = ranked_candidates[:max_c]
  return [c for c,s in ranked_candidates]
    
def generate_word_candidates(word, max_c = 100):
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
  
  # Ranking of candidates
  candidates = rank_candidates(candidates,word,max_c)
  
  return candidates

def is_rare_word(word):
  return (word not in word_counter)
  
def is_rare_biword(biword):
  return (biword not in biword_counter)

def parse_singleword_query(query):
  '''Process a single-word query'''

  return generate_word_candidates(query)

def get_edit_cost_func(method):
  if method == 'uniform': return uniform_cost_edit_distance
  if method == 'empirical': return empirical_cost_edit_distance
  return uniform_cost_edit_distance

def parse_query(query):
  '''Process a multiword query'''

  candidate_list = deque([])
  max_candidates = 500
  
  # Split query into biwords after converting to lowercase
  query = query.lower()
  words = query.split()
  len_q = len(words)

  candidates_per_word = max_candidates/len_q;

  if len_q == 1:
    return parse_singleword_query(words[0])

  
  # Biword counts
  for biword in izip(words[1:], words[:-1]):
    # Decide if biword is rare enough
    if is_rare_biword(biword):
      for word in reversed(biword):
        candidates = generate_word_candidates(word,candidates_per_word)
        candidate_list.append((word,candidates))
    else:
      for word in reversed(biword):
        candidate_list.append((word,[word]))

  final_query_list = []
  final_query_list.append(candidate_list.popleft()[1])  
  for i in xrange(0,len(candidate_list)-1,2):
    e1 = candidate_list.popleft()
    e2 = candidate_list.popleft()
    if len(e1[1]) > 0:
      final_query_list.append(e1[1])
    elif len(e2[1]) > 0:
      final_query_list.append(e2[1])
  final_query_list.append(candidate_list.popleft()[1])  
  
  #print >> sys.stderr,final_query_list
  
  candidates =  [" ".join(q) for q in islice(product(*final_query_list),0,max_candidates)]
  return rank_candidates(candidates,query,max_candidates)

def read_query_data(queries_loc,gold_loc,google_loc):
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
  global edit_cost_func
  global biword_prob_func
  
  edit_cost_func = uniform_cost_edit_distance
  biword_prob_func = calculate_biword_log_prob
      
  if len(sys.argv) == 3:
    method = sys.argv[1]
    edit_cost_func = get_edit_cost_func(method)
    if method == 'extra': biword_prob_func = calculate_biword_log_prob_sb # use stupid backoff
    
    print >> sys.stderr,method
    queries_loc = sys.argv[2]
    with open(queries_loc) as f:
      queries = [line.rstrip() for line in f]
      
    for query in queries:
      cands = parse_query(query)
      best_cand = cands[0] if len(cands) > 0 else ""
      print best_cand
      
      
  if len(sys.argv) >= 4:
    queries,golds,googles = read_query_data(sys.argv[1],sys.argv[2],sys.argv[3])
    total = 0
    correct = 0
    google_correct = 0

    for query,gold,google in izip(queries,golds,googles):
      cands = parse_query(query)
      best_cand = cands[0] if len(cands) > 0 else ""
  
      if best_cand == gold:
        result = "Right"
        correct+=1
      else:
        result = "Wrong"
      
      if google == gold:
        google_correct += 1
      
      total +=1
      print >> sys.stderr,query,len(best_cand),len(gold)
      print "|".join([result,query,best_cand,gold,google])
  
    print >> sys.stderr, correct,"out of",total,"correct."
    print >> sys.stderr, google_correct,"out of",total,"correct for google."
