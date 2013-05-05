import sys
import collections
from itertools import izip
from math import log,exp
from collections import Counter
import cPickle as marshal

import sys
from collections import deque
from itertools import izip,product,islice
import cPickle as marshal
from math import exp, log
import operator

def serialize_data(data, fname):
  """
  Writes `data` to a file named `fname`
  """
  with open(fname, 'wb') as f:
    marshal.dump(data, f)
    
def unserialize_data(fname):
  """
  Reads a pickled data structure from a file named `fname` and returns it
  IMPORTANT: Only call marshal.load( .. ) on a file that was written to using marshal.dump( .. )
  marshal has a whole bunch of brittle caveats you can take a look at in teh documentation
  It is faster than everything else by several orders of magnitude though
  """
  with open(fname, 'rb') as f:
    return marshal.load(f)    


def edit_distance_1(a,b,cutoff):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
        
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            
            if a[j-1] != b[i-1]:
                change = change + 1
                
            current[j] = min(add, delete, change)
    return current[n]

def edit_distance_2(a,b,cutoff=sys.maxint):
  '''Calculate edit distance between two strings'''
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
        # Deletion, Insertion, Substitution
        d[i][j] = min( [d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + 1] )
        
        # Transposition
        if (i>1 and j>1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]):
          d[i][j] = min( [d[i][j], d[i-2][j-2] + 1] )
        
        
      if i == j and d[i][j] >= cutoff:
        return d[i][j]     
      
  return d[m][n]


  
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
        # Deletion, Insertion, Substitution
        d[i][j] = min( [d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + 1] )
        
        # Transposition
        if (i>1 and j>1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]):
          d[i][j] = min( [d[i][j], d[i-2][j-2] + 1] )
        
        
      if i == j and d[i][j] >= cutoff:
        return d[i][j]     
      
  return d[m][n]

word_log_prob = {'pepe':log(0.3), 'tuyo':log(0.5)
                 }
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
  
def findEditOperation2(actualWord,intendedWord):
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
  operations = []

  match         = 0
  deletion      = 1
  substitution  = 2
  transposition = 3
  insertion     = 4
  
  lenActualWord   = len(actualWord)
  lenIntendedWord = len(intendedWord)
  
  actualWordIdx   = 0
  intendedWordIdx = 0
  
  while (actualWordIdx < lenActualWord):
    actualWordNextIdx   = actualWordIdx + 1
    intendedWordNextIdx = intendedWordIdx + 1
    
    #Transposition
    if (actualWordNextIdx < lenActualWord) and (intendedWordIdx < lenIntendedWord):
      if (actualWord[actualWordIdx] == intendedWord[intendedWordNextIdx]) and (intendedWord[intendedWordNextIdx] == actualWord[actualWordIdx]):
        operations += [(transposition, intendedWord[intendedWordIdx], intendedWord[intendedWordNextIdx])]
        actualWordIdx   += 2
        intendedWordIdx += 2
        
      
  return operations

def findEditOperation2(finalWord,intendedWord):
  """
    Takes two words separated by 1 edit distance
    Returns a tuple with the edit operation applied and the characters involved
    Follows the approach defined by Kernighan, Church and Gale in 'A Spelling Correction Program Based On Noisy Channel Model'
    e.g. findEditOperation(hpello,hello) returns (4,'h','p')
    
    deletion      = 0
    substitution  = 1
    transposition = 2
    insertion     = 3
    
  """
  deletion      = 0
  substitution  = 1
  transposition = 2
  insertion     = 3
  
  result = []
  
  if edit_distance(finalWord,intendedWord) > 2:
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
        result = [deletion, (lastIntendedLetter,intendedWord[intendedWordIdx])]
        editFound = True
        break

      # Transposition      
      if (finalWordLength == intendedWordLength) and (finalWordNextIdx < finalWordLength) and (intendedWordNextIdx < intendedWordLength):
        if (intendedWord[intendedWordIdx] == finalWord[finalWordNextIdx] and intendedWord[intendedWordNextIdx] == finalWord[finalWordIdx]):
          result = [transposition, (intendedWord[intendedWordIdx], intendedWord[intendedWordNextIdx])]
          editFound = True
          break
        
      # Insertion
      if (finalWordLength == intendedWordLength + 1) and (finalWordNextIdx < finalWordLength) and intendedWord[intendedWordIdx] == finalWord[finalWordNextIdx]:
        result = [insertion, (lastIntendedLetter,finalWord[finalWordIdx])]
        editFound = True
        break
    
      # Substitutions
      #if (finalWordLength == intendedWordLength) and (finalWord[finalWordIdx] != intendedWord[intendedWordIdx]):
      if (finalWord[finalWordIdx] != intendedWord[intendedWordIdx]):
        result = [substitution, (finalWord[finalWordIdx],intendedWord[intendedWordIdx])]
        editFound = True
        break    
        
  if not editFound and (intendedWordIdx == intendedWordLength) and (finalWordIdx < finalWordLength):
    result = [insertion, (lastIntendedLetter,finalWord[finalWordIdx])]
    editFound = True

  if not editFound and (finalWordIdx == finalWordLength) and (intendedWordIdx < intendedWordLength):
    result = [deletion, (lastIntendedLetter,intendedWord[intendedWordIdx])]
    editFound = True
    
  return result


  
def findEditOperation(finalWord,intendedWord):
  """
    Takes two words separated by 1 edit distance
    Returns a tuple with the edit operation applied and the characters involved
    e.g. findEditOperation(hpello,hello) returns (4,'h','p')
    
    deletion      = 0
    substitution  = 1
    transposition = 2
    insertion     = 3
    
  """
  deletion      = 0
  substitution  = 1
  transposition = 2
  insertion     = 3
  
  result = []
  
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
        result = [deletion, (lastIntendedLetter,intendedWord[intendedWordIdx])]
        editFound = True
        break

      # Transposition      
      if (finalWordLength == intendedWordLength) and (finalWordNextIdx < finalWordLength) and (intendedWordNextIdx < intendedWordLength):
        if (intendedWord[intendedWordIdx] == finalWord[finalWordNextIdx] and intendedWord[intendedWordNextIdx] == finalWord[finalWordIdx]):
          result = [transposition, (intendedWord[intendedWordIdx], intendedWord[intendedWordNextIdx])]
          editFound = True
          break
      
      # Substitutions
      if (finalWordLength == intendedWordLength) and (finalWord[finalWordIdx] != intendedWord[intendedWordIdx]):
        result = [substitution, (finalWord[finalWordIdx],intendedWord[intendedWordIdx])]
        editFound = True
        break
        
      # Insertion
      if (finalWordLength == intendedWordLength + 1) and (finalWordNextIdx < finalWordLength) and intendedWord[intendedWordIdx] == finalWord[finalWordNextIdx]:
        result = [insertion, (lastIntendedLetter,finalWord[finalWordIdx])]
        editFound = True
        break
        
  if not editFound and (intendedWordIdx == intendedWordLength) and (finalWordIdx < finalWordLength):
    result = [insertion, (lastIntendedLetter,finalWord[finalWordIdx])]
    editFound = True

  if not editFound and (finalWordIdx == finalWordLength) and (intendedWordIdx < intendedWordLength):
    result = [deletion, (lastIntendedLetter,intendedWord[intendedWordIdx])]
    editFound = True
    
  return result

def trainNoisyChannel(trainingFile):
  """
  Uses a training file to create Edit Distance confusion matrices and uniChar and biChar indexes
  Returns a list with the 4 confusion matrix and the uniChar and biChar indexes [delMatrix,subMatrix,traMatrix,insMatrix,uniChar,biChar]
  
  Matrices and indexes are implemented as Counter (char1,char2) -> counts
  Order of elements in tuple (char1,char2) is defined using the approach described by Kernighan, Church and Gale in 'A Spelling Correction Program Based On Noisy Channel Model'
  
  del[(x,y)] = count(xy typed as x)
  sub[(x,y)] = count(y typed as x)
  tra[(x,y)] = count(xy typed as yx)
  ins[(x,y)] = count(x typed as xy)
  
  """
  delCounter = Counter()
  subCounter = Counter()
  traCounter = Counter()
  insCounter = Counter()
  
  matrices = [delCounter,subCounter,traCounter,insCounter]
  
  with open(trainingFile) as fTraining:
    for line in fTraining:
      actualQuery,intendedQuery= line.split('\t',1)
      
      actualQuery = actualQuery.split()
      intendedQuery = intendedQuery.split()
      noOperation = []
      
      # Not considering splits or merges right now
      if len(actualQuery) == len(intendedQuery):
        for idx in range(len(actualQuery)):
          edit1 = findEditOperation(actualQuery[idx],intendedQuery[idx])
          
          if edit1 != noOperation:
            matrix = matrices[edit1[0]]
            matrix[edit1[1]] += 1
  
  serialize_data(delCounter,"edits_del_counter.mrshl")
  serialize_data(subCounter,"edits_sub_counter.mrshl")
  serialize_data(traCounter,"edits_tra_counter.mrshl")
  serialize_data(insCounter,"edits_ins_counter.mrshl")
  
  (charCounter, biCharCounter) = generateNGramsFromNoisyFile(trainingFile)
  serialize_data(charCounter,"edits_char_counter.mrshl")
  serialize_data(biCharCounter,"edits_bichar_counter.mrshl")
  
  return matrices + [charCounter, biCharCounter] 

def generateNGramsFromNoisyFile(trainingFile):
  charCounter   = Counter()
  biCharCounter = Counter()
  
  with open(trainingFile) as fTraining:
    for line in fTraining:
      actualQuery,intendedQuery = line.split('\t',1)
      
      intendedQueryChars = []
      intendedQueryChars.extend(intendedQuery.replace(' ','#'))
      
      # Count Individual Chars
      for c in intendedQueryChars:
        charCounter[c] += 1 
      
      # Count Bichars
      for bichar in izip(intendedQueryChars[:-1], intendedQueryChars[1:]):
        biCharCounter[bichar] += 1
        
  return (charCounter, biCharCounter)        
      
def empirical_cost_edit_distance(r,q, delCounter,subCounter,traCounter,insCounter,charCounter,biCharCounter, uniform_cost = 0.1):
    """
    Estimates the probability of writing 'r' when meaning 'q'
    The cost of a single edit in the Damerau-Levenshtein distance is calculated from a noisy chanel model
    
    Returns:
    log (P(r|q))
    
    if editDistance(r,q) == 1 then P(r|q) is taken from the empirical noisy model
    if editDistance(r,q) > 1 then P(r|q) = P_empirical(r|q) * P_uniform(r|q)^(distance-1)
    
    """

    d                  = edit_distance(r,q)  
    editOperation      = findEditOperation(r,q)
    log_prob_q         = calculate_log_prob(q)
    confusion_matrices = [delCounter,subCounter,traCounter,insCounter]
    
    if d == 0:
        return log_prob_q  # is this right? Where to use P(r|q) where r==q?
    else:
    
        # editOperation e.g. [0, ('#','s')]  from: actual = un; intended = sun
        editName      = editOperation[0]
        editArguments = editOperation[1]
        
        # How many such edits were found on the training file for the noisy model
        numerator = confusion_matrices[editName][editArguments]
        
        if editName == 0: # deletion
            denominator = biCharCounter[editArguments]
        elif editName == 1: # substitution
            denominator = charCounter[editArguments[1]]
        elif editName == 2: # transposition
            denominator = biCharCounter[editArguments]
        elif editName == 3: # insertion
            denominator = charCounter[editArguments[0]]
        
        # Add-1 smoothing
        numberOfCharsInAlphabet = len(charsCounter)
        prob_r_q = (numerator + 1) / (denominator + numberOfCharsInAlphabet) 
    
        log_prob_r_q = log(prob_r_q) + (d-1)*log(uniform_cost) + log_prob_q
        
        return log_prob_r_q
    
def mytest():
    return ["pepe","juan"]


if __name__=="__main__":
    from sys import argv
    
    print ""
    print "'quad','quadri'"
    print edit_distance_2('quad','quadri')
    print findEditOperation2('quad','quadri')
    
    print ""
    print "'qudar','quadri'"
    print edit_distance_2('qudar','quadri')
    print findEditOperation2('qudar','quadri')

    


    

    print ""
    print "'quxrdi','quadri'"
    print edit_distance_2('quxrdi','quadri')
    print findEditOperation2('quxrdi','quadri')

    
    print ""
    print "'qauxri','quadri'"
    print edit_distance_2('qauxri','quadri')
    print findEditOperation2('qauxri','quadri')
    
    print ""
    print "'quade','quadri'"
    print edit_distance_2('quade','quadri')
    print findEditOperation2('quade','quadri')
    
    print ""
    print "'quad','quadri'"
    print edit_distance_2('quad','quadri')
    print findEditOperation2('quad','quadri')

    print ""
    print "'uqxdri','quadri'"
    print edit_distance_2('uqxdri','quadri')
    print findEditOperation2('uqxdri','quadri')

    print ""
    print "'xaudri','quadri'"
    print edit_distance_2('xaudri','quadri')
    print findEditOperation2('xaudri','quadri')
    
    
   #print edit_distance_2('decre','decreto')
    #print findEditOperation2('decre','decreto')
    
    
    #ngram_indexes = mytest()
    
    #print ngram_indexes[0]
    #print ngram_indexes[1]
    
    #(charCounter,biCharCounter) = generateNGramsFromNoisyFile("./edits1mmm.txt")
    #
    #print "charCounter ==="
    #print charCounter
    #print "biCharCounter ==="
    #print biCharCounter
    #
    #lcharCounter = unserialize_data("edits_char_counter.mrshl")
    #lbicharCounter = unserialize_data("edits_bichar_counter.mrshl")
    #
    #print "=========== "
    #print "=========== "
    #
    #print "charCounter ==="
    #print lcharCounter
    #print "biCharCounter ==="
    #print lbicharCounter

    
    
    #print edit_distance(argv[1],argv[2],int(argv[3]))
    #print uniform_cost_edit_distance("pepaa","pepe",0.01
    #print "Edit Distance: " + str(edit_distance(argv[1],argv[2]))
    #print findEditOperation(argv[1],argv[2])
    
    #for i in trainNoisyChannel("./data/edit1s.txt"):
    #for i in trainNoisyChannel("./edits1mm.txt"):    
    #  print i
    #  
    #matrices = trainNoisyChannel("./edits1mm.txt")
    #
    #a = empirical_cost_edit_distance("pnd","and",*matrices)
    #print "Empirical Cost"
    #print a
                                
    #
    #assert findEditOperation("hello","hello") == []
    #assert findEditOperation("abcde","hello") == []
    #
    ##Delete
    #assert findEditOperation("ello","hello") == [0, ('#', 'h')]
    #assert findEditOperation("hllo","hello") ==[0, ('h', 'e')]
    #assert findEditOperation("hell","hello") == [0, ('l', 'o')]
    #
    ## Substitution
    #assert findEditOperation("pello","hello") == [1, ('p', 'h')]
    #assert findEditOperation("heplo","hello") == [1, ('p', 'l')]
    #assert findEditOperation("hellp","hello") == [1, ('p', 'o')]
    #
    ## Transposition
    #assert findEditOperation("ehllo","hello") == [2, ('h', 'e')]
    #assert findEditOperation("hlelo","hello") == [2, ('e', 'l')]
    #assert findEditOperation("helol","hello") == [2, ('l', 'o')]
    #
    ## Insertion
    #assert findEditOperation("phello","hello") == [3, ('#','p')]
    #assert findEditOperation("hpello","hello") == [3, ('h','p')]
    #assert findEditOperation("hellop","hello") == [3, ('o','p')]
    
    
    assert findEditOperation2("hello","hello") == []
    assert findEditOperation2("abcde","hello") == []
    
    #Delete
    assert findEditOperation2("ello","hello") == [0, ('#', 'h')]
    assert findEditOperation2("hllo","hello") ==[0, ('h', 'e')]
    assert findEditOperation2("hell","hello") == [0, ('l', 'o')]
    
    # Substitution
    assert findEditOperation2("pello","hello") == [1, ('p', 'h')]
    assert findEditOperation2("heplo","hello") == [1, ('p', 'l')]
    assert findEditOperation2("hellp","hello") == [1, ('p', 'o')]
    
    # Transposition
    assert findEditOperation2("ehllo","hello") == [2, ('h', 'e')]
    assert findEditOperation2("hlelo","hello") == [2, ('e', 'l')]
    assert findEditOperation2("helol","hello") == [2, ('l', 'o')]
    
    # Insertion
    assert findEditOperation2("phello","hello") == [3, ('#','p')]
    assert findEditOperation2("hpello","hello") == [3, ('h','p')]
    assert findEditOperation2("hellop","hello") == [3, ('o','p')]
    
