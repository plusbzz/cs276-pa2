import sys

def findEditOperation(finalWord,intendedWord):
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


def edit_distance(a,b,cutoff=sys.maxint):
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

