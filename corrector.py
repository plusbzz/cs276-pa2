
import sys

queries_loc = 'data/queries.txt'
gold_loc = 'data/gold.txt'
google_loc = 'data/google.txt'

alphabet = "abcdefghijklmnopqrstuvwxyz0123546789&$+_' "


def jaccard_coeff(s1,s2):
  s1 = set([(t1+t2) for t1,t2 in zip(s1[:-1],s1[1:])])
  s2 = set([(t1+t2) for t1,t2 in zip(s2[:-1],s2[1:])])
  
  return (1.0*len(s1.intersection(s2)))/len(s1.union(s2))


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
