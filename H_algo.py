######################################################################
# Key point is the median of two 
# SORTED lists DIVIDES the total 
# of two lists into two equal length parts
# one part always is less than the other part
# https://leetcode.com/problems/median-of-two-sorted-arrays/solution/
######################################################################

def median_of_two_sorted_arrs(A,B):
    m,n = len(A),len(B)
    # always fix on a larger arr size, n > m
    if m > n:
        A,B,m,n = B,A,n,m
    
    # edge case:
    if n == 0:
        raise ValueError
    
    # define range of index i, 
    # i traverses m, the smaller length
    imin,imax,half_len = 0,m,(m+n+1)/2
    
    # start a binary search in the range [imin,imax]
    # we are trying to search the proper index i
    # that will divides the lists 
    # in this binary search, the return condition 
    # is the median condition,
    # when we need to search for a larger or smaller i,
    # we adjust the range, much like binary search
    while imin <= imax:
        i = int((imin+imax)/2)
        j = int(half_len - i )
        
        # by setting j = halflen - i, we make 
        # sure left and right parts are the same length
        
        # case 1, B[j-1]<=A[i] && A[i-1]<=B[j]
        # since A, B are already sorted (i.e A[i-1]<=A[i]
        # and B[j-1]<=B[j]), this meresult
        # we've found the median
        if B[j-1] <= A[i] and A[i-1]<= B[j]:
            if i == 0:
                max_of_left = B[j-1]
            elif j == 0:
                max_of_left = A[i-1]
            else:
                max_of_left = max(A[i-1],B[j-1])
            
            if i == m: 
                min_of_right = B[j]
            elif j == n:
                min_of_right = A[i]
            else:
                min_of_right = min(A[i],B[j])
            
            if (m+n)%2 == 1:
                return max_of_left
            else:
                return (max_of_left+min_of_right)/2
        elif B[j-1] > A[i]:
            # i and j go in reverse direction
            # we want A to be larger, and B to be
            # smaller
            # to do this, we can only increment i
            # which decrements j
            imin = i+1
            
        elif A[i-1] > B[j]:
            imax = i - 1
##########################################################
# since the problem is super open ended
# there are wayyyyy to many edge cases
# first attempt would be to use a finite
# state machine to solve this
# https://leetcode.com/problems/valid-number/discuss/23728\
#/A-simple-solution-in-Python-based-on-DFA
###########################################################
def valid_number(s):
    # define states
    # list of states
    # index stands for state
    # state[0] is state q0
    
    state = [{},\
             {'blank':1,'sign':2,'digit':3,'dot':4},\
             {'digit':3,'dot':4},\
             {'digit':3,'e':6,'dot':5,'blank':9},\
             {'digit':5},\
             {'digit':5, 'e':6, 'blank':9},\
            {'sign':7, 'digit':8},\
            {'digit':8},\
            {'digit':8, 'blank':9},\
            {'blank':9}]
    curS = 1 
    for c in s: 
        if c >= '0' and c <='9':
            c = 'digit'
        if c == ' ':
            c = 'blank'
        if c in ['+','-']:
            c = 'sign'
        if c not in state[curS].keys():
            return False
        nxtS = state[curS][c]
        curS = nxtS
    if curS not in [3,5,8,9]:
        return False
    return True


# the idea is to loop through
# every pt, and construct a line 
# with every other pt'. Record
# the slopes as a dictionary, with 
# slope as key. Note, this dictionary is 
# only for this particular pt. Meaning,
# how many other points share the same slope
# as well as being through pt.
def max_pts_on_a_line(points):
    # need to define helper funcs 
    # to avoid floating pt precision problem
    
    # use greatest common divisor and fraction
    # to express floating point in terms of 
    # rationals (i.e integer division)
    # gcd here assures 4/8 actually is 1/2, 
    # this will make sure the dictionary of 
    # slopes don't duplicate/explode unnecessarily
    def gcd(a,b):
        if b == 0:
            return a
        return gcd(b,a%b)
    def frac(x,y):
        g = gcd(x,y)
        return (x//g, y//g)
    
    l = len(points)
    m = 0 
    for i in range(l):
        ptdict = {'inf':1} # infinite slope 
        same = 0 
        ix = points[i].x
        iy = points[i].y
        # other points
        for j in range(i+1,l):
            jx = points[j].x
            jy = points[j].y
            if ix == jx and iy == jy:
                same += 1
                continue
            if ix == jx: # on the same verticle line, infinite slope
                slope = 'inf'
            else:
                slope = frac(jy-iy,jx-ix)
            
            if slope not in ptdict.keys():
                ptdict[slope] = 1 
            ptdict[slope]+= 1
            
        print(ptdict)
        m = max(m,max(ptdict.values())+same)
    return m

# word Ladder II
# for this one, we basically build the BFS tree
# level by level with words in the wordList
# and with a smart list comprehension trick, we 
# will get the final resultwer
def word_ladder_ii(beginWord,endWord,wordList):
    import collections
    import string
    # use collections defaultdict allows us to keep track of 
    # tree structure, at the same time add path
    # info to nodes of interest
    if beginWord not in wordList:
        wordList+=beginWord
    if endWord not in wordList:
        wordList+=endWord

    thislevel = {beginWord}
    parents = collections.defaultdict(set)

    while thislevel and endWord not in parents:
        nextlevel = collections.defaultdict(set)
        for node in thislevel:
            for char in string.ascii_lowercase:
                for i in range(len(beginWord)):
                    n = node[:i]+char+node[i+1:]
                    if n in wordList and n not in parents:
                        nextlevel[n].add(node) # adding node while exploring 
                                                # tree allows us to build the 
                                                # path back up
        thislevel = nextlevel
        parents.update(nextlevel)
    result = [[endWord]]
    while result and result[0][0] != beginWord:
        result = [[p]+r for r in result for p in parents[result[0][0]]]
    return result

"""
we can rephrase this as a problem about
the prefix sums of A. 
Let P[i] = sum(A[i]) for i = 0...i-1
We want the smallest j - i such that
j > i and 
P[j] - P[i] >= K
"""
 
def shortest_subarray(A,k):
    import collections
    N = len(A)
    B = [0] * (N+1)
    
    # assemble cumulative sum arr B; O(N)
    for i in range (N): B[i+1] = B[i] + A[i]
    
    # initialize deque d, to keep track
    # of 
    d = collections.deque()
    res = N + 1 
    
    # loop through every ending position of B
    # i.e loop through every j 
    for i in xrange(N+1):
        # continuously find shorter (popleft)
        # and shorter subarrs , i.e, 
        # index i, 
        # that satisfies the 
        # K condition
        while d and B[i] - B[d[0]] >= k:
            subarr_len = i - d.popleft()
            res = min(res, subarr_len)
        
        # while loop to make sure 
        # the d-deque actually contains
        # indices that are increasing
        # B's value
        while d and B[i] - B[d[-1]] <= 0:
            d.pop()
        d.append(i)
    return res if res <= N else -1 

##############################
## text justification
##############################
def text_justification(words,maxWidth):
    res,cur,num_of_letters = [],[],0

    for w in words:
        # there is need for rearrangement
        if num_of_letters + len(w) + len(cur) > maxWidth:
            for i in range(maxWidth - num_of_letters):
                cur[i%(len(cur) -1 or 1)]+= ' '
            res.append(''.join(cur))
            cur,num_of_letters = [],0
        cur +=[w]
        num_of_letters += len(w)
    return res+ [' '.join(cur).ljust(maxWidth)]

# don't use brute force O(N^2)
# use merge sort
class reversePairsSolution(object):
    def __init__(self):
        self.cnt = 0 
    def reversePairs(self,nums):
        def msort(lst):
            # merge sort body 
            L = len(lst)
            if L < 1:
                return lst 
            else:
                return merge(msort(lst[:int(L/2)]),msort(lst[int(L/2):]))
        # in this method, we are NOT really
        # sorting things, but rather
        # in the if and else block
        # we are summing how many times left is 
        # more than 2 times larger than right, i.e a valid SWAP
        # but in the end, the merge function still has to return 
        # the proper sorted list though
        def merge(left,right):
            l,r = 0,0
            while l <(len(left)) and r < len(right):
                if left[l] <= 2 * right[r]:
                    l += 1
                else:
                    self.cnt +=1
                    r += 1
            return sorted(left+right)
    
        msort(nums)
        return self.cnt

def wildcard_matching(s,p):
    
    # main DP recursion
    # i indicates s
    # j indicates p
    # T[i][j] = (T[i-1][j] or T[i][j-1]) if p[j-1] == '*'
    #         = T[i-1][j-1] if (p[j-1] == '?' or p[j-1] == s[i-1])
    #         = False
    # where T[i][j] is the subproblem of 
    # True of False, s[:i-1] and p[:j-1] is a vliade
    # wildcard matching
    T = [[None for _ in range(len(p)+1)] for _ in range(len(s)+1)]
    T[0][0] = True
    # base case for T[0]
    for i in range(1,len(p)+1):
        if p[i-1] == '*':
            T[0][i] = T[0][i-1]
        else:
            T[0][i] = False
    # base case for T[:][0]
    for i in range(1,len(s)+1):
        T[i][0] = False
    
    # main recursion
    for i in range(1,len(s)+1):
        for j in range(1,len(p)+1):
            if p[j-1] == '?' or p[j-1] == s[i-1]:
                T[i][j] = T[i-1][j-1]
            elif p[j-1] == '*':
                T[i][j] = (T[i-1][j]) or (T[i][j-1])
            else:
                T[i][j] = False
    
        
    return T[len(s)][len(p)]

# Key observation:
# avg of two lists is also the avg of A 
#
def split_arr_with_same_avg(A):
    from fractions import Fraction
    N = len(A)
    S = sum(A)
    A = [z - Fraction(S,N) for z in A]
    if N == 1:return False
    
    left = {A[0]} # all powersets of first Half of A
                # if left sums to zero then True
                # if any powerset of left plus a 
                # powerset of right is zero, then True 
    right = {A[-1]}
    for i in range(1,int(N/2)):
        left = {z + A[i] for z in left}|left|{A[i]} # adding powersets
               # always keep a set to be sum of all current left elements
                # then or it with current left
                # and or it with new A[i] element
        print(left)
    if 0 in left: return True
    for i in range(int(N/2),N-1):
        right = {z + A[i] for z in right} | right | {A[i]}
        
    if 0 in right: return True
    
    sleft = sum(A[i] for i in range(int(N/2)))
    sright = sum(A[i] for i in range(int(N/2),N))
    
    return any(-ha in right and (ha, -ha) != (sleft, sright) for ha in left)

# The key to this problem is to 
# always choose the smallest next cell to go, among
# all possible adjacent cells
# we make sure to use 'seen' set 
# so that we will always keep the heap
# as the container for candidate cells
def swim_in_water(grid):
    import heapq
    # initalize size, heap, set of visited nodes, res
    N, pq, seen, res = len(grid),[(grid[0][0],0,0)],set([(0,0)]),0
    while True:
        T,x,y = heapq.heappop(pq)
        res = max(res,T)
        if x == y == N-1:
            return res
        for i,j in [(x+1,y),(x,y+1),(x-1,y),(x,y-1)]:
            if (i,j) not in seen:
                if 0 <= i < N and 0<=j < N:
                    heapq.heappush(pq,(grid[i][j],i,j))
                    seen.add((i,j))

# This question was briefly mentioned in 
# cracking the coding interview book 
# early pages
# heaps always keeps mins of a bigger half
# and a smaller half
# therefore the median would be either one or avg
#  the min-heap keeps the larger half of numbers
# the max-heap (implemented as - of minheap) keeps the 
# smaller half of the numbers
# then, intuitively, the root of max-heap, is the max of 
# small numbers
# root of the min-heap, is the min
# of large numbers
# as long as two heaps are close to the same size
# we can get median all we want
class MedianFinder(object):
    import heapq
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.small = [] # this keep large numbers
        self.large = [] # this keep very small numbers 

    def addNum(self, num):
        """
        :type num: int
        :rtype: void
        """
        # adding new number to two heaps
        # making sure we are balancing among the small and
        # large heaps
        heapq.heappush(self.small,-heapq.heappushpop(self.large,num))
        
        # balance two lengths
        if len(self.large) < len(self.small):
            heapq.heappush(self.large,-heapq.heappop(self.small))
        

    def findMedian(self):
        """
        :rtype: float
        """
        # median should be in the larger lump 
        if len(self.large) > len(self.small):
            return float(self.large[0])
        return (self.large[0] - self.small[0])/2.0
            

# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

def min_refuel_stops(target,startFuel,stations):
    import heapq
    past_stations = [] # this will be a max_heap
    stations.append((target,float('inf')))
    
    incremental_miles = 0
    res = 0 
    fuelLeft = startFuel
    
    for locale,fuelamount in stations:
        # at every new station
        # first update fuel left
        fuelLeft -= locale - incremental_miles

        # check if we succeeded
        if fuelLeft >=0 and incremental_miles == target:
            return res
        # if we ran out of fuel
        # we can 'regret' and refuel at previous most 
        # capacious gas station till we are full
        # note, we are not going back in time
        # we are still at the current location, station. 
        # we are just regretting previous mishaps
        while fuelLeft < 0 and past_stations:
            fuelLeft += -heapq.heappop(past_stations)
            res +=1
        # if there is no possible way we can refuel 
        # back, we decalre failure
        if fuelLeft < 0:return -1
        
        # in the end, if everything above executes fine
        # and we are still in the game
        # let's add this station to the heap,
        # for now, we are not refueling at this station,
        # but how knows, we may regret later and pop it off
        # the heap
        heapq.heappush(past_stations,-fuelamount)
        
        # update our incremental miles
        incremental_miles = locale
    return res

# Algo:
# Compare the heads of every list, pick the smallest one to 
# the new list. 
# Do this with priority queue (implemented with heap underneath)
# overall runtime would be O(NlogK)
def merge_k_sorted_linkedlists(lists):
        from Queue import PriorityQueue
        newhead = ListNode(0)
        current = newhead
        q = PriorityQueue()

        # putting the headnodes of each list
        # to the q is the same as putting 
        # the whole list there, because we are 
        # dealing with LINKED list here. 
        for headnode in lists:
            if headnode:q.put((headnode.val,headnode))
        while q.qsize()>0:
            current.next = q.get()[1] # index [1] to get the actual node
            current = current.next # move the new position in the new list
            # while we are at it,
            # we put the current's next node to q
            if current.next:
                q.put((current.next.val,current.next))
        return newhead.next

# Algo:
# nearest palindrome
def nearestPalindromic( S):
    """
    :type n: str
    :rtype: str
    """
    def palindsize(x):
        return abs(int(S) - int(x))
    
    L = len(S)
    #Start with basic candidates, 
    # ones which start with 10..., cause those ones
    # could potentially be the most minimizing ones
    # covering edgecases like 100, which should be 99 instead of 101
    cands = [str(10**l+tiny_delta) for l in (L-1,L) for tiny_delta in (-1,1)]   
    prefix = S[:(L+1)/2]        
    P = int(prefix)
    
    # now, for the first part, +/- one of the middle or middle parity
    for firstpart in (P-1, P, P+1):
        firstpart = str(firstpart)
        secondpart = firstpart[:-1] if L%2 else firstpart
        cands.append(firstpart + secondpart[::-1])

    result = None
    for cand in cands:
        if cand != S and not cand.startswith('00'):
            # if result is still null, take the cand for now
            # or if candidate is a smaller palindrome, take it. 
            if (result is None or palindsize(cand) < palindsize(result)):
                result = cand
            # if current cand palind size is the same with previous result
            # AND the absolute value of cand is smaller, take the cand
            elif palindsize(cand) == palindsize(result) and int(cand) < int(result):
                result = cand
    return result


