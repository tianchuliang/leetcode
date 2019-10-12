from typing import *
import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        direclty use use min heap without turning things to negative
        """
        hp = [float('-inf') for i in range(k)] # note we are only keep k spots
        heapq.heapify(hp)
        
        for n in nums:
            if n > hp[0]:
                heapq.heappop(hp) # take out a top small element,
                                  # stick n to the bottom of the heap
                                  # after all elements put into heap
                                  # the heap's all k -inf values are substituted with
                                  # top k largest element, with top element being the kth largest
                heapq.heappush(hp,n)
        return hp[0]

if __name__=="__main__":
	sol = Solution()
	tst = [3,2,1,5,6,4]
	assert sol.findKthLargest(tst,2) == 5
