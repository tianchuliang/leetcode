from typing import *
import collections

class Solution:
    def longestIncreasingPath_dfs_memo(self, matrix: List[List[int]]) -> int:
        """
            recursive DFS + Memoization
            
            recursive DFS: finds longest increasing path STARTing from any cell, do this for every cell.
            Memoization: stores partial results so dont need to overcompute.
            ...
            ...
            
            (DP is hard in this case, because there is no 
            clear direction on the dependency, we don't know how
            to properly iterate through cells to fill the DP relation
            
            Maybe need something like topological sort, then DP based on sort)
            """

        def dfs(matrix: List[List[int]], i: int, j: int, memo: List[List[int]]) -> int:
            if memo[i][j] != 0:
                return memo[i][j]
            for nb in get_neighbor(i, j, m, n):
                # if maintaining increasing trend, keep dfs
                if matrix[nb[0]][nb[1]] > matrix[i][j]:
                    memo[i][j] = max(memo[i][j], dfs(matrix, nb[0], nb[1], memo))

            # visited here, at least, even if there is no more increasing neighbor
            # the node i,j itself is a singleton increasing sequence of length 1
            memo[i][j] += 1
            return memo[i][j]

        """
            ------------------------------
            """

        def get_neighbor(i, j, m, n):
            nbs = []

            for d in dirs:
                nnb = [i + d[0], j + d[1]]
                if nnb[0] >= 0 and nnb[0] < m and nnb[1] >= 0 and nnb[1] < n:
                    nbs.append(nnb)
            return nbs

        """
            ------------------------------
            """
        dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]

        if len(matrix) == 0:
            return 0

        m, n = len(matrix), len(matrix[0])
        memo = [
            [0 for i in range(n)] for j in range(m)
        ]  # stores longest length starting at i,j
        ans = 0
        for i in range(m):
            for j in range(n):
                ans = max(ans, dfs(matrix, i, j, memo))
        return ans


if __name__ == "__main__":
    sol = Solution()
    testmat = [[3,4,5],[3,2,6],[2,2,1]]
    assert sol.longestIncreasingPath_dfs_memo(testmat) == 4
