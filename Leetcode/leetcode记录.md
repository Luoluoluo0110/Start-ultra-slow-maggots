| Types of Algorithm Problems | Link                                                         |                                                              |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Dynamic Programming         | [118. 杨辉三角 - 力扣（LeetCode）](https://leetcode.cn/problems/pascals-triangle/?envType=study-plan-v2&envId=top-100-liked) |                                                              |
|                             | [198. 打家劫舍 - 力扣（LeetCode）](https://leetcode.cn/problems/house-robber/submissions/730098148/?envType=study-plan-v2&envId=top-100-liked) |                                                              |
|                             | [279. 完全平方数 - 力扣（LeetCode）](https://leetcode.cn/problems/perfect-squares/description/?envType=study-plan-v2&envId=top-100-liked) |                                                              |
|                             | [322. 零钱兑换 - 力扣（LeetCode）](https://leetcode.cn/problems/coin-change/?envType=study-plan-v2&envId=top-100-liked) |                                                              |
|                             | [139. 单词拆分 - 力扣（LeetCode）](https://leetcode.cn/problems/word-break/?envType=study-plan-v2&envId=top-100-liked) |                                                              |
|                             | [300. 最长递增子序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-increasing-subsequence/?envType=study-plan-v2&envId=top-100-liked) | 这题有个坑，最后要max(ans), 因为ans[i], 表示的是i结尾的字符最长递增子序列，问的最长子序列，ans[n-1]是以第n个结尾的最长子序列 |
|                             | [152. 乘积最大子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-product-subarray/?envType=study-plan-v2&envId=top-100-liked) | 这题有点坑，看到最大非空连续子数组，想到滑动窗口，但是因为会有0和负数，导致乘积反转，所以用动态规划 |
|                             | [53. 最大子数组和 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-subarray/) | 详解在下面，应该先看这个题目，再看乘法版本152                |
|                             | [416. 分割等和子集 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-product-subarray/description/?envType=study-plan-v2&envId=top-100-liked) | 01背包经典题目                                               |

```
# 最大子数和 leetcode 53
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ans = nums[0] # 全局的最大子数组和
        dp1 = 0 # 当前的数字结束的最大子数组和
        
        for i in range(len(nums)):
            # 如果上一个数字结尾的最大子数和是正的，带着一起走
            # 如果上一个数字结尾的最大子数和是负的, 扔掉前面的
            dp1 = max(dp1, 0) + nums[i]
            # 更新全局最大值
            ans = max(ans, dp1)
        return ans
```

```
# 最大字数积 leetcode 152
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        ans = nums[0]
        cur_min = nums[0]
        cur_max = nums[0]
        
        n = len(nums)
        for i in range(1, n):
            if nums[i] < 0:
                # 遇到负数。最大变最小，最小变最大
                cur_min, cur_max = cur_max, cur_min
            cur_max = max(nums[i], cur_max * nums[i])
            cur_min = min(nums[i], cur_min * nums[i])
            ans = max(ans, cur_max)
        return ans
```

```
# leetcode 436
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        s = sum(nums)
        if s % 2 != 0:
            return False
        target = s // 2
        # dp[i]: 能不能凑出i
        dp = [True] + [False] * target
        
        for num in nums:
            # 01背包倒叙遍历，每个数只能用一次
            for i in range(target, num-1, -1):
                dp[i] = dp[i] or dp[i-num]
        return dp[target]
```

