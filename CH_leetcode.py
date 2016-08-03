# -*- coding: utf-8 -*-
a = [10,15,1,3,5,8]

# selection sort complexity o(n*2)

def selsort(a):
    n=len(a)
    for i in xrange(n):
        mini=i
        for ii in xrange(i+1,n):
            if a[ii] < a[mini]:
              mini = ii
        a[i], a[mini] = a[mini], a[i]    
    return a
 
# insertion sort
def insertsort(a):
    for i in xrange(1,len(a)):
        tmp = a[i]
        j = i-1
        while j>=0 and a[j]>tmp:
             a[j+1]=a[j]
             j -=1
        a[j+1]=tmp
    return a
 
# jump game
def jump_game(a):
    step=a[0]
    for i in range(1,len(a)):
        if step > 0:
          step -=1
          step = max(a[i],step)       
        else:
            return False
    return True
          

#string multiply
def multiply( num1, num2):
    num1 = num1[::-1]; num2 = num2[::-1]
    arr = [0 for i in range(len(num1)+len(num2))]
    for i in range(len(num1)):
        for j in range(len(num2)):
            arr[i+j] += int(num1[i]) * int(num2[j])
    ans = []
    for i in range(len(arr)):
        digit = arr[i] % 10
        carry = arr[i] / 10
        if i < len(arr)-1:         # example 11 -> digit = 1 and carry = 1 and go to next arr[i+1] as 個位數
            arr[i+1] += carry
        ans.insert(0, str(digit))
    while ans[0] == '0' and len(ans) > 1:   # remove first few digits with 0 ahead
        del ans[0]
    return ''.join(ans)          

# count and say
def countAndSay(n):
    s = '1'        
    for i in xrange(n-1):            
        prev = newS = ''            
        num = 0            
        for curr in s:                
            if prev != '' and prev != curr:     # 21 --> prev is '2' and 2 !=1 then get the previous string                
                newS += str(num) + prev                    
                num = 1                
            else:                    
                num += 1                
            prev = curr            
        newS += str(num) + prev                 # 有遇到不同的num 所以跳出loop要再把前面的加進來       
        s = newS        
    return s

#remove element and return the length
def reElement(num,rm_num):
    j=0
    n=len(num)
    for i in xrange(n):
        if(num[i]!=rm_num):
            num[j]=num[i]
            j+=1
    return j        

def reElement1(num,target):
    rmcnt=num.count(target)
    for i in xrange(rmcnt):
       num.remove(target)
    return len(num)   

#remove duplicate and return the length
def rmduplicate(num):
    j=0
    n=len(num) 
    for i in xrange(0,n):
        if(num[i]!=num[j]):
            num[j+1],num[i] = num[i], num[j+1] 
            j=j+1
    return j+1

def singleNum(a):
    res=0
    for i in a:
        res=res^i
    return res

#reverse integer
def revInt(num):
    answer=0
    if num < 0:
        sign=-1
    else:
        sign=1
    num=abs(num)
    while num>0:
        answer = answer*10 + num%10
        num = num/10
    return answer*sign

#return int sqrt(x)
def sqrtInt(x):
    if x==0:
        return 0
    i=1
    j=x/2+1
    while i<=j:
        mid = (i+j)/2
        if mid**2 == x:
            return mid
        elif mid**2>x:
            j=mid-1
        else:
            i=mid+1
    return j



def sqrt_int(num):
   l=1
   r=num/2 +1
   while l<=r:
     mid = (l+r)/2
     if mid*mid == num: return mid
     elif mid*mid < num: l=mid
     else: r = mid
   return r

# rotated and sorted array with duplicated num
def findMinII(num):
    L=0
    R=len(num)-1
    while L<R and num[L]>=num[R]:
        mid = (L+R)/2
        if num[mid]<num[R]:
            R = mid
        elif num[mid]>num[L]:
            L = mid +1
        else:
            L = L+1
    return num[L]
    
# rotated and sorted array without duplicated num
def findMinI(num):
    L=0
    R=len(num)-1
    while L<R and num[L]>num[R]:
        M = (L+R)/2
        if num[M]<num[R]:
            R = M
        else:
            L = M+1
    return num[L]
    
    
# power function
def pow(x,n):
    if n==0: return 1
    elif n<0: return 1/pow(x,n) 
    elif n%2==0: return pow(x*x,n/2)
    else: return pow(x*x,n/2)*x       
    
# maximum contiguous subsum
def max_subarray(A):
    max_ending_here = max_so_far = 0
    for x in A:
        max_ending_here = max(0,max_ending_here+x)
        max_so_far = max(max_so_far,max_ending_here)
    return max_so_far


#max product
def maxProduct(A):
    if len(A) == 0:
        return 0
    min_tmp = A[0]
    max_tmp = A[0]
    result = A[0]
    for i in range(1, len(A)):
        a = A[i] * min_tmp
        b = A[i] * max_tmp
        c = A[i]
        max_tmp = max(max(a,b),c)
        min_tmp = min(min(a,b),c)
        result = max_tmp if max_tmp > result else result
    return result
    
            
# bubble sort
def bubble(A):
    for n in xrange(len(A) -1,0 , -1):
        for i in xrange(0,n,1):
            if A[i]>A[i+1]:
               A[i],A[i+1]=A[i+1],A[i]
    return A

def partition12(num,begin,end):
    pivot = begin
    for i in xrange(begin+1,end+1):
       if num[i]<=num[begin]:
           pivot+=1
           num[i], num[pivot] = num[pivot],num[i]
    num[begin], num[pivot] = num[pivot],num[begin]
    return pivot;


def partition(array, begin, end):
    pivot = begin
    for i in xrange(begin+1, end+1):
        if array[i] <= array[begin]:
            pivot += 1
            array[i], array[pivot] = array[pivot], array[i]
    array[pivot], array[begin] = array[begin], array[pivot]
    return pivot

#find median faster than sorting 
def findMedian(a,k):
    pivot = a[0]
    less =[x for x in a if x<pivot]
    if len(less)>k:
        return findMedian(less, k)
    k=k-len(less)

    # stopping criteria
    equal=[x for x in a if x==pivot]
    cnt_equal = len(equal)
    if k<=cnt_equal:
        return pivot
    k=k-cnt_equal
    
    great=[x for x in a if x>pivot]
    return findMedian(great,k)


# Majority count
a=[5,3,2,3,1,3,3,3,3,3,3,3,1,2,2,2,5,2,3]
#a=[]
def MajorityCount(num):
    val=num[0]
    count=1
    for i in xrange(1,len(num)):
        if val == num[i]:
            count+=1
        else: #val !=num[i]
            count-=1
            if count==0:
                val = num[i]
                count=1
        print count
    return val
        

def quicksort(array, begin=0, end=None):
    if end is None:
        end = len(array) - 1
    if begin >= end:
        return
    pivot = partition(array, begin, end)
    quicksort(array, begin, pivot-1)
    quicksort(array, pivot+1, end)

# generate all permutation of a string by recursive function
def permutations(s):
    if len(s)==1:
        return s
    subs = permutations(s[1:])
    start= s[0]
    result=[]
    for perm in subs:  # all sublist generated already['bc','cb']
        for i in xrange(len(perm)+1):
            result.append(perm[:i]+start+perm[i:])
    return result
    

# sort s1 and s2 if equal then true it's O(nlogn)
# the following is O(n)
import string
def findanagram(s1,s2):
  if(len(s1)!=len(s2)): 
      return False
  num=[0 for i in xrange(26)]
  letter=string.lowercase[:26]
  for i in xrange(len(s1)):
      num[letter.index(s1[i])] +=  1
  for i in xrange(len(s2)):
      num[letter.index(s2[i])] -=  1
  if(num.count(0)==26):
    return True
  else:
    return False  
    
# quick sort
def sort(array=[12,4,5,6,7,3,1,15]):
    less = []
    equal = []
    greater = []
    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            if x == pivot:
                equal.append(x)
            if x > pivot:
                greater.append(x)
        # Don't forget to return something!
        return sort(less)+ equal +sort(greater)  
    else:  # You need to hande the part at the end of the recursion - when you only have one element in your array, just return the array.
        return array
        
def lengthOfLongestSubstring1(s):
    n=len(s)
    maxlen=0
    start = 0
    dict1={}
    for i in xrange(n):
        dict1[s[i]]=-1
    for i in xrange(n):
        if dict1[s[i]]!=-1:   # if the letter is in the previous string then compute how many unique letters before 
            while start<=dict1[s[i]]:
                dict1[s[start]]=-1 # reset to be -1 since we will restart with new start
                start+=1
        dict1[s[i]]=i        # if the letter is not in dict then set value as it location                
        if i-start+1 > maxlen : maxlen = i-start+1      
    return maxlen


def twosum(num,target):
    n=len(num)
    dict={}
    for i in xrange(n):
        x=num[i]
        if target - x in dict:
            return (dict[target-x]+1,i+1)
        dict[x]=i
        
#Permutation Sequence 
def getPermutation( n, k):
    res = ''
    k -= 1
    fac = 1
    for i in range(1, n): fac *= i
    num = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in reversed(range(n)):
        curr = num[k/fac]
        res += str(curr)
        num.remove(curr)
        if i !=0:
            k %= fac
            fac /= i
    return res     
    

def singlenum(num):
    n=len(num)
    out=num[0]
    for i in range(1,n):
        out=out^num[i]
    return out
    

#?
def anagrams(strs):
    dict = {}
    for word in strs:
        sortedword = ''.join(sorted(word))
        dict[sortedword] = [word] if sortedword not in dict else dict[sortedword] + [word]
    res = []
    for item in dict:
        if len(dict[item]) >= 2:
            res += dict[item]
    return res
        
def anagrams1(str):
    n=len(str)
    dict={}
    for i in xrange(n):
        sword = ''.join(sorted(str[i])) # string sorted will become list
        dict[sword] = [str[i]] if sword not in dict else dict[sword] + [str[i]]
#    print dict
    out=[]
    for i in dict:
        if len(dict[i])>=2:
            out.append(dict[i]) 
    return out       
                         
# robot movement number of unique path
def uniqPath(m,n):
    if m==1 and n==1:
        list=[[1]]
    elif m==1 and n>1:
        list = [[1 for i in range(n)]]     
    elif m>1 and n==1:
        list = [[1 for i in range(m)]]
    else:
        list = [[0 for i in range(n)] for j in range(m)]
        for i in range(n):
            list[0][i]=1
        for i in range(m):
            list[i][0]=1
        for i in range(1,m):
            for j in range(1,n):
                list[i][j]=list[i-1][j]+list[i][j-1]
    return list[m-1][n-1]
    

# search insert position
def insertPos(num,target):
    left=0
    right=len(num)-1
    while left<=right:
        mid=(left+right)/2
        if num[mid]<target:
            left = mid+1
        elif num[mid]>target:
            right = mid-1
        else:
            return mid
    return left
    
# sort 3 color [0,2,1,2]
def sortColor(num):
    n=len(num)
    p0=0
    p2=n-1
    i=0
    while i<=p2:
        if num[i]==0:
            num[p0],num[i] = num[i],num[p0]
            p0 += 1
            i +=1
        elif num[i]==2:
            num[p2],num[i] = num[i],num[p2]
            p2 -= 1
            # i +=1 the same place need to check again
        else:
            i +=1
    return num;
 
# last word lengthi
def lenLastWord(word):
    return len(word.split(' ')[-1]) if word.split()!=[] else 0

#Given s = "the sky is blue",
#return "blue is sky the".
# reverse word
def RevWord(s):
    return ' '.join(s.split()[::-1])

#rotate n*n matrix (transpose first and the reverse)
def rotM(matrix):
    n=len(matrix)
    for i in xrange(n):
      for j in xrange(i+1,n):
          matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
    for i in xrange(n):
      matrix[i].reverse()
    return matrix
    
# operation plus 1
def plusOne(a):  
    flag=1
    for i in xrange(len(a)-1,-1,-1):
        if a[i]+flag==10:
            a[i]=0
            flag=1
        else:
            a[i]=a[i]+flag
            return a
    if flag==1:
        a.insert(0,1)
    return a

# merge 2 sort arrays:
def merge(A,B):
    m = len(A)
    n = len(B)
    tmp = [0 for x in xrange(n+m)]
    i=j=k=0;
    while i < m and j<n:
        if A[i]<=B[j]:
            tmp[k]=A[i]
            i+=1
        else:
            tmp[k]=B[j]
            j+=1
        k+=1
    if i == m :
        while k<m+n:
            tmp[k]=B[j]
            j+=1
            k+=1
    else:
        while k<m+n:
            tmp[k]=A[i]
            i+=1
            k+=1
    return tmp     
    
# matrix zero
def MatZero(mat):
    n = len(mat[0])
    m = len(mat) 
    rowflag = [0 for i in xrange(m)]
    colflag = [0 for i in xrange(n)]
    for i in xrange(m):
        for j in xrange(n):
            if mat[i][j]==0:
                rowflag[i]=1
                colflag[j]=1

    for i in xrange(m):
        for j in xrange(n):
            if rowflag[i]==1 or colflag[j]==1:
                mat[i][j]=0
    return mat


# pascal triangle
def pascal1(numRows):
#    if numRows==0:
#        return []
#    if numRows==1:
#        return [[1]]
#    if numRows==2:
#        return [[1],[1,1]]
# if numRows>2:
        list1 = [ [] for i in xrange(numRows)]
        for i in xrange(numRows):
            list1[i] = [1 for j in xrange(i+1)]
        for i in xrange(2,numRows):
            for j in xrange(1,i):
              list1[i][j] = list1[i-1][j-1]+list1[i-1][j]
        return list1

# min path sum
def MinPathSum(tri):
    n = len(tri)
    array = [0 for i in xrange(n)]
    array[0]=tri[0][0]
    for i in xrange(1,n,1):
        for j in xrange(len(tri[i])-1,-1,-1):
            if j == 0:
                array[j]=array[j]+tri[i][j]
            elif j == len(tri[i])-1:
                array[j]=array[j-1]+tri[i][j]
            else:
                array[j] = min(array[j-1],array[j])+tri[i][j]
    return min(array)

#max profit
a = [100, 180, 260, 310, 40, 535, 695]
def maxprofit(num):
    n=len(num)
    maxp = 0
    low = num[0]
    for i in xrange(1,n):
        if low >= num[i]:          
          low = num[i]
        maxp = max(maxp,num[i]-low)
    return maxp      

#max consecutive sequence in O(n)
# a = [100, 4, 200, 1, 3, 2]
def MaxConSeq(a):
    dict1= {x:0 for x in a}
    maxL=-1
    for i in dict1:
        if dict1[i]==0:
            dict1[i] =1
            curr =i+1            # search to the right
            lenright=0
            while curr in dict1:
                lenright +=1
                dict1[curr]=1
                curr +=1
            curr = i-1           # search to the left
            lenleft=0
            while curr in dict1:
                lenleft +=1
                dict1[curr]=1
                curr -= 1
        maxL = max(maxL,lenleft+1+lenright)      
    return maxL
 
 

# dynamic programming
 
 
# match the words
def wordBreak(self, s, dict):
    dp = [False for i in range(len(s)+1)]
    dp[0] = True
    for i in range(1, len(s)+1):
        for k in range(i):
            if dp[k] and s[k:i] in dict:
                dp[i] = True
    return dp[len(s)]      
    

        
# number of unique subsequence
def numDistinct(S, T):
    m=len(S)
    n=len(T)
    dp = [[0 for i in xrange(n+1)] for j in xrange(m+1)]
    for i in xrange(m+1):
        dp[i][0]=1
    
    for i in xrange(1,m+1):
        for j in xrange(1,min(i+1, n+1)):
            if S[i-1]==T[j-1]:
                dp[i][j]=dp[i-1][j-1]+dp[i-1][j]
            else:
                dp[i][j]=dp[i-1][j]
    return dp#[len(S)][len(T)]
    

#edit distance
def minDistance(word1, word2):
    m=len(word1)+1; n=len(word2)+1
    dp = [[0 for i in range(n)] for j in range(m)]
    for i in range(n):
        dp[0][i]=i
    for i in range(m):
        dp[i][0]=i
    for i in range(1,m):
        for j in range(1,n):
            dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if word1[i-1]==word2[j-1] else 1))
    return dp[m-1][n-1]                  

# generate all subsets of list integers
def subsets(S):
    def dfs(depth, start, valuelist):
        res.append(valuelist)
        if depth == len(S): return
        for i in range(start, len(S)): # if start is n-1, actually the for loop will only run once (also a stop criteria)
            dfs(depth+1, i+1, valuelist+[S[i]])
    S.sort()
    res = []
    dfs(0, 0, [])
    return res    

def subset1(S):
    n = len(S)
    list=[[]]
    for i in xrange(1,2**n):
        new=[]
        j=0
        while i>0:
          if i%2==1:
            new.append(S[j])
          i=i/2
          j+=1
        if new not in list:list.append(new)
    return list
            
# generate all permutations
# can use recursive to generate all
def perm(S):
    if len(S)==0: return []
    if len(S)==1: return [S]
    list=[]
    for i in xrange(len(S)):
        for j in perm(S[:i]+S[i+1:]):
            new=[S[i]]+j
            if new not in list:   # not append if the list is already in the list
              list.append([S[i]]+j)            
    return list

#climb fibnac
def climb(n):
    s=[1 for i in xrange(n+1)]
    for i in xrange(2,n+1):
        s[i] = s[i-1]+s[i-2]
    return s[n]

     
# dfs all combination to the target
def dfs(candidate, target, start, valist,sol):
    n=len(candidate)
    if target == 0 :
        return sol.append(valist)
    for i in xrange(start,n):
        if target < candidate[i]:return # return nothing if target < candidate[i]
        dfs(candidate,target - candidate[i], i , valist + [candidate[i]],sol)     

def combination(candidate, target):
    candidate.sort()
    sol=[]
    dfs(candidate,target,0,[],sol)
    return sol
    
#######################################

# generate all combination
def combn(n,k):
    out=[]
    count=0
    def dfs(start, list1, count):
        if count==k: out.append(list1); return
        else:
            for i in xrange(start,n+1):
                count += 1
                dfs(i+1,list1+[i],count)          
                count -= 1                # used in last stage to change numbers
    dfs(1,[],count) 
    return out


# @return a string :"PAYPALISHIRING" ==>'PAHNAPLSIIGYIR'
def zigzag(s,nrow):
    if len(s)==1: return s
    tmp = ['' for i in xrange(nrow)]
    index=-1
    step=1
    for i in xrange(len(s)):
        index +=step
        if index == nrow:
            index = index -2
            step = -1
        elif index ==-1:
            index = index + 2
            step = 1
        tmp[index] += s[i] 
    return ''.join(tmp)
    
# not included float number isdigit() funtion
def mixsort():
    string = raw_input('Enter a string to sort: ').encode('ascii')
    str_list = string.split(' ')
    list_int = []
    list_word = []
    for i in xrange(len(str_list)):
        if str_list[i].isdigit():
            list_int.append(int(str_list[i]))
            str_list[i]=1
        else:
            list_word.append(str_list[i])
            str_list[i]=0
    list_int.sort()
    list_word.sort()
    ind_int = 0
    ind_word = 0
    for i in xrange(len(str_list)):
        if str_list[i]==1:
            str_list[i] = str(list_int[ind_int])
            ind_int +=1
        else:
            str_list[i] = list_word[ind_word]
            ind_word +=1
    return ' '.join(str_list)


    ######################
# data center problem
def DBU():
    def readInput():
        text = ""
        print "please input your data:"
        while True:
            line = raw_input()
            if line.strip() == "":
                break
            text += "%s\n" % line
        return text
    string=readInput().rstrip("\n").split("\n")    
    dict={}
    for i in xrange(1,int(string[0])+1):
        line = string[i].rstrip("\n ").split(' ')
        for j in xrange(1,int(line[0])+1):
            if line[j] not in dict:
                dict[line[j]]=str(i)
    sets=dict.keys()
    for i in xrange(1,int(string[0])+1):
        line = string[i].rstrip("\n ").split(' ')[1:]
        for j in sets:
            if j not in line:
                out=' '.join([j,dict[j],str(i)])
                print(out)
    print('done')    


def bin_search(a,k):
    l=0
    r=len(a)-1
    flag=False
    while l<=r and not flag:
        m = (l+r)/2
        if a[m]==k:
            flag=True
        elif a[m]>k:
            r=m-1
        else:
            l=m+1
    if flag:    return m
    else: return '%s not found' % k

def ExcelToNum(s):
    letters ='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    s=s.strip().upper() 
    n=len(s)
    result=0
    if n==0: return result
    else:
        for i in s:
          result = 26*result + letters.index(i)+1
    return result

# generate all subsets of list integers    
def subset(s):
    def dfs(depth,start,valuelist):
        sol.append(valuelist)
        if depth ==len(s): return
        for i in xrange(start,len(s)):
            dfs(depth+1,i+1,valuelist+[s[i]])            
    s.sort()
    sol=[]
    dfs(0,0,[])
    return sol

# get all permutations I with recursive
# remove duplicate by stop the process if s[i]==prenum permutation II
def permute(s):
    if len(s)==0: return        
    if len(s)==1: return [s]# list need to have [] otherwise can not use + in list concatenation
    s.sort()
    res=[]
    #prenum = []
    for i in xrange(len(s)):
        #if s[i] == prenum: continue
        #prenum = s[i]
        for j in permute(s[:i] + s[(i+1):]): # permute the remaining s[i] put aside and combine them all
            res.append([s[i]]+j)
    return res

# all combination of legal parenthesis    
def genParenthesis(n):
    def dfs(L,R,item):
        if R<L: return
        if L==0 and R==0:
            return sol.append(item)
        if L>0:
            dfs(L-1,R,item+'(')
        if R>0:
            dfs(L,R-1,item+')')
    sol=[]
    dfs(n,n,'')
    return sol


def combnSum(s,target):
     def dfs(s,target, start,valuelist):
         if target == 0: return res.append(valuelist)
         for i in xrange(start,len(s)):
             if target > 0 :             
                 dfs(s,target-s[i],i,valuelist+[s[i]])
     s.sort() # for non-decreasing order
     res=[]
     dfs(s,target,0,[])
     return res

# consecutive sum of subarray [1,-1,3,4,-3,2,5] target = 7 or 8
def con_sum(num,T):
    def dfs(start, target, valuelist):
        for i in xrange(len(valuelist)-1):
            if (num.index(valuelist[i+1]) - num.index(valuelist[i]))!=1:
                return
        if target == 0: return sol.append(valuelist)        
        for i in xrange(start,len(num)):        
          if target !=0:
              dfs(i+1,target - num[i],valuelist+[num[i]])    

    sol=[]
    dfs(0,T,[])
    return sol
    
def con_sum1(num,T):
     size = len(num)
     res=[]
     for i in xrange(size):
        sum_array=0
        for j in xrange(i,size):
             sum_array += num[j]
             if sum_array==T:
                 res.append(num[i:(j+1)])   
     return res
     
# Given "25525511135",
# return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)
def restoreIP(s1):
    def dfs(s,block,valuelist):
        if block==4:
            if s=='':
                return res.append(valuelist[1:]) # first is , when call dfs function
        for i in xrange(1,4):
            if(int(s[:i])<255):
                dfs(s[i:],block,valuelist+','+s[:i])
        
    res=[]
    dfs(s1,0,'')
    return res

#restore IP address
def restoreIpAddresses(self, s):
    def dfs(s, sub, ips, ip):
        if sub == 4:                                        # should be 4 parts
            if s == '':
                ips.append(ip[1:])                          # remove first '.'
            return
        for i in range(1, 4):                               # the three ifs' order cannot be changed!
            if i <= len(s):                                 # if i > len(s), s[:i] will make false!!!!
                if int(s[:i]) <= 255:
                    dfs(s[i:], sub+1, ips, ip+'.'+s[:i])
                if s[0] == '0': break                       # make sure that res just can be '0.0.0.0' and remove like '00'
    ips = []
    dfs(s, 0, ips, '')
    return ips
        
        
        
# permutation with recursive function
def perm_recur(arr):
    if len(arr)==0: return []
    if len(arr)==1: return [arr]
    res=[]
    for i in xrange(len(arr)):
        for j in perm_recur(arr[:i]+arr[i+1:]):
            res.append([arr[i]]+j)
    return res
    
def isPalindrome(s):
    for i in xrange(len(s)/2):
        if s[i]!=s[len(s)-1-i]:return False
    return True

def valid_parentheses(s):
     stack=[]
     for i in s:
         if i=="(" or i=="[" or i=="{":
             stack.append(i)
         elif i==")":
             if stack==[] or stack.pop()!="(": return False
         elif i=="]":
             if stack==[] or stack.pop()!="[": return False
         elif i=="}":
             if stack==[] or stack.pop()!="{": return False
     if len(stack)!=0: return False
     else: return True

"()()"
"(())"
")())(())"

def long_valid_parentheses(s):
    maxlen=0
    ind=-1  # not start from 0 (0 is also a one count)
    l=[]
    for i in xrange(len(s)):
        if s[i]=='(':l.append(i)  # save last index
        else:
            if l==[]:
                ind=i
            else:
                l.pop()
                if l==[]: maxlen=max(maxlen,i-ind)      
                else: maxlen = max(maxlen, i-l[len(l)-1])       
    return maxlen


#Given [100, 4, 200, 1, 3, 2],
#The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.
#Your algorithm should run in O(n) complexity.


def longestConsecutive(s):
    dict_s={x:-1 for x in s} # initial the dictionary for key as the number and value as -1
    maxlen=0
    for i in dict_s:
        if dict_s[i]==-1:
            dict_s[i]=1
            right=i+1
            rightlen=0
            while right in dict_s:
                dict_s[right]=1
                right+=1
                rightlen+=1
            left=i-1
            leftlen=0
            while left in dict_s:
                dict_s[left]=1
                left-=1
                leftlen+=1
            maxlen = max(maxlen,rightlen+1+leftlen)
    return maxlen
    
def largest_num(num):
    num = [str(x) for x in num]
    num.sort(cmp= lambda x,y: cmp(y+x,x+y))
    largest = ''.join(num) 
    return largest.lstrip('0') or '0'  # in case 0 of the list
    
# two same numbers within k distance
def nearby_k(num,k):
    dict1={}
    for i in xrange(len(num)):
        if num[i] not in dict1:
            dict1[num[i]]=i
        else:
            if i - dict1[num[i]]<=k:
                return True
            else:
                dict1[num[i]]=i
    return False
          

# word ladder
#dict=["hot","dot","dog","lot","log"]
#start = "hit"
#end = "cog"
def ladderLength(start, end, dict):
    dict.add(end)
    q = []
    q.append((start, 1))
    while q:
        curr = q.pop(0)
        currword = curr[0]; currlen = curr[1]
        if currword == end: return currlen
        for i in range(len(start)):
            part1 = currword[:i]; part2 = currword[i+1:]
            for j in 'abcdefghijklmnopqrstuvwxyz':
                if currword[i] != j:
                    nextword = part1 + j + part2
                    if nextword in dict:
                        q.append((nextword, currlen+1)); 
                        dict.remove(nextword)
    return 0

######## tree class and treenode class ####
# simple tree 
class Tree:
    def __init__(self):
        self.root = None
        
class TreeNode:
    def __init__(self,v):
        self.val=v
        self.left=None
        self.right = None
        self.parent = None
        
class Solution:
    def TreeInsert(self,T,z):
        y = None
        x = T.root
        while x!=None:
            y=x
            if z.val<x.val:
                x=x.left
            else:
                x=x.right
        z.parent =y
        if y==None:
            T.root = z
        elif z.val < y.val:
            y.left = z
        else:
            y.right = z
            
    def InOrder(self,T):
        if T!=None:
            self.InOrder(T.left)
            print T.val
            self.InOrder(T.right)

    def PreOrder(self,x):
        if x!=None:
            print x.val
            self.PreOrder(x.left)
            self.PreOrder(x.right)

    def TreeMax(self,x):
        while x.right!=None:
            x=x.right
        return x
        
    def TreeMin(self,x):
        while x.left !=None:
            x=x.left
        return x

    def maxDepth(self,x):
        if x==None:
            return 0
        else:
            return max(self.maxDepth(x.left),self.maxDepth(x.right))+1

                
T = Tree()
nodes = [6,18,3,7,17,20,2,4,13,9]
nodes = [6,3,7]
s = Solution()
for node in nodes:
    s.TreeInsert(T,TreeNode(node))
d=s.maxDepth(T.root)

s.InOrder(T.root)
s.PreOrder(T.root)
a=s.TreeMax(T.root)
a_min=s.TreeMin(T.root)
