---
title: Python篇(1)
published: 2025-11-20
description: 'python基础，遇到就补'
image: ''
tags: [学习笔记]
category: 'AI'
draft: false 
lang: ''
---
# 基础语法

参考python官方文档学习即可

>python官方文档直通车：https://docs.python.org/3.12

# Numpy库

用于科学计算的基础库

## 1. 数组/矩阵创建

```Python
import numpy as np

# 一维数组
arr_1 = np.array([1, 2, 3])
"""
[1 2 3]
"""

# 2x3矩阵
arr_2 = np.array([[1, 2, 3], [4, 5, 6]])
"""
[[1 2 3]
 [4 5 6]]
"""

# 全0矩阵
arr_3 = np.zeros((3, 3), dtype=int)
"""
[[0 0 0]
 [0 0 0]
 [0 0 0]]
"""

# 全1矩阵
arr_4 = np.ones((2, 2), dtype=int)
"""
[[1 1]
 [1 1]]
"""

# 创建全指定值的矩阵
arr_5 = np.full((2, 2), 10, dtype=int)
"""
[[10 10]
 [10 10]]
"""

# 创建等差数列 (start, stop, step), stop不可取
arr_6 = np.arange(0,10,2)
"""
[0 2 4 6 8]
"""

# 创建等间隔数列 (start, stop, num), 包含stop
arr_7 = np.linspace(0,1,5)
print(arr_7)
"""
[0.   0.25 0.5  0.75 1.  ]
"""
```

`dtype`就是指定数据类型的，只要有可能为浮点数，就会生成默认`dtype=float`的矩阵，需要自己指定类型

## 2. 获得数组属性

```Python
import numpy as np
"""
[[ 1.  2.  3.  4.]
 [ 5.  6.  7.  8.]
 [ 9. 10. 11. 12.]]
"""
arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=float)
print(arr)

# 维度 2
print(arr.ndim)

# 形状 (3, 4), 表示3行4列, 每个参数代表每一维的大小
print(arr.shape)

# 总数 行 × 列 值 12
print(arr.size)

# 数据类型 float64
print(arr.dtype)
```

## 3. 数组切片/索引

### 一维数组索引

和普通python的列表索引切片没区别

```Python
import numpy as np

"""
[1 2 3 4 5]
"""
arr = np.array([1, 2, 3, 4, 5])

print(arr[0])
"""
1
"""

print(arr[-1])
"""
5
"""

print(arr[1:4])
"""
[2 3 4]
"""
```

### 二维/多维数组索引

```Python
import numpy as np

"""
[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""
arr = np.array([[1,2,3], [4,5,6], [7,8,9]])

# 获取某一个元素
print(arr[2,2])
"""
9
"""

# 获取某一行
print(arr[1])
"""
[4 5 6]
"""

# 获取某一列
print(arr[:,1])
"""
[2 5 8]
"""

# 获取某一个子矩阵，同样可以用来取某几个数据
print(arr[0:2,0:2])
"""
[[1 2]
 [4 5]]
"""
```

## 4. 数组计算

### 逐元素计算

```Python
import numpy as np

arr_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
"""
[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""
arr_2 = np.full((3, 3), 2, dtype=int)
"""
[[2 2 2]
 [2 2 2]
 [2 2 2]]
"""

# 加法
add_result = arr_1 + arr_2
print(add_result)
"""
[[ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
"""

# 减法
sub_result = arr_1 - arr_2
print(sub_result)
"""
[[ -1   0   1]
 [  2   3   4]
 [  5   6   7]]
"""

# 乘法
mul_result = arr_1 * arr_2
print(mul_result)
"""
[[ 2  4  6]
 [ 8 10 12]
 [14 16 18]]
"""

# 除法
div_result = arr_1 / arr_2
print(div_result)
"""
[[0.5 1.  1.5]
 [2.  2.5 3. ]
 [3.5 4.  4.5]]
"""

# 单矩阵和参数计算
new_arr_1 = arr_1 + 1
print(new_arr_1)
"""
[[ 2  3  4]
 [ 5  6  7]
 [ 8  9 10]]
"""
new_arr_2 = arr_1 * 3
print(new_arr_2)
"""
[[ 3  6  9]
 [12 15 18]
 [21 24 27]]
"""
```

### 矩阵乘法

$$
C_{ij} = A_{i1} B_{1j} + A_{i2} B_{2j} + A_{i3} B_{3j}
$$

```Python
import numpy as np

arr_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
"""
[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""

arr_2 = np.full((3, 2), 2, dtype=int)
"""
[[2 2]
 [2 2]
 [2 2]]
"""

# 矩阵相乘，前行乘后列
mul_result2 = np.dot(arr_1 , arr_2)
print(mul_result2)
"""
[[12 12]
 [30 30]
 [48 48]]
"""
```

## 5. 统计函数

```Python
import numpy as np

arr = np.array([[1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1],
                [6, 7, 8, 9, 10],
                [10, 9, 8, 7, 6]])
"""
[[ 1  2  3  4  5]
 [ 5  4  3  2  1]
 [ 6  7  8  9 10]
 [10  9  8  7  6]]
"""

sum_arr = np.sum(arr)
"""
110 对矩阵所有元素求和
"""

mean_arr = np.mean(arr)
"""
5.5 对矩阵所有元素求平均值
"""
mean_arr2 = np.mean(arr[2, :])
"""
8.0 对矩阵第三行元素求平均值
"""

max_arr = np.max(arr)
"""
10 矩阵中最大值
"""

min_arr = np.min(arr)
"""
1 矩阵中最小值
"""

std_arr = np.std(arr)
"""
2.8722813232690143 矩阵中所有元素的标准差
"""

var_arr = np.var(arr)
print(var_arr)
"""
8.25 矩阵中所有元素的方差
"""
```

$$
标准差公式: \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2}
$$

$$
方差公式: \sigma^2 = \frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2
$$

## 6. 数组变形拼接

```Python
import numpy as np

# 拆分， 同样可以用来行向量转为列向量(1, -1) -> (-1, 1)
arr_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
reshape_arr = arr_1.reshape(2, 5)
"""
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]]
"""

# 拼接
arr_2 = np.array([[1, 2, 3], [4, 5, 6]])
arr_3 = np.array([[7, 8, 9], [10, 11, 12]])
concat_arr = np.concatenate((arr_2, arr_3), axis=1)
"""
axis=0 按行拼接(上下), 等价于 np.vstack((arr_2, arr_3))
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
 
axis=1 按列拼接(左右),等价于 np.hstack((arr_2, arr_3))
[[ 1  2  3  7  8  9]
 [ 4  5  6 10 11 12]]
"""
```

## 7. 过滤元素

```Python
import numpy as np

arr = np.array([1,2,3,4,5,6,7,8])
arr1 = arr > 3 # 生成布尔数组
"""
[False False False  True  True  True  True  True]
"""
arr2 = arr[arr1] # 过滤小于等于3的元素
"""
[4 5 6 7 8] 
"""
```

## 8. 矩阵转置&逆矩阵

$$
转置公式: A = [a_{ij}] \;\Rightarrow\; A^{T} = [a_{ji}]
$$

检查矩阵是否可逆
$$
A =
\begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{bmatrix}
$$

$$
求行列式: \det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)
$$

求逆矩阵的常规计算方法

我们知道逆矩阵和原矩阵的关系是
$$
A A^{-1} = A^{-1} A = I
$$
据此有基础的计算2x2矩阵和3x3矩阵计算过程

2x2矩阵：
$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

$$
A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}, \quad \text{前提：} ad-bc \neq 0
$$

3x3矩阵：

- step1 计算行列式

$$
A = \begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{bmatrix}
$$

$$
\det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)
$$

- step2 求代数余子式矩阵

$$
C_{11} = +(ei - fh), \quad
C_{12} = -(di - fg), \quad
C_{13} = +(dh - eg)...
$$

$$
C =
\begin{bmatrix}
C_{11} & C_{12} & C_{13} \\
C_{21} & C_{22} & C_{23} \\
C_{31} & C_{32} & C_{33}
\end{bmatrix}
$$

- step3 转置代数余子式矩阵➡️伴随矩阵:

$$
\operatorname{adj}(A) = C^T
$$

- step4 除以行列式得到逆矩阵

$$
A^{-1} = \frac{1}{\det(A)} \operatorname{adj}(A)
$$

```python
import numpy as np

arr = np.array([[1,2,3],
                [4,5,6],
                [7,21,10]])

arr_t = arr.T

# 检查矩阵是否可逆
check = np.linalg.det(arr)
if check != 0:
    arr_inv = np.linalg.inv(arr)
    print(arr_inv)
    """
    [[-1.01333333  0.57333333 -0.04      ]
     [ 0.02666667 -0.14666667  0.08      ]
     [ 0.65333333 -0.09333333 -0.04      ]]
    """

    # 验证逆矩阵和矩阵相乘是否等于单位矩阵
    result = np.dot(arr_inv, arr)
    print(result)
    """
    [[ 1.00000000e+00  2.15105711e-16  6.52256027e-16]
     [-1.52655666e-16  1.00000000e+00 -1.38777878e-16]
     [ 2.08166817e-17  6.24500451e-17  1.00000000e+00]]
    """
```

# Pandas库

用于数据处理和分析的强大库，它提供了高效且灵活的数据结构

`Pandas` 支持多种文件格式的读写操作，如 CSV、Excel、SQL 数据库等，具体操作时再去搜索即可，这里不展开。

## Series数组 - 一维带标签数组

```Python
import pandas as pd
# 从列表创建Series数组 不指定标签 标签默认为角标
s1 = pd.Series([1,True,3.14,"Hello"])
print(s1)

# 指定标签
s2 = pd.Series([1,True,3.14,"Hello"], index=['a','b','c','d'])
print(s2)

# 从字典创建Series数组
dic = {'a':10, 'b':20, 'c':30, 'd':40}
s3 = pd.Series(dic)
print(s3)

"""
0        1
1     True
2     3.14
3    Hello
dtype: object
a        1
b     True
c     3.14
d    Hello
dtype: object
a    10
b    20
c    30
d    40
dtype: int64
"""
```

## DataFrame - 二维表格型数据结构

```Python
import pandas as pd

# 从字典创建DataFrame
dic = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [24, 27, 22, 32],
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}
df1 = pd.DataFrame(dic)
print(df1)
"""
      name  age         city
1    Alice   24     New York
2      Bob   27  Los Angeles
3  Charlie   22      Chicago
4    David   32      Houston
"""

# 从列表创建DataFrame
lst = [['Alice', 24, 'New York'],
       ['Bob', 27, 'Los Angeles'],
       ['Charlie', 22, 'Chicago'],
       ['David', 32, 'Houston']]
df2 = pd.DataFrame(lst, columns=['name', 'age', 'city'])
print(df2)
"""
      name  age         city
0    Alice   24     New York
1      Bob   27  Los Angeles
2  Charlie   22      Chicago
3    David   32      Houston
"""
```

同样也可以加`index`来指定标签，默认是从0开始的数字标签

## 数据合并与连接

`pd.merge()` 可以根据一个或多个键将不同 DataFrame 的行连接起来

- 内连接（默认）：仅返回两个 DataFrame 中键匹配的行。
- 左连接：返回左 DataFrame 的所有行，以及右 DataFrame 中匹配的行，若右表无匹配则用 **NaN** 填充。
- 右连接：返回右 DataFrame 的所有行，以及左 DataFrame 中匹配的行，若左表无匹配则用 **NaN** 填充。
- 外连接：返回两个 DataFrame 中所有行，无匹配的位置用 **NaN** 填充。

```Python
import pandas as pd

df1 = pd.DataFrame(
    {
        'key':['A','B','C','D'],
        'value':[1,2,3,4]
    }
)
df2 = pd.DataFrame({
    'key':['B', 'D', 'E', 'F'],
    'value':[4,6,7,8]
})

# 内链接 按指定的列来取交集, 默认方式
print(pd.merge(df1, df2, on='key', how='inner'))
"""
  key  value_x  value_y
0   B        2        4
1   D        4        6
"""
# 左链接 左边与交集的并集
print(pd.merge(df1, df2, on='key', how="left"))
"""
  key  value_x  value_y
0   A        1      NaN
1   B        2      4.0
2   C        3      NaN
3   D        4      6.0
"""
# 右链接 右边与交集的并集
print(pd.merge(df1, df2, on="key", how='right'))
"""
  key  value_x  value_y
0   B      2.0        4
1   D      4.0        6
2   E      NaN        7
3   F      NaN        8

"""
# 外链接 并集
print(pd.merge(df1, df2, on="key", how='outer'))
"""
  key  value_x  value_y
0   A      1.0      NaN
1   B      2.0      4.0
2   C      3.0      NaN
3   D      4.0      6.0
4   E      NaN      7.0
5   F      NaN      8.0
"""
```

