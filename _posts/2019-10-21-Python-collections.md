---
layout:     post
title:      Python Collections
subtitle:   import collections
date:       2019-10-21
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Python
    - Collections
---

# Collections

a）Counter： 计数器，用于统计元素的数量

b）OrderDict：有序字典

c）defaultdict：值带有默认类型的字典

d）namedtuple：可命名元组，通过名字来访问元组元素

e）deque :双向队列，队列头尾都可以放，也都可以取（与单向队列对比，单向队列只能一头放，另一头取）

## collections.Counter()

>计数器，用于统计对象中每个元素出现的个数

```python
#通过字典形式统计每个元素重复的次数传  
res = collections.Counter('abcdabcaba')  
print(res)                                  #结果Counter({'a': 4, 'b': 3, 'c': 2, 'd': 1})  
  
#dict的子类，所以也可以以字典的形式取得键值对  
for k in res:  
    print(k, res[k], end='  |  ')           #结果 a 4  |  b 3  |  c 2  |  d 1  |  
for k, v in res.items():  
    print(k, v, end='  |  ')                #结果 a 4  |  b 3  |  c 2  |  d 1  |  
  
#通过most_common(n)，返回前n个重复次数最多的键值对  
print(res.most_common())                    #结果None  
print(res.most_common(2))                   #结果[('a', 4), ('b', 3)]  
  
#通过update来增加元素的重复次数，通过subtract来减少元素重复的次数  
a = collections.Counter('abcde')  
res.update(a)  
print(res)                                  #结果Counter({'a': 5, 'b': 4, 'c': 3, 'd': 2, 'e': 1})，比原来的res增加了重复次数  
  
b = collections.Counter('aaafff')  
res.subtract(b)  
print(res)                                  #结果Counter({'b': 4, 'c': 3, 'a': 2, 'd': 2, 'e': 1, 'f': -3})，还有负值，要注意  
  
#fromkeys功能还没实现，使用的话会报错
```


## collections.OrderedDict()

>有序字典，数据结构字典Dict是无序的，有时使用起来不是很方便，Collections里提供一个有序字典OrderDict，用起来就很方便了

dict的方法OrderDict基本都可以使用，比如keys(), values(), clear()

注意，因为OrderDict有序，有些方法不同，比如，pop()和popitem()

另外OrderDict增加了一个move_to_end的方法

```python
#创建一个有序字典
dic = collections.OrderedDict()
dic['name'] = 'winter'
dic['age'] = 18
dic['gender'] = 'male'

print(dic)                         #结果OrderedDict([('name', 'winter'), ('age', 18), ('gender', 'male')])

#将一个键值对放入最后
dic.move_to_end('name')
print(dic)                         #结果OrderedDict([('age', 18), ('gender', 'male'), ('name', 'winter')])
```

## collections.defaultdict(int)

>默认字典，为字典设置一个默认类型

```python
people = [['male', 'winter'], ['female', 'elly'], ['male', 'frank'], ['female', 'emma']]

gender_sort = collections.defaultdict(list)
for info in people:
    gender_sort[info[0]].append(info[1])

print(gender_sort)      #结果defaultdict(<class 'list'>, {'male': ['winter', 'frank'], 'female': ['elly', 'emma']})
```

## collections.namedtuple()

```python
position_module = collections.namedtuple('position', ['x', 'y', 'z'])   #'position'相当于指定一个类型，类似于上面的OrderedDict([('age', 18), ('gender', 'male'), ('name', 'winter')])中的OrderdDict

a_position = position_module(3, 5, 7)
print(a_position)                                   #结果position(x=3, y=5, z=7)
print(a_position.x, a_position.y, a_position.z)     #结果3 5 7
```

## collections.deque()

>deque其实是 double-ended queue 的缩写，双向队列


```pythoon
raw = [1,2,3]
d = collections.deque(raw)
print(d)                    #结果deque([1, 2, 3])

#右增
d.append(4)
print(d)                    #结果deque([1, 2, 3, 4])
#左增
d.appendleft(0)
print(d)                    #结果deque([0, 1, 2, 3, 4])

#左扩展
d.extend([5,6,7])
print(d)                    #结果deque([0, 1, 2, 3, 4, 5, 6, 7])
#右扩展
d.extendleft([-3,-2,-1])
print(d)                    #结果deque([-1, -2, -3, 0, 1, 2, 3, 4, 5, 6, 7])

#右弹出
r_pop = d.pop()
print(r_pop)                #结果7
print(d)                    #结果deque([-1, -2, -3, 0, 1, 2, 3, 4, 5, 6])
#左弹出
l_pop = d.popleft()
print(l_pop)                #结果-1
print(d)                    #结果deque([-2, -3, 0, 1, 2, 3, 4, 5, 6])

#将右边n个元素值取出加入到左边
print(d)                    #原队列deque([-2, -3, 0, 1, 2, 3, 4, 5, 6])
d.rotate(3)
print(d)                    #rotate以后为deque([4, 5, 6, -2, -3, 0, 1, 2, 3])
```

