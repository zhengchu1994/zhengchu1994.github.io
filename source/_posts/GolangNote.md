---
title: GolangNote
mathjax: true
date: 2020-05-08 16:00:00
tags: Go
categories: Go
visible:

---



## 基本语法

### `:=`

> Inside a function, the `:=` short assignment statement can be used in place of a `var` declaration with implicit type.
>
> Outside a function, every statement begins with a keyword (`var`, `func`, and so on) and so the `:=` construct is not available.

### 切片 = 数组引用

```go
func main() {
	names := [4]string{
		"John",
		"Paul",
		"George",
		"Ringo",
	}
	fmt.Println(names)

	a := names[0:2]
	b := names[1:3]
	fmt.Println(a, b)

	b[0] = "XXX"
	fmt.Println(a, b)
	fmt.Println(names)
}
/*
output:

[John Paul George Ringo]
[John Paul] [Paul George]
[John XXX] [XXX George]
[John XXX George Ringo]
*/
```



### 结构体切片初始化

```go
func main() {
	s := []struct {
		i int
		b bool
	}{
		{2, true},
		{3, false},
		{5, true},
		{7, true},
		{11, false},
		{13, true},
	}
	fmt.Println(s)
}
```



### 这个赋值之后还是它？

```go
package main

import "fmt"

func main() {
	s := []int{2, 3, 5, 7, 11, 13}
	printSlice(s)

	s = s[:0]
	printSlice(s)

	s = s[:4]
	printSlice(s)

	s = s[2:]
	printSlice(s)
}

func printSlice(s []int) {
	fmt.Printf("len=%d cap=%d %v\n", len(s), cap(s), s)
}

/*
len=6 cap=6 [2 3 5 7 11 13]
len=0 cap=6 []
len=4 cap=6 [2 3 5 7]
len=2 cap=4 [5 7]
*/
```





### nil

切片的零值是 `nil`。



### make创造切片



`make` 函数会分配一个元素为零值的数组并返回一个引用了它的切片：

```go
b := make([]int, 0, 5) // len(b)=0, cap(b)=5

b = b[:cap(b)] // len(b)=5, cap(b)=5
b = b[1:]      // len(b)=4, cap(b)=4
```



### append内建函数



向元素类型为`T`的切片`s`中追加同类型元素，返回该类型的切片。

```go
func append(s []T, vs ...T) []T
```



### range

`for` 循环的 `range` 形式可遍历切片或映射。

当使用 `for` 循环遍历切片时，每次迭代都会返回两个值。第一个值为当前元素的下标，第二个值为该下标所对应元素的一份副本。

```go
var pow = []int {1, 2}
func main(){
	for i, v := range pow{
		fmt.Printf("...")
	}
}
```



