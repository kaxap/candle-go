package main

import (
	"fmt"
	"sort"
	"sync"
	"testing"
)

func TestThreadSafe(test *testing.T) {
	strs := []string{"hello", "world", "rust", "go"}
	e := CreateEmbeddings(strs)
	const n = 100
	ch := make(chan [][]float32, n)
	wg := sync.WaitGroup{}
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(g int) {
			defer wg.Done()
			ch <- CreateEmbeddings(strs)
			fmt.Printf("%d is done\n", g)
		}(i)
	}
	fmt.Println("waiting")
	wg.Wait()
	close(ch)
	fmt.Println("done. Comparing")
	for t := range ch {
		for i := range t {
			for j := range t[i] {
				if e[i][j] != t[i][j] {
					test.Error("not equal")
					test.Fail()
				}
			}
		}
	}
}

func l2dist(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += (a[i] - b[i]) * (a[i] - b[i])
	}
	return sum
}

func TestEmbeddings(test *testing.T) {
	strs := []string{
		"query: сколько протеина должна есть женщина",
		"query: сколько белка в твороге",
		"query: how much protein should a female eat",
		"query: сколько белка в egg",
		"query: сколько белка нужно в день при тренировках",
		"query: сколько белка в яйце",
		"query: how much is the time?",
		"query: how much protein a male should eat",
		"query: 16th century philosophers",
		"query: 18th century philosophers",
		"query: Critique of Pure Reason",
		"query: what the reason for being not nice",
		"query: the cause of bad behaviour",
		"query: the works of Francis Bacon",
		"query: the books of Immanuel Kant",
		"query: feline anatomy",
		"query: protein shakes and other stuff",
		"query: cat's body",
		"query: burnouts and how to deal with them",
		"query: overworking leads to depression",
	}
	e := CreateEmbeddings(strs)

	type Item struct {
		index int
		score float32
	}

	distances := make(map[int][]Item)
	for i := range e {
		for j := range e {
			if i == j {
				continue
			}
			distances[i] = append(distances[i], Item{j, l2dist(e[i], e[j])})
		}
	}
	for k := range distances {
		sort.Slice(distances[k], func(i, j int) bool {
			return distances[k][i].score < distances[k][j].score
		})
	}
	for i := range distances {
		fmt.Printf("top3 for '%s':\n", strs[i])
		for j := range distances[i][:3] {
			fmt.Printf("dist: %f\t'%s'\n", distances[i][j].score, strs[distances[i][j].index])
		}
	}
}
