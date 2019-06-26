all: ftrl

ftrl: FTRL.cpp sparse_vector.cpp corpus.cpp
	g++ -std=c++11 FTRL.cpp sparse_vector.cpp corpus.cpp -o ftrl -fopenmp

# ftrl: ./include/FTRL.cpp ./include/sparse_vector.cpp ./include/corpus.cpp
# 	g++ -std=c++11 ./include/FTRL.cpp ./include/sparse_vector.cpp ./include/corpus.cpp -o ftrl -fopenmp