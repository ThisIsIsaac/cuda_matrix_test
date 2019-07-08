test : test.cpp matrix.hpp half.hpp
	nvcc -std=c++14 test.cpp -o test
	./test
