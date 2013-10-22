# Makefile for the Runge-Kutta Package
# Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
# Time-stamp: <2013-10-22 00:28:30 (jonah)>

# The default compiler is g++
CXX = g++

# Flags for the compiler. Ask for warnings. Enable the debugger.
CXXFLAGS = -Wall -g

default: rkf45_test_driver
all: rkf45_test_driver

lib: rkf45.o

rkf45_test_driver: rkf45_test_driver.bin
rkf45_test_driver.bin: rkf45.hpp rkf45.o rkf45_test_driver.o
	$(CXX) $(CXXFLAGS) -o $@ $^

rkf45_test_driver.o: rkf45.hpp rkf45.o
rkf45.o: rkf45.hpp

.PHONY: default all rkf45_test_driver

clean:
	$(RM) *.bin *.o