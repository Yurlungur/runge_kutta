// rkf45.cpp

// Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
// Time-stamp: <2013-10-14 14:43:22 (jonah)>

// This is my implementation of the 4-5 Runge-Kutta-Feldberg adaptive
// step size integrator. For simplicity, and so that I can bundle
// public and private methods together, I take an object-oriented
// approach.


// Background
// ----------------------------------------------------------------------

// Assume an ODE system y' = f(t,y) for some y(t). Assume it is an
// initial value problem with y(0) = y0. (y can be a vector).

// The classical Runge-Kutta methods simulate higher-order terms in a
// taylor series expansion of the function f to generate a high-order
// approximation for y' and thus iteratively solves for y(t).

// We get the "simulated" higher-order terms in the expansion by
// evaluating the function f multiple times during a single
// time-step. RK4 evaluates the function 4 times. RK5 evaluates it 5
// times. etc.

// The Runge-Kutta-Feldberg method runs both an RK4 and an RK5
// algorithm together. The RK4 step is the one that will actually be
// output for the next time step. However, RK5-RK4 gives the estimated
// trunctation error, which can help determine the step size.

// I have taken the algorithm details from the articles on Runge-Kutta
// methods and the Runge-Kutta-Feldberg method on wikipedia:
// http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
// http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method

// ----------------------------------------------------------------------



// Usage
// ----------------------------------------------------------------------

// The RKF45 integrator expects a function of the form
// double f(double t, const dVector& y),
// where y is the array representing the n-dimensional ODE system at
// time t. The length of the double vector is assumed to be
// appropriate.

// Optionally, you can have a function f of the form
// double f(double t, dVector& y, const dVector& optional_args),
// where the optional arguments modify f and thus f(t,y).

// This means that if your system is not in this form you'll need to
// write it in this form before you can use RKF45.

// To start the algorithm you need an initial step size, so this must
// be input by hand. If you like, you can also impose a maximum step size.

// The error tolerance is the sum of two error terms, a relative error
// term and an absolute error term,

// error_tolerance = rtoll * l2_norm(y) + atoll,

// where rtoll is the relative error tolerance and atoll is the
// absolute error tolerance. l2_norm(y) is the L2 norm of y.

// By default the relative error tolerance is the square root of
// matchine epsilon and the maximum error tolerance is the square root
// of <maximum double value>

// You can also set the relative error tolerance. The relative error
// tolerance is, by default, 0.01% of the absolute value of the
// smallest element of y.

// The step size is chosen as
// dt = (1-safety_margin) * absolute_error_tolerance / estimated_error,
// Where the estimated error is chosen using the adaptive step size
// method. Safety keeps the step size a little smaller than the
// maximum allowed error which might be unstable. By default,
// safety_margin is 0.1, but you can change it.

// You can also set the maximum step size. By default, it is the
// square root of the largest double value.

// You can choose how RKF45 outptus the solution after a given
// time. You can have it set fill a pre-allocated double array
// or you can have it output a vector.

// ----------------------------------------------------------------------


// Includes
#include <vector>   // for output and internal variables
#include <iostream> // For printing a given ode system
#include <iomanip> // For manipulating the output.
#include <float.h> // For machine precision information
#include <cmath> // for math
#include <assert.h> // For error checking
#include "rkf45.hpp" // the header for this program
// Namespace specification. For convenience.
using std::cout;
using std::endl;
using std::vector;
using std::ostream;
using std::istream;

// Private methods
// ----------------------------------------------------------------------

// A convenience function. Wraps the function f = y'. Depending on
// whether or not f takes optional arguments does the correct thing.
double RKF45::f(int t, const dVector& y) const {
  if ( use_optional_arguments) {
    return f_with_optional_args(t,y,optional_args);
  }
  else {
    return f_no_optional_args(t,y);
  }
}

// Adds two dVectors a and b and outputs a new dVector that is their
// sum. Assumes that the vectors are the same size. If they're not,
// raises an error.
dVector RKF45::sum(const dVector& a, const dVector& b) const {
  assert ( a.size() == b.size()
	   && "To sum two vectors, they must be the same size.");
  dVector output;
  int size = a.size();
  output.resize(size);
  for (int i = 0; i < size; i++) {
    output[i] = a[i] + b[i];
  }
  return output;
}

// Takes the scalar product of a dVector v with a scalar k
dVector RKF45::scalar_product(double k, const dVector& v) const {
  dVector output;
  for (dVector::const_iterator it=v.begin(); it != v.end(); ++it) {
    output.push_back(k * (*it));
  }
  return output;
}

// Calculuates the sum of all elements in a dVector.
double RKF45::sum(const dVector& v) const {
  double output = 0;
  for (dVector::const_iterator it=v.begin(); it != v.end(); ++it) {
    output += (*it);
  }
  return output;
}

// Calculates the 2-norm of the dVector v
double RKF45::norm(const dVector& v) const {
  double output = sum(v);
  output /= v.size();
  return output;
}

// Finds the relative error tolerance based on the current state of
// the system.
double RKF45::get_relative_error_tolerance() const {
  if (ys.size() > 0) {
    return relative_error_factor * norm(ys.back());
  }
  else {
    return relative_error_factor * norm(y0);
  }
}
