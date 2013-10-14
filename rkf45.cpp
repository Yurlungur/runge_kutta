// rkf45.cpp

// Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
// Time-stamp: <2013-10-14 18:00:37 (jonah)>

// This is my implementation of the 4-5 Runge-Kutta-Feldberg adaptive
// step size integrator. For simplicity, and so that I can bundle
// public and private methods together, I take an object-oriented
// approach.

// For background and usage, see the header file. 
// ----------------------------------------------------------------------


// Includes
#include <vector>   // for output and internal variables
#include <iostream> // For printing a given ode system
#include <iomanip> // For manipulating the output.
#include <float.h> // For machine precision information
#include <cmath> // for math
#include <cassert> // For error checking
#include "rkf45.hpp" // the header for this program
// Namespace specification. For convenience.
using std::cout;
using std::endl;
using std::vector;
using std::ostream;
using std::istream;


// Getters
// ----------------------------------------------------------------------
// Returns the total number of steps the integrated has iterated through
int RKF45::steps() const {
  return ys.size();
}

// Finds the error tolerance based on the current state of the system.
double RKF45::get_error_tolerance() const {
  return get_relative_error_tolerance() + get_absolute_error();
}

// Returns the size of the system. Assumes the system stays the same
// size. If initial data has not yet been set, will return zero.
int RKF45::size() const {
  if ( ys.size() > 0 ) {
    return ys.front().size();
  }
  else {
    return 0;
  }
}

// Returns the initial data. Returns an empty vector if no initial
// data has been set yet. Does NOT pass by reference. If no initial
// data has been set, returns an empty vector.
dVector RKF45::get_y0() const {
  dVector output;
  if ( size() > 0 ) {
    output = y0;
  }
  return output;
}

// Returns the start time.
double RKF45::get_t0() const {
  return t0;
}

// Returns true if the system is set to use optional arguments,
// false otherwise.
bool RKF45::using_optional_args() const {
  return use_optional_arguments;
}

// Returns the optional arguments if they are in use. If they are
// not in use or if they haven't been set yet, returns an empty
// array otherwise.
dVector RKF45::get_optional_args() const {
  dVector output;
  if ( using_optional_args() ) {
    output = optional_args;
  }
  return output;
}

// Returns the maximum step size allowed.
double RKF45::get_max_dt() const {
  return max_dt;
}

// Returns the initial step size.
double RKF45::get_dt0() const {
  return dt0;
}

// Returns the previous step size. If no steps have been performed,
// returns -1.
double RKF45::get_last_dt() const {
  if ( steps() > 0 ) {
    return last_dt;
  }
  else {
    return -1;
  }
}

// Returns the next step size.
double RKF45::get_next_dt() const {
  return next_dt;
}

// Returns the current absolute error
double RKF45::get_absolute_error() const {
  return absolute_error;
}

// Returns the current relative error factor
double RKF45::get_relative_error_factor() const {
  return relative_error_factor;
}

// Returns the safety margin for step-size choice
double RKF45::get_safety_margin() const {
  return safety_margin;
}
// ----------------------------------------------------------------------


// Setters
// ----------------------------------------------------------------------
// Sets the function y'=f. One version takes optional arguments. One
// does not. The second vector is optional arguments.
void RKF45::set_f(double (*f)(double,const dVector&)) {
  use_optional_arguments = false;
  f_no_optional_args = f;
}
void RKF45::set_f(double (*f)(double,const dVector&,const dVector&)) {
  use_optional_arguments = true;
  f_with_optional_args = f;
}

// Sets the initial y vector.
void RKF45::set_y0(const dVector& y0) {
  this->y0 = y0;
}

// Sets the start time. Passing in no arguments resets to the
// default.
void RKF45::set_t0(double t0) {
  this->t0 = t0;
}
void RKF45::set_t0() {
  set_t0(DEFAULT_T0);
}

// Sets the optional arguments. If the function f does not take
// optional arguments, raises an error.
void RKF45::set_optional_args(const dVector& optional_args) {
  assert ( using_optional_args()
	   && "Set the function to take optional arguments first." );
  this->optional_args = optional_args;
}

// Sets the maximum allowed step size. Passing in no arguments
// resets to the default.
void RKF45::set_max_dt(double max_dt) {
  assert ( max_dt > 0 && "Steps must be positive." );
  this->max_dt = max_dt;
}
void RKF45::set_max_dt() {
  set_max_dt(default_max_dt());
}

// Sets the initial step size. Passing in no arguments resets to the
// default.
void RKF45::set_dt0(double dt0) {
  assert ( dt0 > 0 && "Steps must be positive." );
  this->dt0 = dt0;
}

// Sets the next step size. Use with caution! If you set the next
// step size, the value chosen automatically is forgotten!
void RKF45::set_next_dt(double next_dt) {
  assert (next_dt > 0 && "Steps must be positive." );
  this->next_dt = next_dt;
}

// Sets the absolute error. Passing in no arguments resets to the
// default.
void RKF45::set_absolute_error(double absolute_error) {
  assert ( absolute_error > 0 && "Error must be positive." );
  this->absolute_error = absolute_error;
}
void RKF45::set_absolute_error() {
  set_absolute_error(default_absolute_error());
}

// Sets the relative error factor. Passing in no arguments resets to
// the default.
void RKF45::set_relative_error_factor(double relative_error_factor) {
  assert ( relative_error_factor > 0 && "Error must be positive." );
  this->relative_error_factor = relative_error_factor;
}
void RKF45::set_relative_error_factor() {
  set_relative_error_factor(DEFAULT_RELATIVE_ERROR_FACTOR);
}

// Sets the safety margin for step size choices. Passing in no
// arguments resets to the default.
void RKF45::set_safety_margin(double safety_margin) {
  assert ( safety_margin > 0 && "Error must be positive." );
  this->safety_margin = safety_margin;
}
void RKF45::set_safety_margin() {
  set_safety_margin(DEFAULT_SAFETY_MARGIN);
}
// ----------------------------------------------------------------------


// Private methods
// ----------------------------------------------------------------------

// A convenience function. Sets all the fields to their default values.
void RKF45::set_defaults() {
  t0 = DEFAULT_T0;
  use_optional_arguments = DEFAULT_USE_OPTIONAL_ARGS;
  max_dt = default_max_dt();
  dt0 = DEFAULT_DT0;
  next_dt = dt0;
  absolute_error = default_absolute_error();
  relative_error_factor = DEFAULT_RELATIVE_ERROR_FACTOR;
  safety_margin = DEFAULT_SAFETY_MARGIN;
}

// A convenience function. Wraps the function f = y'. Depending on
// whether or not f takes optional arguments does the correct thing.
double RKF45::f(int t, const dVector& y) const {
  if ( using_optional_args() ) {
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
  if ( ys.size() > 0 ) {
    return get_relative_error_factor() * norm(ys.back());
  }
  else {
    return get_relative_error_factor() * norm(y0);
  }
}



