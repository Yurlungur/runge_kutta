// rkf45.cpp

// Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
// Time-stamp: <2013-10-17 00:42:17 (jonah)>

// This is my implementation of the 4-5 Runge-Kutta-Fehlberg adaptive
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
#include <iomanip> // for controlling io
#include "rkf45.hpp" // the header for this program
// Namespace specification. For convenience.
using std::cout;
using std::endl;
using std::vector;
using std::ostream;
using std::istream;
using std::right;
using std::abs;

// Constructors
// ----------------------------------------------------------------------
// Creates an empty integrator, to be initialized later.
RKF45::RKF45() {
  set_defaults();
}

// Creates an integrator for the ode system with y'=f(t,y).  The other
// properties are assumed to be set by setter methods. The size of
// the vector is assumed to be appropriate.
RKF45::RKF45(dVector (*y)(double,const dVector&)) {
  set_defaults();
  set_f(y);
}

// Creates an integrator for the ode system with y'=f(t,y), where
// f(t,y) = f(t,y,optional_args). The size of the vectors is assumed
// to be appropriate.
RKF45::RKF45(dVector (*y)(double,const dVector&,const dVector&)) {
  set_defaults();
  set_f(y);
}

// Creates an integrator for an ode system with y'=f(t,y), with
// initial time t0 and initial conditions y0. The size of the system
// is inferred from the size of the initial conditions vector.
RKF45::RKF45(dVector (*y)(double, const dVector&), double t0,
	     const dVector& y0) {
  set_defaults();
  set_f(y);
  set_t0(t0);
  set_y0(y0);
  set_dt0();
}

// Creates an integrator for an ode system with
// y'=f(t,y)=f(t,y,optional_args), with initial time t0 and initial
// conditions y0. The size of the system is inferred from the size
// of the initial conditions vector. The optional_args are passed in
// now too. The size is inferred from the input vector.
RKF45::RKF45(dVector (*y)(double,const dVector&,const dVector&),
	     double t0, const dVector& y0,
	     const dVector& optional_args) {
  set_defaults();
  set_f(y);
  set_t0(t0);
  set_y0(y0);
  set_optional_args(optional_args);
  set_dt0();
}

// Creates an integrator of an ode system with y'=f(t,y), initial
// time t0, initial conditions y0, initial step size delta_t0,
// relative error tolerance relative_tolerance, and absolute error
// tolerance absolute_tolerance. The dimension of the system is
// inferred from the size of y0.
RKF45::RKF45(dVector (*y)(double,const dVector&),
	     double t0, const dVector& y0,
	     double delta_t0, double relative_tolerance,
	     double absolute_tolerance) {
  set_defaults();
  set_f(y);
  set_t0(t0);
  set_y0(y0);
  set_dt0(delta_t0);
  set_absolute_error(absolute_tolerance);
  set_relative_error_factor(relative_tolerance);
}

// Creates an integrator of an ode system with y'=f(t,y), initial
// time t0 = 0, initial conditions y0, initial step size delta_t0,
// relative error tolerance relative_tolerance, and absolute error
// tolerance absolute_tolerance. The size of the system is inferred
// from the initial data.
RKF45::RKF45(dVector (*y)(double,const dVector&),
	     const dVector& y0, double delta_t0,
	     double relative_tolerance, double absolute_tolerance) {
  set_defaults();
  set_f(y);
  set_y0(y0);
  set_dt0(delta_t0);
  set_absolute_error(absolute_tolerance);
  set_relative_error_factor(relative_tolerance);
}

// Creates an integrator of an ode system with y'=f(t,y), initial
// time t0, initial conditions y0, initial step size delta_t0,
// relative error tolerance relative_tolerance, and absolute error
// tolerance absolute_tolerance.  f(t,y) =
// f(t,y,optional_args). There are m-optional arguments. The
// dimension of the system and the number of optional arguments are
// inferred from size of the vectors.
RKF45::RKF45(dVector (*y)(double, const dVector&,const dVector&),
	     double t0,
	     const dVector& y0, const dVector& optional_args,
	     double delta_t0,
	     double relative_tolerance,double absolute_tolerance) {
  set_defaults();
  set_f(y);
  set_t0(t0);
  set_y0(y0);
  set_dt0(delta_t0);
  set_absolute_error(absolute_tolerance);
  set_relative_error_factor(relative_tolerance);
  set_optional_args(optional_args);
}

// Creates an integrator of an ode system with y'=f(t,y), initial
// time t0=0, initial conditions y0, initial step size delta_t0,
// relative error tolerance relative_tolerance, and absolute error
// tolerance absolute_tolerance.  f(t,y) = f(t,y,optional_args).
RKF45::RKF45(dVector (*y)(double,const dVector&,const dVector&),
	     const dVector& y0, const dVector& optional_args,
	     double delta_t0,
	     double relative_tolerance, double absolute_tolerance) {
  set_defaults();
  set_f(y);
  set_y0(y0);
  set_optional_args(optional_args);
  set_dt0(delta_t0);
  set_relative_error_factor(relative_tolerance);
  set_absolute_error(absolute_tolerance);
}

// Creates an integrator of an ode system with y'=f(t,y), initial
// time t0, initial conditions y0, and initial step size
// delta_t0. The error tolerance is the default error tolerance.
RKF45::RKF45(dVector (*y)(double,const dVector&),
	     double t0, const dVector y0, double delta_t0) {
  set_defaults();
  set_f(y);
  set_t0(t0);
  set_y0(y0);
  set_dt0(delta_t0);
}


// Creates an integrator of the ode system with y'=f(t,y), initial
// time t0, initial conditions y0, initial step size delta_t0.
// f(t,y) = f(t,y,optional_args). The error tolerance is the
// default. f(t,y) = f(t,y,optional_args)
RKF45::RKF45(dVector (*y)(double,const dVector&,const dVector&),
	     double t0, const dVector& y0, const dVector& optional_args,
	     double delta_t0) {
  set_defaults();
  set_f(y);
  set_t0(t0);
  set_y0(y0);
  set_optional_args(optional_args);
  set_dt0(delta_t0);
}

// Copy constructor. Generates a copy of the integrator as it was
// before the first step was made. Includes all initial data. Useful
// for the shooting method.
RKF45::RKF45(const RKF45 &integrator) {
  set_defaults();
  set_f(integrator.get_f());
  set_t0(integrator.get_t0());
  set_y0(integrator.get_y0());
  set_max_dt(integrator.get_max_dt());
  set_dt0(integrator.get_dt0());
  set_absolute_error(integrator.get_absolute_error());
  set_relative_error_factor(integrator.get_relative_error_factor());
  set_safety_margin(integrator.get_safety_margin());

  if ( integrator.using_optional_args() ) {
    set_optional_args(integrator.get_optional_args());
  }
}

// Assignment operator. Copies one object into another like the copy
// constructor. Useful for the shooting method. Only copies initial
// data. Integration data is not copied.
RKF45& RKF45::operator= (const RKF45 &integrator) {
  set_defaults();
  set_f(integrator.get_f());
  set_t0(integrator.get_t0());
  set_y0(integrator.get_y0());
  set_max_dt(integrator.get_max_dt());
  set_dt0(integrator.get_dt0());
  set_absolute_error(integrator.get_absolute_error());
  set_relative_error_factor(integrator.get_relative_error_factor());
  set_safety_margin(integrator.get_safety_margin());
  
  if ( integrator.using_optional_args() ) {
    set_optional_args(integrator.get_optional_args());
  }
  return *this;
}

// Destructor. Empty/
RKF45::~RKF45() { }

// ----------------------------------------------------------------------


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

// Returns a container class containing the function and whether or
// not optional arguments are in use.
Functor RKF45::get_f() const {
  Functor output;
  if ( using_optional_args() ) {
    output.use_optional_arguments = true;
    output.f_with_optional_args = f_with_optional_args;
  }
  else {
    output.use_optional_arguments = false;
    output.f_no_optional_args = f_no_optional_args;
  }
  return output;
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
void RKF45::set_f(dVector (*f)(double,const dVector&)) {
  use_optional_arguments = false;
  f_no_optional_args = f;
}
void RKF45::set_f(dVector (*f)(double,const dVector&,const dVector&)) {
  use_optional_arguments = true;
  f_with_optional_args = f;
}
void RKF45::set_f(Functor f_func) {
  if ( f_func.use_optional_arguments ) {
    set_f(f_func.f_with_optional_args);
  }
  else {
    set_f(f_func.f_no_optional_args);
  }
}

// Sets the initial y vector.
void RKF45::set_y0(const dVector& y0) {
  this->y0 = y0;
  if ( ys.size() == 0 ) {
    ys.push_back(y0);
  }
  else {
    ys[0] = y0;
  }
}

// Sets the start time. Passing in no arguments resets to the
// default.
void RKF45::set_t0(double t0) {
  this->t0 = t0;
  if ( ts.size() == 0 ) {
    ts.push_back(t0);
  }
  else {
    ts[0] = t0;
  }
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
  if ( max_dt < get_next_dt() && debugging ) {
    // Don't do anything about it since its not a fatal error. But
    // warn the user.
    cout << "WARNING: Next step size larger than max step size." << "\n"
	 << "\tMax step size: " << max_dt << "\n"
	 << "\tNext step size: " << get_next_dt() << endl;
  }
  this->max_dt = max_dt;
}
void RKF45::set_max_dt() {
  set_max_dt(default_max_dt());
}

// Sets the minimum allowed step size.
void RKF45::set_min_dt(double min_dt) {
  assert (min_dt > 0 && "Steps must be positive." );
  if (min_dt < get_next_dt() ) {
    // Don't do anything about it since its not a fatal error. But
    // warn the user.
    cout << "WARNING: Next step size smaller than min step size." << endl;
  }
  this->min_dt = min_dt;
}
void RKF45::set_min_dt() {
  set_min_dt(default_min_dt());
}

// Sets the initial step size. Passing in no arguments resets to the
// default. If the input dt0 is too big, sets to max_dt.
void RKF45::set_dt0(double dt0) {
  assert ( dt0 > 0 && "Steps must be positive." );
  if ( dt0 > get_max_dt() ) {
    this->dt0 = get_max_dt();
  }
  else if ( dt0 < get_min_dt() ) {
    this->dt0 = get_min_dt();
  }
  else {
    this->dt0 = dt0;
  }
  // By definition, the first next_dt needs to be dt0.
  if ( ys.size() < 2 ) {
    set_next_dt(this->dt0);
  }
}

// Sets the initial step size automatically. The function and initial
// y data must be set for this to work. f, t0 and y0 must be set
// before you can call this method. Also sets the max dt to the same
// value.
void RKF45::set_dt0() {
  // locals
  double yprime = norm(f(t0,y0));
  double y = norm(y0);
  double length_scale;
  cout << "y = " << y << "\n" << "yprime = " << yprime << endl;
  // y/y' = <change in t> * y/<change in y>. A good approximation of dt0. 
  length_scale = y/yprime;
  cout << "length scale = " << length_scale << endl;
  dt0 = abs(get_relative_error_factor() * length_scale);
  cout << "dt0 = " << dt0 << endl;
  cout << "Max step size: " << get_max_dt() << endl;

  // dt0 must be a finite, positive number. If it's zero or NaN, make
  // it sqrt(machine epsilon).
  if ( !(dt0 > 0) ) {
    if ( debugging ) {
      cout << "WARNING: dt0 non-finite. Setting it to the default minimum dt"
	   << endl;
    }
    set_dt0(default_min_dt());
  }
  else {
    set_dt0(dt0);
  }
}

// Sets the maximum change in dt. No choice sets it to the default.
void RKF45::set_max_delta_dt(double max_delta) {
  assert ( 0 < max_delta 
	   && "max change in dt is current dt multiplied by max_delta_dt. So max_delta_dt must be positive." );
    max_delta_dt = max_delta;
}
void RKF45::set_max_delta_dt() {
  set_max_delta_dt(DEFAULT_MAX_DELTA_DT);
}


// Sets the next step size. Use with caution! If you set the next
// step size, the value chosen automatically is forgotten!
void RKF45::set_next_dt(double next_dt) {
  assert (next_dt > 0 && "Steps must be positive." );
  if ( next_dt > get_max_dt() ) {
    this->next_dt = get_max_dt();
  }
  else if ( next_dt < get_min_dt() ) {
    this->next_dt = get_min_dt();
  }
  else {
    this->next_dt = next_dt;
  }
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


// Data Output
// ----------------------------------------------------------------------
// Returns the current time step.
double RKF45::get_t() const {
  return ts.back();
}

// Returns the time after n steps.
double RKF45::get_t(int n) const {
  assert ( n < (int)ts.size() && "We have not integrated that far yet!" );
  return ts[n];
}

// Returns the y vector after n steps. Passes by value, so very
// slow.
dVector RKF45::get_y(int n) const {
  assert ( n < (int)ts.size() && "We have not integrated that far yet!" );
  return ys[n];
}

// Returns the current state of the y vector. Passes by value, so
// very slow.
dVector RKF45::get_y() const {
  return ys.back();
}

// Fills an input vector with the current state of the y
// vector. 
void RKF45::get_y(dVector& y_data) const {
  y_data = ys.back();
}

// Fills an input vector with the state of the y vector after n
// steps.
void RKF45::get_y(dVector& y_data, int n) const {
    assert ( n < (int)ts.size() && "We have not integrated that far yet!" );
    y_data = ys[n];
}

// Fills an input array with the current state of the y
// vector. Assumes that the array is the appropriate size. If the
// array size is wrong, you will sefgault.
void RKF45::get_y(double y_data[]) const {
  for (unsigned int i = 0; i < ys.back().size(); i++) {
    y_data[i] = ys.back()[i];
  }
}

// Tests whether or not a given set of constraints is
// satisfied. Takes a boolean function of the vector y and tests
// whether y(t) satisfies the constraint function. This is the fast,
// safe solution to the problem of monitoring constriants. t is
// chosen as t after n time steps.
bool RKF45::test_constraints(bool (*constraints)(double,const dVector),
			     int n) const {
  assert ( n < (int)ts.size() && "We have not integrated that far yet!" );
  return constraints(ts[n],ys[n]);
}

// Tests whether or not a given set of constraints is
// satisfied. Takes a boolean function of the vector y and tests
// whether y(t) satisfies the constraint function. t is set to the
// current time. This is the fast, safe solution to the problem of
// monitoring constriants.
bool RKF45::test_constraints(bool (*constraints)(double,const dVector)) const {
  return constraints(ts.back(),ys.back());
}

// Tests by how much the y vector at time t fails to satisfy the
// constraint function passed in. The constraint function should
// return a vector<double> where each element shows how much the y
// vector failed to satisfy the appropriate constraint
// equation. 
dVector RKF45::test_constraint_degree(dVector (*constraints)(double,const dVector)) const {
  return constraints(ts.back(),ys.back());
}


// Tests by how much the y vector at time t fails to satisfy the
// constraint function passed in. The constraint function should
// return a vector<double> where each element shows how much the y
// vector failed to satisfy the appropriate constraint
// equation. Passed by reference.
dVector RKF45::test_constraint_degree(dVector (*constraints)(double,const dVector), int n) const {
  assert ( n < (int)ts.size() && "We have not integrated that far yet!" );
  return constraints(ts[n],ys[n]);
}


// Prints the integration history to an output stream. No stream
// choice means it prints to cout. This is not the just the current
// state of the system. This is everything.
void RKF45::print(ostream& out) const {
  out << "#<time>\t <y vector>" << endl;
  for (unsigned int i = 0; i < ts.size(); i++) {
    out << ts[i] << " ";
    for (unsigned int j = 0; j < ys[i].size() - 1; j++) {
      out << ys[i][j] << " ";
    }
    out << ys[i].back() << endl;
  }
}
void RKF45::print() const {
  print(cout);
}

// Prints the current state of the system to ostream out. For
// debugging really.
void RKF45::print_state(ostream& out) const {
  out << "#<time>\t <y vector>" << endl;
  for (unsigned int i = 0; i < ys.back().size() - 1; i++) {
    out << ys.back()[i] << " ";
  }
  out << ys.back().back() << endl;
}
void RKF45::print_state() const {
  print_state(cout);
}


// Overload the stream input operator. Works like print(out).
ostream& operator <<(ostream& out, const RKF45& in) {
  in.print(out);
  return out;
}
// ----------------------------------------------------------------------


// integration control
// ----------------------------------------------------------------------
// Integrates by a single time step. The time step is either chosen
// internally from the last integration step or set by set_dt0 or
// set_next_dt.
void RKF45::step() {
  // If this is the first step and if dt0 is still the special
  // defualt, choose a dt0 automatically.
  if ( steps() < 2 && get_dt0() == DEFAULT_DT0 ) {
    set_dt0();
  }

  // Some local variables for convenience
  double t = ts.back();
  double h = get_next_dt();
  dVector y_of_t = ys.back();
  dVector y;

  // The new orders for the dynamic step size
  dVector y_new_order4 = y_of_t;
  dVector y_new_order5 = y_of_t;
  double new_step_size;
  
  // This is the doozy. First we need to compute the five k
  // factors. The indexing scheme is off by 1 from the
  // standard. Subtract 1 for the index if you're reading off of
  // wikipedia.
  dVector k[ORDER];
  for (int i = 0; i < ORDER; i++) {
    y = y_of_t;
    for (int j = 0; j < i; j++) {
      y = sum(y,scalar_multiplication(BUTCHER_A[i][j],k[j]));
    }
    k[i] = scalar_multiplication(h,f(t+BUTCHER_C[i]*h,y));
  }

  // Now we can safely compute the new y values.
  for (int i = 0; i < ORDER; i++) {
    y_new_order4 = sum(y_new_order4,scalar_multiplication(BUTCHER_B[1][i],k[i]));
    y_new_order5 = sum(y_new_order5,scalar_multiplication(BUTCHER_B[0][i],k[i]));
  }

  if ( debugging ) {
    cout << "Safety margin: " << get_safety_margin() << endl;
    cout << "Current step size: " << h << endl;
    cout << "Error tolerance: " << get_error_tolerance() << endl;
    cout << "Difference: "
	 << abs(norm(difference(y_new_order5,y_new_order4))) << endl;
  }

  // Then we can compute the next step size
  new_step_size = abs( ( get_safety_margin() * h * get_error_tolerance() ) / norm(difference(y_new_order5, y_new_order4)) );

  // The change in the step size must be acceptable. The step size is
  // allowed to decrease as quickly as it likes, but it must not
  // INCREASE too quickly.
  if ( new_step_size  > h*(get_max_delta_dt_factor() + 1) ) {
    new_step_size = h + get_max_delta_dt_factor() * h;
  }

  // Finally, set the new step size and report it.
  set_next_dt(new_step_size);
  if ( debugging ) {
    cout << "New step size: " << get_next_dt() << endl;
  }
  // set y and t, and end.
  last_dt = h;
  ts.push_back(t+h);
  ys.push_back(y_new_order4);

  if (debugging ) {
    print_state();
  }
}

// Integrates up to to time t_final. If the current time is after
// t_final, does nothing. The initial step size is set by set_dt0 or
// by set_next_dt.
void RKF45::integrate(double t_final) {
  while ( ts.back() < t_final ) {
    step();
  }
}

// Resets the integrator to t0. This is useful for approaches like
// the shooting method.
void RKF45::reset() {
  ys.resize(0);
  ys.push_back(y0);
  ts.resize(0);
  ts.push_back(t0);
  set_next_dt(get_dt0());
  assert ( ys.size() == 1 && ts.size() == 1
	   && "Reset succesfull." );
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
  debugging = DEFAULT_OUTPUT_DEBUG_INFO;
  min_dt = default_min_dt();
  set_max_delta_dt();
}

// A convenience function. Wraps the function f = y'. Depending on
// whether or not f takes optional arguments does the correct thing.
dVector RKF45::f(int t, const dVector& y) const {
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

// Subtracts dVector b from a. Outputs a new vector that is their
// difference. Assumes that the vectors are the same size. If
// they're not, raises an error.
dVector RKF45::difference(const dVector& a, dVector& b) const {
  return sum(a,scalar_multiplication(-1,b));
}

// Takes the scalar multiplicationof a dVector v with a scalar k
dVector RKF45::scalar_multiplication(double k, const dVector& v) const {
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
  double output = 0;
  for (dVector::const_iterator it=v.begin(); it != v.end(); ++it) {
    output += (*it) * (*it);
  }
  return sqrt(output);
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



