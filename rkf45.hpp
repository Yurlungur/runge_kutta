// rkf45.hpp

// Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
// Time-stamp: <2013-10-14 14:43:18 (jonah)>

// This is the prototype for my implementation of the 4-5
// Runge-Kutta-Feldberg adaptive step size integrator. For simplicity,
// and so that I can bundle public and private methods together, I take
// an object-oriented approach.


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


// Include guard
# pragma once
// Includes
#include <vector>   // for output and internal variables
#include <iostream> // For printing a given ode system
#include <iomanip> // For manipulating the output.
#include <float.h> // For machine precision information
#include <cmath> // for math
// Namespace specification. For convenience.
using std::cout;
using std::endl;
using std::vector;
using std::ostream;
using std::istream;
using std::sqrt;

// We use the dVector type a lot, so let's define a type for it
// to make things more readable.
typedef vector<double> dVector;

// Bundles together the background methods and relevant variables for
// the 4-5 Runge-Kutta-Feldberg algorithm. I guess the best thing to
// call it would be an integrator.
class RKF45 {
public: // Constructors, destructors, and assignment operators.

  // Creates an empty integrator, to be initialized later.
  RKF45();

  // Creates an integrator for the ode system with y'=f(t,y).  The other
  // properties are assumed to be set by setter methods. The size of
  // the vector is assumed to be appropriate.
  RKF45(double (*y)(double,const dVector&));

  // Creates an integrator for the ode system with y'=f(t,y), where
  // f(t,y) = f(t,y,optional_args). The size of the vectors is assumed
  // to be appropriate.
  RKF45(double (*y)(double,const dVector&,const dVector&));

  // Creates an integrator for an ode system with y'=f(t,y), with
  // initial time t0 and initial conditions y0. The size of the system
  // is inferred from the size of the initial conditions vector.
  RKF45(double (*y)(double, const dVector&), double t0,
	const dVector& y0);

  // Creates an integrator for an ode system with
  // y'=f(t,y)=f(t,y,optional_args), with initial time t0 and initial
  // conditions y0. The size of the system is inferred from the size
  // of the initial conditions vector. The optional_args are passed in
  // now too. The size is inferred from the input vector.
  RKF45(double (*y)(double,const dVector&,const dVector&),
	const dVector& y0, const dVector& optional_args);

  // Creates an integrator of an ode system with y'=f(t,y), initial
  // time t0, initial conditions y0, initial step size delta_t0,
  // relative error tolerance relative_tolerance, and absolute error
  // tolerance absolute_tolerance. The dimension of the system is
  // inferred from the size of y0.
  RKF45(double (*y)(double,const dVector&),
	double t0, const dVector& y0,
	double delta_t0, double relative_tolerance,
	double absolute_tolerance);
  
  // Creates an integrator of an ode system with y'=f(t,y), initial
  // time t0 = 0, initial conditions y0, initial step size delta_t0,
  // relative error tolerance relative_tolerance, and absolute error
  // tolerance absolute_tolerance. The size of the system is inferred
  // from the initial data.
  RKF45(double (*y)(double,const dVector&),
	const dVector& y0, double delta_t0,
	double relative_tolerance, double absolute_tolerance);
  
  // Creates an integrator of an ode system with y'=f(t,y), initial
  // time t0, initial conditions y0, initial step size delta_t0,
  // relative error tolerance relative_tolerance, and absolute error
  // tolerance absolute_tolerance.  f(t,y) =
  // f(t,y,optional_args). There are m-optional arguments. The
  // dimension of the system and the number of optional arguments are
  // inferred from size of the vectors.
  RKF45(double (*y)(double, const dVector&,const dVector&),
	double t0,
	const dVector& y0, const dVector& optional_args,
	double delta_t0,
	double relative_tolerance,double absolute_tolerance);

  // Creates an integrator of an ode system with y'=f(t,y), initial
  // time t0=0, initial conditions y0, initial step size delta_t0,
  // relative error tolerance relative_tolerance, and absolute error
  // tolerance absolute_tolerance.  f(t,y) = f(t,y,optional_args).
  RKF45(double (*y)(double,const dVector&,const dVector&),
	const dVector& y0, const dVector& optional_args,
	double delta_t0,
	double relative_tolerance, double absolute_tolerance);

  // Creates an integrator of an ode system with y'=f(t,y), initial
  // time t0, initial conditions y0, and initial step size
  // delta_t0. The error tolerance is the default error tolerance.
  RKF45(double (*y)(double,const dVector&),
	double t0, const dVector y0, double delta_t0);

  // Creates an integrator of the ode system with y'=f(t,y), initial
  // time t0, initial conditions y0, initial step size delta_t0.
  // f(t,y) = f(t,y,optional_args). The error tolerance is the
  // default.
  RKF45(double (*y)(double,const dVector&,const dVector&),
	double t0, const dVector y0[], const dVector& optional_args,
	double delta_t0);

  // Creates an integrator of the ode system with y'=f(t,y), initial
  // time t0, initial conditions y0, initial step size delta_t0.
  // f(t,y) = f(t,y,optional_args). The error tolerance is the
  // default. f(t,y) = f(t,y,optional_args)
  RKF45(double (*y)(double,const dVector&,const dVector&),
	double t0, const dVector& y0, const dVector& optional_args,
	double delta_t0);

  // Copy constructor. Generates an exact copy of an integrator.
  RKF45(const RKF45 &integrator);

  // Destructor.
  ~RKF45();

  // Assignment operator. Copies one obje ct into another.
  RKF45& operator = (const RKF45 &integrator);


public: // Public interface

  // Getters 
  // ----------------------------------------------------------------------
  // Finds the error tolerance based on the current state of the system.
  double get_error_tolerance() const;

  // Returns the size of the system. Assumes the system stays the same
  // size. If initial data has not yet been set, may return zero.
  double size() const;

  // Returns the initial data. Returns an empty vector if no
  // initial data has been set yet. Does NOT pass by reference.
  dVector get_y0() const;

  // Returns the start time.
  double get_t0() const;
  
  // Returns true if the system is set to use optional arguments,
  // false otherwise.
  bool using_optional_args() const;

  // Returns the optional arguments if they are in use. If they are
  // not in use or if they haven't been set yet, returns an empty
  // array otherwise.
  dVector get_optional_args() const;

  // Returns the maximum step size allowed.
  double get_max_dt() const;

  // Returns the initial step size.
  double get_dt0() const;

  // Returns the previous step size. If no steps have been performed,
  // returns -1.
  double get_last_dt() const;

  // Returns the next step size.
  double get_next_dt() const;

  // Returns the current absolute error
  double get_absolute_error() const;

  // Returns the current relative error factor
  double get_relative_error_factor() const;

  // Returns the safety margin for step-size choice
  double get_safety_margin() const;

  
  // Setters
  // ----------------------------------------------------------------------
  // Sets the function y'=f. One version takes optional arguments. One
  // does not. The second vector is optional arguments.
  void set_f(double (*f)(const dVector&));
  void set_f(double (*f)(const dVector&,const dVector&));

  // Sets the initial y vector.
  void set_y0(const dVector& y0);

  // Sets the start time. Passing in no arguments resets to the
  // default.
  void set_t0();
  void set_t0(double t0);

  // Sets the optional arguments. If the function f does not take
  // optional arguments, raises an error.
  void set_optional_args(const dVector& optional_args);

  // Sets the maximum allowed step size. Passing in no arguments
  // resets to the default.
  void set_max_dt();
  void set_max_dt(double max_dt);

  // Sets the initial step size. Passing in no arguments resets to the
  // default.
  void set_dt0();
  void set_dt0(double dt0);

  // Sets the next step size. Use with caution! If you set the next
  // step size, the value chosen automatically is forgotten!
  void set_next_dt(double next_dt);

  // Sets the absolute error. Passing in no arguments resets to the
  // default.
  void set_absolute_error();
  void set_absolute_error(double absolute_error);

  // Sets the relative error factor. Passing in no arguments resets to
  // the default.
  void set_relative_error_factor();
  void set_relative_error_factor(double relative_error_factor);

  // Sets the safety margin for step size choices. Passing in no
  // arguments resets to the default.
  void set_safety_margin();
  void set_safety_margin(double safety_margin);


  // Data output
  // ----------------------------------------------------------------------
  // Returns the total number of steps the integrated has iterated through
  int get_num_steps() const;

  // Returns the current time step.
  double get_t() const;

  // Returns the time after n steps.
  double get_t(int n) const;

  // Returns the current state of the y vector. Passes by value, so
  // very slow.
  dVector get_y() const;

  // Returns the y vector after n steps. Passes by value, so very
  // slow.
  dVector get_y(int n) const;
  
  // Fills an input vector with the current state of the y
  // vector. 
  void get_y(dVector& y_data) const;

  // Fills an input vector with the state of the y vector after n
  // steps.
  void get_y(dVector& y_data, int n) const;

  // Fills an input array with the current state of the y
  // vector. Assumes that the array is the appropriate size. If the
  // array size is wrong, you will sefgault.
  void get_y(double y_data[]) const;
  
  // Tests whether or not a given set of constraints is
  // satisfied. Takes a boolean function of the vector y and tests
  // whether y(t) satisfies the constraint function. This is the fast,
  // safe solution to the problem of monitoring constriants.
  bool test_constraints(bool (*constraints)(const dVector), double t) const;

  // Tests whether or not a given set of constraints is
  // satisfied. Takes a boolean function of the vector y and tests
  // whether y(t) satisfies the constraint function. t is set to the
  // current time. This is the fast, safe solution to the problem of
  // monitoring constriants.
  bool test_constraints(bool (*constraints)(const dVector)) const;

  // Tests by how much the y vector at time t fails to satisfy the
  // constraint function passed in. The constraint function should
  // return a vector<double> where each element shows how much the y
  // vector failed to satisfy the appropriate constraint
  // equation. Passed by reference.
  dVector& test_constraints(dVector (*constraints)(const dVector)) const;

  // Prints the integration history to an output stream. No stream
  // choice means it prints to cout. This is not the just the current
  // state of the system. This is everything.
  void print() const;
  void print(ostream& out) const;

  // Overload the stream input operator. Works like print(out).
  friend ostream& operator <<(ostream& out, const RKF45& in);


  // Integration control
  // ----------------------------------------------------------------------
  // Integrates by a single time step. The time step is either chosen
  // internally from the last integration step or set by set_dt0 or
  // set_next_dt.
  void step();

  // Integrates up to to time t_final. If the current time is after
  // t_final, does nothing. The initial step size is set by set_dt0 or
  // by set_next_dt.
  void integrate(double t_final);

  // Resets the integrator to time t_initial. If no initial time is
  // supplied, resets the integrator to the initial time set when the
  // user set the initial data. This is useful for approaches like the
  // shooting method.
  void reset(double t_initial);
  void reset();

private: // Implementation details

  // Private fields.
  // ----------------------------------------------------------------------

  // Default values
  static const double DEFAULT_T0 = 0; // Default initial t data
  // Default for whether or not we use a function with optional
  // arguments
  static const bool DEFAULT_USE_OPTIONAL_ARGS = false;
  // Default maximum step size. 
  static const double DEFAULT_MAX_DT = 1E100;
  // Default initial step size.
  static const double DEFAULT_DT0 = 0.01;
  // The last step size needs a value before any steps have been
  // taken. We set it to minus 1.
  static const double LAST_STEP_SIZE_NO_STEPS_TAKEN = -1;
  // Default relative error factor
  static const double DEFAULT_RELATIVE_ERROR_FACTOR = 0.001;
  // Default safety factor for step size choices. Shrinks the step
  // size slightly for safety.
  static const double DEFAULT_SAFETY_MARGIN = 0.1;
  // Default value for the absolute error
  static const double DEFAULT_ABSOLUTE_ERROR = 0.0003;

  // The pointer to the function f.
  // f(double t, const dVector& y,const dVector& optional_args)
  double (*f_with_optional_args)(double,const dVector&, const dVector&);
  // f(double t, const dVector& y)
  double (*f_no_optional_args)(double,const dVector&);

  // Initial conditions.
  dVector y0; // initial y-data
  double t0; // = DEFAUL_T0; // Initial t-data.

  // Optional arguments
  // Whether or not we use a function with optional arguments
  bool use_optional_arguments; // = DEFAULT_USE_OPTIONAL_ARGS;
  dVector optional_args;

  // Step-size data
  // The maximum step size.
  double max_dt;// = DEFAULT_MAX_DT;
  // The initial step size. This has a default value, but you should
  // really put it in by hand. Note that if you set your max step size
  // but not your delta_t0, you might run into trouble if your max
  // step size is smaller than delta_t0.
  double dt0; // = DEFAULT_DT0;
  // The last step size. This is what we used in the most recent
  // step. 
  double last_dt; // = LAST_STEP_SIZE_NO_STEPS_TAKEN;
  // The next step size. This is calculated during a step.
  double next_dt; // = dt0;

  // Error tolerance values.
  double absolute_error;// = DEFAULT_ABSOLUTE_ERROR;
  // Relative error is relative_error_factor * min(abs(y))
  double relative_error_factor;// = DEFAULT_RELATIVE_ERROR_FACTOR;
  // The safety margin for step-size choice
  double safety_margin;// = DEFAULT_SAFETY_MARGIN;

  // These fields keep track of the system over many
  // timesteps. They're used to keep track of the current state of the
  // system.
  dVector ts; // A vector listing all t values.
  vector< dVector > ys; // A vector listing all the y values. 
  
  // Private methods
  // ----------------------------------------------------------------------
  
  // A convenience function. Wraps the function f = y'. Depending on
  // whether or not f takes optional arguments does the correct thing.
  double f(int t, const dVector& y) const;

  // Finds the relative error tolerance based on the current state of
  // the system.
  double get_relative_error_tolerance() const;

  // Calculates the 2-norm of the vector v
  double norm(const dVector& v) const;

  // Adds two dVectors a and b and outputs a new dVector that is their
  // sum. Assumes that the vectors are the same size. If they're not,
  // raises an error.
  dVector sum(const dVector& a, const dVector& b) const;

  // Calculuates the sum of all elements in a dVector.
  double sum(const dVector& v) const;

  // Takes the scalar product of a dVector v with a scalar k
  dVector scalar_product(double k, const dVector& v) const;

};

