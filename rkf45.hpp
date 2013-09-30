// rkf45.hpp

// Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
// Time-stamp: <2013-09-30 16:04:57 (jonah)>

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
// methods and the Runge-Kutta-Feldberg method on wikipedia:\
// http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
// http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method

// ----------------------------------------------------------------------



// Usage
// ----------------------------------------------------------------------

// The RKF45 integrator expects a function of the form
// void f(double t, const double[n] y, int n),
// where y is the array representing the N-dimensional ODE system at time t.

// This means that if your system is not in this form you'll need to
// write it in this form before you can use RKF45.

// To start the algorithm you need an initial step size, so this must
// be input by hand.

// The error tolerance is, by default 0.1% of the absolute value of
// the smallest element in y. You can set absolute or relative error
// tolerance by using setter methods. You can also set the absolute
// error by using the constructor.

// You can choose how RKF45 outptus the solution after a given
// time. You can have it set fill a pre-allocated double array
// or you can have it output a vector.

// ----------------------------------------------------------------------


// Include guard
# pragma once
// Invludes
#include <vector>   // for output
#include <iostream> // For printing a given ode system
#include <iomanip> // For manipulating the output.
// Namespace specification. For convenience.
using namespace std;

// Bundles together the background methods and relevant variables for
// the 4-5 Runge-Kutta-Feldberg algorithm. I guess the best thing to
// call it would be an integrator.
class RKF45 {
public: // Constructors, destructors, and assignment operators.
  // Creates an empty integrator, to be initialized later.
  RKF45();
  // Creates an integrator of an n-dimensional ode system with
  // y'=f(t,y), initial time t0, initial conditions y0, initial step
  // size delta_t0, and absolute error tolerance absolute_tolerance
  RKF45(int n,void (*y)(double,double[],int),double t0, double[] y0,
	double delta_t0, double absolute_tolerance);
  // Creates an integrator of an n-dimensional ode system with
  // y'=f(t,y), initial time t0, initial conditions y0, and initial step
  // size delta_t0. The error tolerance is the default error tolerance.
  RKF45(int n,void (*y)(double,double[],int),double t0, double[] y0,
	double delta_t0);
  // Creates an integrator for the n-dimensional ode system with y'=f(t,x).
  // The other properties are assumed to be set by setter methods.
  RKF45(int n, void (*y)(double,double[],int));
  // Creates an integrator for an n-dimensional ode system. The other
  // properties are assumed to be set by setter methods.
  RKF45(int n);
}
