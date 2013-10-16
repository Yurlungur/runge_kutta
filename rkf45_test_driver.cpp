// rkf45_test_driver.cpp

// Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
// Time-stamp: <2013-10-15 15:52:12 (jonah)>

// This is my test driver for the 4-5 Runge-Kutta_Fehlberg adatprive
// step size integrator.


// Includes
#include <vector>   // for output and internal variables
#include <iostream> // For printing a given ode system
#include <iomanip> // For manipulating the output.
#include <float.h> // For machine precision information
#include <cmath> // for math
#include <cassert> // For error checking
#include <fstream> // For files
#include "rkf45.hpp" // the header for this program
// Namespace specification. For convenience.
using std::cout;
using std::endl;
using std::vector;
using std::ostream;
using std::istream;
using std::abs;
using std::pow;
using std::ofstream;

dVector f_harmonic(double t, const dVector& v) {
  dVector output;
  output.push_back(-v[1]);
  output.push_back(v[0]);
  return output;
}

dVector fxp(double t, const dVector& v,const dVector& power) {
  dVector output;
  double prefactor = 1;
  for (int i = 0; i < power[0]; i++) {
    prefactor *= (power[0] - i);
    output.push_back(prefactor * pow(t,power[0] - (i + 1)));
  }
  return output;
}

dVector INITIAL_Y;
dVector OPTIONAL_ARGS;
const double T0 = 0;

int main () {
  // We'll reuse this stream several times.
  ofstream myfile;

  cout << "Testing the SHO." << endl;
  INITIAL_Y.push_back(1);
  INITIAL_Y.push_back(0);

  RKF45 integrator(f_harmonic,T0,INITIAL_Y);
  integrator.set_debug_output(false);
  integrator.set_absolute_error(1.0E-2);
  integrator.set_relative_error_factor(1);
  integrator.set_min_dt(.1);
  integrator.set_max_dt(.1);

  cout << "Testing the initial state." << endl;
  cout << integrator << endl;

  cout << "\nTesting one step." << endl;
  integrator.step();
  cout << integrator << endl;

  cout << "\nTesting integration to finite t, say 6 pi." << endl;
  integrator.integrate(6*M_PI);
  cout << integrator << endl;

  cout << "\nSaving this output to file for plotting." << endl;
  myfile.open("SHO.dat");
  myfile << integrator << endl;
  myfile.close();
  /*
  cout << "\nNow we want to test accuracy. We Is it 4th order accurate?"
       << endl;
  OPTIONAL_ARGS.push_back(4);
  integrator.set_f(fxp);
  integrator.set_optional_args(OPTIONAL_ARGS);
  integrator.set_y0(INITIAL_Y);
  integrator.set_t0(T0);
  integrator.reset();
  INITIAL_Y = dVector(4,0);
  INITIAL_Y.push_back(24);

  cout << "Now let's try integrating the interval [0,1]" << endl;
  integrator.integrate(1);
  cout << integrator << endl;
  cout << "The integrator should have found the answer to be 1/5. Is it?"
       << endl;
  integrator.print_state(cout);
  cout << endl;
  */

  return 0;
}
