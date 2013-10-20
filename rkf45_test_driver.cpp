// rkf45_test_driver.cpp

// Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
// Time-stamp: <2013-10-19 13:23:18 (jonah)>

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
using std::ofstream;

dVector f_harmonic(double t, const dVector& v) {
  dVector output;
  output.push_back(v[1]);
  output.push_back(-v[0]);
  return output;
}

dVector fxp(double t, const dVector& v,const dVector& power) {
  dVector output;
  output.push_back( pow(t,power[0]) );
  return output;
}

void print(dVector v) {
  cout << "[ ";
  for (dVector::const_iterator it = v.begin(); it != v.end(); ++it) {
    cout << (*it) << " ";
  }
  cout << "]";
}

dVector INITIAL_Y;
dVector OPTIONAL_ARGS;
const double T0 = 0;

int main () {
  // We'll reuse this stream several times.
  ofstream myfile;

  cout << "Testing the SHO." << "\n"
       << "------------------------------------------------------------\n"
       << "------------------------------------------------------------\n"
       << "\n" << endl;
    
  INITIAL_Y.push_back(0);
  INITIAL_Y.push_back(1);

  RKF45 integrator(f_harmonic,T0,INITIAL_Y);
  integrator.set_debug_output(true);
  integrator.set_absolute_error(1E-3);
  integrator.set_relative_error_factor(0);

  cout << "\nTesting the error method." << endl;
  cout << "Absolute error: " << integrator.get_absolute_error() << "\n"
       << "Relative error factor: "
       << integrator.get_relative_error_factor() << "\n"
       << "Error tolerance: " << integrator.get_error_tolerance() << "\n"
       << endl;

  cout << "Testing the initial state." << "\n"
       << "\tNumber of steps: " << integrator.steps() << "\n"
       << "\tdt0 = " << integrator.get_dt0() << "\n"
       << "\tNext dt = " << integrator.get_next_dt() << "\n"
       << "\tMax delta dt factor = "
       << integrator.get_max_delta_dt_factor() << "\n"
       << "-----------------------------------------------\n"
       << integrator << endl;

  cout << "\nTesting one step." << endl;
  integrator.step();
  cout << integrator << endl;

  cout << "\nTesting integration to finite t, say 6 pi." << endl;
  integrator.integrate(6*M_PI);

  cout << "\nSaving this output to file for plotting." << endl;
  myfile.open("SHO.dat");
  myfile << integrator << endl;
  myfile.close();
  /*  
  cout << "This concludes the test of the simple harmonic oscillator.\n"
       << endl;
  cout << "\nNow we want to test accuracy. We Is it 4th order accurate?\n"
       << "-------------------------------------------------------------"
       << endl;
  INITIAL_Y.resize(0);
  OPTIONAL_ARGS.resize(0);
  INITIAL_Y.push_back(0);
  OPTIONAL_ARGS.push_back(4);
  cout << "Initial y vector: ";
  print(INITIAL_Y);
  cout << endl;
  cout << "Optional arguments: ";
  print(OPTIONAL_ARGS);
  cout << endl;
  cout << "And f(0.1) = ";
  print(fxp(0.1,INITIAL_Y,OPTIONAL_ARGS));
  cout << " (it should equal 10^(-4).)" << endl;
  cout << "Testing integrator reset functions.\n"
       << "We will set the step size to 0.1 to avoid fitting to error."
       << endl;

  integrator.set_f(fxp);
  integrator.set_optional_args(OPTIONAL_ARGS);
  integrator.set_y0(INITIAL_Y);
  integrator.set_t0(1);
  integrator.set_absolute_error(0);
  integrator.set_relative_error_factor(1E-2);
  //  integrator.set_min_dt(0.1);
  //  integrator.set_max_dt(0.1);
  integrator.reset();
  cout << "Before integrating, here's the integrator pre-reset.\n"
       << "\tSteps: " << integrator.steps() << "\n"
       << "\tSize: " << integrator.size() << "\n"
       << "\ty0: ";
  print(integrator.get_y0());
  cout << "\n\tt0: " << integrator.get_t0() << "\n"
       << "\tdt0: " << integrator.get_dt0() << "\n"

       << "\tt: " << integrator.get_t() << "\n";
  cout << "State\n" << integrator << endl;

  cout << "Now let's try integrating the interval [0,1]" << endl;
  integrator.integrate(2);
  cout << integrator << endl;
  cout << "The integrator should have found the answer to be:\n"
       << "about 6.2. Is it?"
       << endl;
  integrator.print_state(cout);
  cout << endl;
  */
  return 0;
}
