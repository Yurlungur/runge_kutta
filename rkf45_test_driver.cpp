// rkf45_test_driver.cpp

// Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
// Time-stamp: <2013-12-08 18:05:48 (jonah)>

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
  // The coefficient in front from power[0] derivatives
  double coefficient = 1;
  for (int i = 0; i < (int)power[0]-1; i++) {
    output.push_back(v[i+1]);
    coefficient *= (int)power[0] - i;
  }
  output.push_back(coefficient);
  // The last component is time.
  output.push_back(1);
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
  double step_size;
  bool printing_power_data;

  cout << "Testing the SHO." << "\n"
       << "------------------------------------------------------------\n"
       << "------------------------------------------------------------\n"
       << "\n" << endl;
    
  INITIAL_Y.push_back(0);
  INITIAL_Y.push_back(1);

  RKF45 integrator(f_harmonic,T0,INITIAL_Y);
  integrator.set_debug_output(1);
  integrator.set_absolute_error(1E-10);
  integrator.set_relative_error_factor(1E-5);

  integrator.print_settings(cout);
  cout << "Integrator debug level: "
       << integrator.output_debug_info()
       << endl;
  cout << "\nTesting one step." << endl;
  integrator.step();
  cout << integrator << endl;

  cout << "\nTesting integration to finite t, say 6 pi." << endl;
  integrator.integrate(6*M_PI);

  cout << "\nSaving this output to file for plotting." << endl;
  myfile.open("SHO.dat");
  myfile << integrator << endl;
  myfile.close();

  cout << "This concludes the test of the simple harmonic oscillator.\n"
       << endl;
  cout << "\nNow we want to test accuracy. Is it 4th order accurate?\n"
       << "-------------------------------------------------------------"
       << endl;
  INITIAL_Y.resize(0);
  OPTIONAL_ARGS.resize(0);
  OPTIONAL_ARGS.push_back(4);
  for (int i = 0; i < 4+1; i++) {
    INITIAL_Y.push_back(0);
  }
  cout << "Initial y vector: ";
  print(INITIAL_Y);
  cout << endl;
  cout << "Optional arguments: ";
  print(OPTIONAL_ARGS);
  cout << endl;
  cout << "And f(0.1) = ";
  print(fxp(0.1,INITIAL_Y,OPTIONAL_ARGS));
  cout << "\nTesting integrator reset functions.\n"
       << "We will set the step size to 0.1 to avoid fitting to error."
       << endl;

  // ----------------------------------------------------------------------
  step_size = 1;
  printing_power_data = false;
  // ----------------------------------------------------------------------

  integrator.set_f(fxp);
  integrator.set_optional_args(OPTIONAL_ARGS);
  integrator.set_y0(INITIAL_Y);
  integrator.set_t0(0);
  integrator.set_absolute_error(100);
  integrator.set_relative_error_factor(100);
  integrator.set_next_dt(step_size);
  integrator.set_min_dt(step_size);
  integrator.set_max_dt(step_size);
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

  cout << "Now let's try integrating the interval [0,1].\n"
       << "Because the method is 4th-order accurate,\n"
       << "we should get the right answer in just one step."
       <<endl;
  integrator.integrate(1);
  cout << integrator << endl;
  cout << "The integrator should have found y[0] to be:\n"
       << "about 1.4641. Is it?"
       << endl;
  integrator.print_state(cout);
  if ( printing_power_data ) {
    cout << "\nPrinting the output to a file." << endl;
    myfile.open("x_to_the_fourth.dat");
    myfile << integrator << endl;
    myfile.close();
  }

  cout << "\n-------------------------------------------------------------"
       << endl;

  cout << "Let's see how accurate the integrator is to fifth order.\n"
       << "We don't expect perfect accuracy anymore."
       << endl;

  OPTIONAL_ARGS[0] = 5;
  integrator.set_optional_args(OPTIONAL_ARGS);
  INITIAL_Y.push_back(0);
  integrator.set_y0(INITIAL_Y);
  integrator.set_t0(0);
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
  integrator.integrate(1);
  cout << integrator << endl;
  cout << "The correct value for y[0] is:\n"
       << "about 1.0 Is it? (We don't expect it to be.)"
       << endl;
  integrator.print_state(cout);
  if ( printing_power_data ) {
    cout << "\nPrinting the output to a file." << endl;
    myfile.open("x_to_the_fifth_fourth_order_tools.dat");
    myfile << integrator << endl;
    myfile.close();
  }

  cout << "\n-------------------------------------------------------------"
       << endl;

  cout << "If we use the fifth-order Butcher Tableau for the integrator\n"
       << "to generate data rather than error estimates, we might get the\n"
       << "correct answer here. Let's try it."
       << endl;

  integrator.reset();
  integrator.set_use_5th_order_terms(true);
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
  integrator.integrate(1);
  cout << integrator << endl;
  cout << "The correct value for y[0] is:\n"
       << "about 1.0 Is it? (We do expect it to be.)"
       << endl;
  integrator.print_state(cout);
  if ( printing_power_data ) {
    cout << "\nPrinting the output to a file." << endl;
    myfile.open("x_to_the_fifth_fifth_order_tools.dat");
    myfile << integrator << endl;
    myfile.close();
  }

  cout << "\n-------------------------------------------------------------"
       << endl;


  cout << "\n\nThis concludes the test of the RKF45 class. Thanks!" << endl;

  return 0;
}
