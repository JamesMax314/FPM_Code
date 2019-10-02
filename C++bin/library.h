#ifndef C__BIN_LIBRARY_H
#define C__BIN_LIBRARY_H

#include <iostream>
#include <complex>

using namespace std;

void div_part(complex<float>* n, complex<float>* d, complex<float>* o, int len);
void abs_part(complex<float>* ptr, int len);
void conjug_part(complex<float>* ptr, int len);

#endif //C__BIN_LIBRARY_H