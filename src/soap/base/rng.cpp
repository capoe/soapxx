#include <cmath>
#include <iostream>
#include <stdexcept>
#include "soap/base/rng.hpp"

namespace soap { namespace base {

//using namespace std;

void RNG::init(int nA1, int nA2, int nA3, int nB1) {

  nA1 = nA1 % 178 + 1;
  nA2 = nA2 % 178 + 1;
  nA3 = nA3 % 178 + 1;
  nB1 = nB1 % 169;

  if ((nA1 == 1) && (nA2 == 1) && (nA3 == 1)) {
    // Should not all be unity
    std::cout << std::flush << "WARNING: MARSAGLIA RNG INITIALISED INCORRECTLY. "
         << "ADAPTING SEEDS APPROPRIATELY." << std::endl;
    nA1 += nB1;
  }

  int mA1, mA2, mA3, mANEW, mB1, mHELP;
  int i1, i2;
  double varS, varT;
  MARSarray = std::vector<double>(MARS_FIELD_SIZE);
  mA1 = nA1;
  mA2 = nA2;
  mA3 = nA3;
  mB1 = nB1;
  MARSi = 97;
  MARSj = 33;

  for (i1 = 1; i1 < MARS_FIELD_SIZE; i1++) {
    varS = 0.0;
    varT = 0.5;
    for (i2 = 1; i2 < 25; i2++) {
      mANEW = (((mA1 * mA2) % 179) * mA3) % 179;
      mA1 = mA2;
      mA2 = mA3;
      mA3 = mANEW;
      mB1 = (53 * mB1 + 1) % 169;
      mHELP = (mB1 * mANEW) % 64;
      if (mHELP > 31) varS += varT;
      varT *= 0.5;
    }

    MARSarray[i1] = varS;
  }

  MARSc = 362436.0 / 16777216.0;
  MARScd = 7654321.0 / 16777216.0;
  MARScm = 16777213.0 / 16777216.0;

  return;
}

double RNG::uniform(void) {

  double ranMARS;

  ranMARS = MARSarray[MARSi] - MARSarray[MARSj];
  if (ranMARS < 0.0) ranMARS += 1.0;

  MARSarray[MARSi] = ranMARS;

  MARSi--;
  if (MARSi < 1) MARSi = 97;

  MARSj--;
  if (MARSj < 1) MARSj = 97;

  MARSc -= MARScd;
  if (MARSc < 0.0) MARSc += MARScm;

  ranMARS -= MARSc;
  if (ranMARS < 0.0) ranMARS += 1.0;

  return ranMARS;
}

int RNG::uniform_int(int max_int) {
  return floor(max_int * uniform());
}

double RNG::gaussian(double sigma) {

  double r = sigma * sqrt(-2.0 * log(1 - RNG::uniform()));
  double theta = 2.0 * 3.14159265359 * RNG::uniform();
  return r * cos(theta);  // second independent number is r*sin(theta)
}

}}
