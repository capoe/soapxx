#include "power.hpp"

namespace soap {

PowerExpansion::PowerExpansion(Basis *basis) :
	_basis(basis),
	_L(basis->getAngBasis()->L()),
	_N(basis->getRadBasis()->N()) {
	_coeff = coeff_zero_t(_N*_N, _L+1);
}

void PowerExpansion::computeCoefficients(BasisExpansion *basex1, BasisExpansion *basex2) {
	if (!_basis) throw soap::base::APIError("PowerExpansion::computeCoefficients, basis not initialised.");
	BasisExpansion::coeff_t &coeff1 = basex1->getCoefficients();
	BasisExpansion::coeff_t &coeff2 = basex2->getCoefficients();
	for (int n = 0; n < _N; ++n) {
		for (int k = 0; k < _N; ++k) {
			for (int l = 0; l < (_L+1); ++l) {
				//std::cout << n << " " << k << " " << l << " : " << std::flush;
				std::complex<double> c_nkl = 0.0;
				for (int m = -l; m <= l; ++m) {
					//std::cout << m << " " << std::flush;
					c_nkl += coeff1(n, l*l+l+m)*std::conj(coeff2(k, l*l+l+m));
				}
				_coeff(n*_N+k, l) = c_nkl;
				//std::cout << std::endl;
			}
		}
	}
	//throw soap::base::APIError("");
	return;
}
void PowerExpansion::add(PowerExpansion *other) {
	assert(other->_basis == _basis &&
		"Should not sum expansions linked against different bases.");
	_coeff = _coeff + other->_coeff;
	return;
}

}

