# Changelog

## [0.0.3](https://github.com/jejjohnson/tracerax/compare/v0.0.2...v0.0.3) (2026-11-25)


### Features

* **bin:** cells from edges, and nearest and cloud-in-cell deposits that conserve mass at walls ([#36](https://github.com/jejjohnson/tracerax/issues/36)) ([32bb3bf](https://github.com/jejjohnson/tracerax/commit/32bb3bfed2bd467ebb77a773122eaa4d75decc81))
* **bin:** the residence accumulator, and snapshot and steady concentrations ([#38](https://github.com/jejjohnson/tracerax/issues/38)) ([da9195d](https://github.com/jejjohnson/tracerax/commit/da9195d3aef875798b77c777ec748c828ab3ad9c))
* **footprint:** surface footprints from backward runs ([#39](https://github.com/jejjohnson/tracerax/issues/39)) ([70fbc56](https://github.com/jejjohnson/tracerax/commit/70fbc56d00852d46c76991daf165d2120f36b1b1))

## [0.0.2](https://github.com/jejjohnson/tracerax/compare/v0.0.1...v0.0.2) (2026-11-24)


### Features

* **advect:** deterministic advection, as a solve and as a step ([#31](https://github.com/jejjohnson/tracerax/issues/31)) ([1f1d57f](https://github.com/jejjohnson/tracerax/commit/1f1d57f6d094784b3b9c563b1fe5ea8df6d80906))
* **advect:** random displacement with a diffusivity, so that a uniform cloud stays uniform ([#34](https://github.com/jejjohnson/tracerax/issues/34)) ([d81014f](https://github.com/jejjohnson/tracerax/commit/d81014f6feeea398af8cba3752fae80857560557))
* **advect:** the integration loop for any particle step ([#33](https://github.com/jejjohnson/tracerax/issues/33)) ([a6e6262](https://github.com/jejjohnson/tracerax/commit/a6e6262e94691ffde1db0af01db89b881e696076))
* **advect:** the turbulence protocol, the well-mixed Langevin step and walls that fold any number of times ([#32](https://github.com/jejjohnson/tracerax/issues/32)) ([794faf6](https://github.com/jejjohnson/tracerax/commit/794faf694de671d9bca39fff0949a9af59144676))
* **particles:** the particle state, the step protocol, and seeding at points, along lines and over boxes ([#28](https://github.com/jejjohnson/tracerax/issues/28)) ([3f16f72](https://github.com/jejjohnson/tracerax/commit/3f16f72bef1bea4b93f35ffa3bd620ae367ee9ad))
* **velocity:** velocities at particles, uniform, gridded on staggered grids, and weighted sums ([#30](https://github.com/jejjohnson/tracerax/issues/30)) ([0bb5502](https://github.com/jejjohnson/tracerax/commit/0bb5502fec25f0a730928b63d1df7add16d96a4b))


### Bug Fixes

* **advect:** release gradients, float32 durations, masked particles and domains ([#35](https://github.com/jejjohnson/tracerax/issues/35)) ([a042c86](https://github.com/jejjohnson/tracerax/commit/a042c86d8a13d86d88d1c98d443febb47ad9d37b))

## 0.0.1 (2026-11-24)


### Tests

* reference harness with closed forms, stored plumax values, statistical tests and transform checks ([#26](https://github.com/jejjohnson/tracerax/issues/26)) ([9ac67a3](https://github.com/jejjohnson/tracerax/commit/9ac67a3d69d4bc9141d5f4e3c70618a0dce08548))

## Changelog
