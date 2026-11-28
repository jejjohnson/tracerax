# Changelog

## [0.2.1](https://github.com/jejjohnson/tracerax/compare/v0.2.0...v0.2.1) (2026-11-28)


### Miscellaneous

* release 0.2.1 ([#86](https://github.com/jejjohnson/tracerax/issues/86)) ([de92db9](https://github.com/jejjohnson/tracerax/commit/de92db90d52250305aca97753cbffd93b3955f61))

## [0.2.0](https://github.com/jejjohnson/tracerax/compare/v0.1.0...v0.2.0) (2026-11-27)


### ⚠ BREAKING CHANGES

* **bin:** cgrid_grids(times, x, y, z) becomes cgrid_grids(times, u, v, w), with each argument the positions of one component along (x, y) or (x, y, z); the gridax extra is removed.
* **velocity:** cgrid_grids(times, x, y, z) becomes cgrid_grids(times, u, v, w), with each argument the positions of one component along (x, y) or (x, y, z); the gridax extra is removed.

### Code Refactoring

* **bin:** a deposit is one Binning plan on the grid of the cells, from functax ([#83](https://github.com/jejjohnson/tracerax/issues/83)) ([22cd05e](https://github.com/jejjohnson/tracerax/commit/22cd05ee64668a5bc52956f7dd172e2f4b7fec85))
* **velocity:** cgrid_grids takes the positions of each component, and the unused gridax extra goes ([#82](https://github.com/jejjohnson/tracerax/issues/82)) ([d4ea873](https://github.com/jejjohnson/tracerax/commit/d4ea873c6cc863f9144c756aca37b30b0289254f))

## [0.1.0](https://github.com/jejjohnson/tracerax/compare/v0.0.7...v0.1.0) (2026-11-27)


### ⚠ BREAKING CHANGES

* **analysis:** tracerax.footprint (surface, Footprint) is removed. A backward run of Langevin under integrate with Residence(cells, where=...) gives the residence; plumax divides it by N rho h_s.
* **footprint:** tracerax.footprint (surface, Footprint) is removed. A backward run of Langevin under integrate with Residence(cells, where=...) gives the residence; plumax divides it by N rho h_s.
* **advect:** integrate with a Deterministic step is the one way to advect, and the diffrax solve path is removed ([#74](https://github.com/jejjohnson/tracerax/issues/74))

### Code Refactoring

* **advect:** integrate with a Deterministic step is the one way to advect, and the diffrax solve path is removed ([#74](https://github.com/jejjohnson/tracerax/issues/74)) ([5563805](https://github.com/jejjohnson/tracerax/commit/5563805c4f6ba326899bc409eb15dc3a4ac9c101))
* **analysis:** the trajectory scores and the dispersions go to diagnostix ([#79](https://github.com/jejjohnson/tracerax/issues/79)) ([ecf44bb](https://github.com/jejjohnson/tracerax/commit/ecf44bb610aefdc61de7e690f9922f97592bc2db))
* **footprint:** the footprint module goes, and plumax builds it from tracerax's pieces ([#78](https://github.com/jejjohnson/tracerax/issues/78)) ([64acbbf](https://github.com/jejjohnson/tracerax/commit/64acbbfd9372fb4d7f791982f064e632cb3fdd3e))

## [0.0.7](https://github.com/jejjohnson/tracerax/compare/v0.0.6...v0.0.7) (2026-11-26)


### Bug Fixes

* **advect:** a NaN diffusivity gives NaN, so a particle on missing land stops, through pipekit-array's safe_sqrt ([#72](https://github.com/jejjohnson/tracerax/issues/72)) ([869f698](https://github.com/jejjohnson/tracerax/commit/869f6986758d5d8d6a5a3b188d8dd225a324951e)), closes [#60](https://github.com/jejjohnson/tracerax/issues/60)
* **advect:** integrate refuses an adjoint it does not take, in place of an AttributeError ([#73](https://github.com/jejjohnson/tracerax/issues/73)) ([208c5c1](https://github.com/jejjohnson/tracerax/commit/208c5c1f9e946077e502b99e15b476c9c24b27fe)), closes [#61](https://github.com/jejjohnson/tracerax/issues/61)

## [0.0.6](https://github.com/jejjohnson/tracerax/compare/v0.0.5...v0.0.6) (2026-11-25)


### Bug Fixes

* **advect:** the walls of Walls are finite ([#57](https://github.com/jejjohnson/tracerax/issues/57)) ([2027e23](https://github.com/jejjohnson/tracerax/commit/2027e23a7d15a50ed36ad4f9584f1421d8503622))
* **velocity:** cgrid_grids takes axes from outside a jit ([#56](https://github.com/jejjohnson/tracerax/issues/56)) ([532b012](https://github.com/jejjohnson/tracerax/commit/532b012d53722e609bcf59cdc50029d8231bc871))

## [0.0.5](https://github.com/jejjohnson/tracerax/compare/v0.0.4...v0.0.5) (2026-11-25)


### Bug Fixes

* **deps:** geophysix v0.0.9, integrix v0.0.11 and gridax v0.3.3 ([#54](https://github.com/jejjohnson/tracerax/issues/54)) ([ba5d483](https://github.com/jejjohnson/tracerax/commit/ba5d483332431ec640e4c6906a011109dbeefafa))

## [0.0.4](https://github.com/jejjohnson/tracerax/compare/v0.0.3...v0.0.4) (2026-11-25)


### Features

* **advect:** inertial particles, by the Maxey–Riley equation and its slow manifold ([#49](https://github.com/jejjohnson/tracerax/issues/49)) ([277117b](https://github.com/jejjohnson/tracerax/commit/277117b7a02580aa3004b47124c3359823630a72)), closes [#44](https://github.com/jejjohnson/tracerax/issues/44)
* **advect:** particles on the sphere, stepped as unit vectors ([#47](https://github.com/jejjohnson/tracerax/issues/47)) ([8f114ef](https://github.com/jejjohnson/tracerax/commit/8f114eff6aad0fee6f516469520e0f1276be9a8b)), closes [#40](https://github.com/jejjohnson/tracerax/issues/40)
* **analysis:** dispersion and the skill of a trajectory ([#52](https://github.com/jejjohnson/tracerax/issues/52)) ([0fe825c](https://github.com/jejjohnson/tracerax/commit/0fe825c389abe586dd75b80da139e41794b39432)), closes [#43](https://github.com/jejjohnson/tracerax/issues/43) [#45](https://github.com/jejjohnson/tracerax/issues/45)
* **analysis:** finite-time Lyapunov exponents from the flow map's Jacobian ([#51](https://github.com/jejjohnson/tracerax/issues/51)) ([98822df](https://github.com/jejjohnson/tracerax/commit/98822dfdeda73ba345852dbdd23e5bd8b12dfffe)), closes [#41](https://github.com/jejjohnson/tracerax/issues/41)
* **analysis:** residence times and connectivity between labelled regions ([#50](https://github.com/jejjohnson/tracerax/issues/50)) ([c1ab886](https://github.com/jejjohnson/tracerax/commit/c1ab886d90c965d9b8fd7a41b4bb02bcf725baec)), closes [#42](https://github.com/jejjohnson/tracerax/issues/42)

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
