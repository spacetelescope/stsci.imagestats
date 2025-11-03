.. _release_notes:

=============
Release Notes
=============

1.9.0 (03-November-2025)
========================

- Added support for Python 3.13 and 3.14. Package maintenance.

1.8.3 (12-June-2024)
====================

- rename ``master`` branch to ``main`` [#57]
- build on macOS ARM64 [#62]
- build with Numpy 2.0 release candidate [#63]

1.8.2 (04-April-2024)
=====================

- Bug Fix: Improve handling of floating point accuracy issues that can lead to
  memory violation on some systems. The fix from [#58] was only partial. [#59]


1.8.1 (22-March-2024)
=====================

- Bug Fix: Improve handling of floating point accuracy issues that can lead to
  memory violation on some systems. [#58]


1.8.0 (07-December-2023)
========================

- Removed support for Python 3.7 and 3.8. [#52]

- Improvements in the package infrastructure. [#52, 55]


1.7.0 (03-December-2023)
========================

- Improve speed of computing histograms (``histogram1d``). [#44]

- Added unit tests. Significantly improved code coverage. See [#40, #46]

- Compatibility improvements with Python 3.12. Package maintenance.
