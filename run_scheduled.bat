@echo off
REM Daily collection (run from the package root). Not needed to reproduce the analysis.
cd /d "%~dp0"
python collect_gt.py
python collect_gt_nocat.py
