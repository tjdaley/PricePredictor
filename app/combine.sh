#!/bin/sh
cd ../data
rm XYZ_enriched_data.csv
awk '(NR == 1) || (FNR > 1)' *_enriched_*.csv > XYZ_enriched_data.csv