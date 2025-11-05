#!/bin/bash
for file in *.chk; do
    formchk "$file" "${file%.chk}.fchk"
done
