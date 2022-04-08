#!/bin/bash
doExp()
{
echo "${1} evaluate starting..."
#python reg_exp.py -f DATA/"${1}"_0.png -m DATA/"${1}"_1.png -ma DATA/"${1}"_m.png -o rst/reg/"${1}"
#python reg_result.py -fo DATA/"${1}"_0.png -f DATA/Fiji/"${1}"/"${1}"_0_e.png -m DATA/"${1}"_1.png -r DATA/Fiji/"${1}"/"${1}"_1_e.png -n elastic -o rst/reg/"${1}"
python reg_result.py  -fo DATA/"${1}"_0.png -f DATA/Fiji/"${1}"/"${1}"_0_b.png -m DATA/"${1}"_1.png -r DATA/Fiji/"${1}"/"${1}"_1_b.png -n bUnwarpj -o rst/reg/"${1}"
mkdir rst/reg/"${1}"
echo rst/reg/"${1}" " is created"
echo " ---------------------------------------------------------------"
}
mkdir rst/reg
source activate pytorch
doExp SW13L63
doExp ZW35L414
doExp DOLW7
doExp ZW7L18
doExp SS2L30

echo "All done"

