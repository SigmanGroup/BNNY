#!/bin/bash
#SBATCH --partition=notchpeak-shared-short
#SBATCH --account=notchpeak-shared-short
#SBATCH --time=8:00:00
#SBATCH --ntasks=4
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N

# How to use on the CHPC:
# run this script by navigating into the directory with the .sdf files and “mm_reference.com” file
#      run "sbatch prep_MM_npsh.bash"
# potentially useful Schrodinger commands:
#      ml schrodinger
#      $SCHRODINGER/jobcontrol -list all (list all your jobs)
#      $SCHRODINGER/licadmin stat (shows token usage for the Sigman Group)
#      $SCHRODINGER/jobcontrol -list (shows only your incomplete jobs)
#      $SCHRODINGER/jobcontrol -delete all (deletes only completed jobs)

module load schrodinger
export SCHRODINGER='/uufs/chpc.utah.edu/sys/installdir/schrodinger/2024-4'
export SCHRODINGER_LICENSE_RETRY=300m

COM_REFERENCE=${PWD}'/mm_reference.com' #run a manual macromodel job and save the .com file after removing the first two lines with input/output file names

##python SMART_sbc_DuBois.py 31  ##can call a script to add constraints to the molecules

if [[ ! -d ${PWD}/output/ ]]; then #if the output directory doesn't exist, make it y
mkdir ${PWD}/output/
fi

if [[ ! -d ${PWD}/output/clustered ]]; then #if the output directory doesn't have a cluster folder, make one
mkdir ${PWD}/output/clustered
fi

for file in $PWD/*.mae; do #convert all .sdfs to maestro files
echo $file
filename=$(basename $file) #file.sdf
nopathnoext=${filename%.*} #file

if [[ ! -d ${PWD}/${nopathnoext} ]]; then # make subdirectory for each structure

mkdir $nopathnoext #make a directory for each compound for all output files and move the maestro file into it
#cp ${nopathnoext}.mae ${nopathnoext}/${nopathnoext}_Rh1.mae #can make multiple copies and add M/change oxidation/etc.
mv ${nopathnoext}.mae ${nopathnoext}/${nopathnoext}.mae
#mv ${nopathnoext}*.sbc ${nopathnoext}/

cd ${nopathnoext}/

#for r in {1..2..1}; do #this was for when each sdf file was made into two separate .coms that were modified, loop through them

echo "${nopathnoext}.mae" > ${nopathnoext}.com #make the file names consistent on all lines
echo "${nopathnoext}_out.maegz" >> ${nopathnoext}.com
cat $COM_REFERENCE >> ${nopathnoext}.com #for each maestro file, add names for input and output into the reference com file, ultimately generating a com file for each compound

#echo "running prep wizard"
#echo " "
#$SCHRODINGER/utilities/prepwizard -nohtreat -noprotassign ${nopathnoext}_Rh${r}.mae ${nopathnoext}_Rh${r}-prep.mae -WAIT

#cd ../ ##this section is to modify the structure
#echo "adjusting .mae"
#echo " "
#$SCHRODINGER/run fix_prepwizard_DuBois_SMART.py ${nopathnoext}/${nopathnoext}_Rh${r}.mae
#cd ${nopathnoext}/

pass=`$SCHRODINGER/jobcontrol -list | grep -c ’running\|done\|launched’`
while [ $pass -ge $SLURM_NTASKS ]; do
sleep 60
pass=`$SCHRODINGER/jobcontrol -list | grep -c ’running\|done\|launched’`
done

$SCHRODINGER/bmin ${nopathnoext} #submit button

# check for avaliable MM licence tokens
lic=`$SCHRODINGER/licadmin STAT | grep 'MMOD_MACROMODEL'`
read -ra spl <<<"$lic"
while [ ${spl[10]} -ge 24 ]; do
sleep 60
lic=`$SCHRODINGER/licadmin STAT | grep 'MMOD_MACROMODEL'`
read -ra spl <<<"$lic"
done

$SCHRODINGER/bmin ${nopathnoext} # submit conf search to Schrodinger queue

#done

cd ../

fi
done
#done

# wait for all Schrodinger jobs to finish running before exiting control script
end=`$SCHRODINGER/jobcontrol -list | grep -c 'running\|done\|launched'`
while [ $end -gt 0 ]; do
sleep 30
end=`$SCHRODINGER/jobcontrol -list | grep -c 'running\|done\|launched'`
done

$SCHRODINGER/jobcontrol -list all | grep 'died' > Schrodinger_err.log # save err log to Schrodinger_err.log
