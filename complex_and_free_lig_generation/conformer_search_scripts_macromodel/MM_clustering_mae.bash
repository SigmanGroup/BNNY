#!/bin/bash
#SBATCH --partition=notchpeak-shared-short
#SBATCH --account=notchpeak-shared-short
#SBATCH --time=8:00:00
#SBATCH --ntasks=4
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N

# How to use on the CHPC:
# run this script in the same directory as you ran “prep_MM_mae.sh” 
#      run "bash MM_clustering_mae.bash"
# this script will cluster ligands from a conformer search but output is mae's so that ligands can be edited 

module load schrodinger
WORKDIR=$PWD

for PDB in $WORKDIR/*/;do #for every folder in working directory

cd $PDB	#enter the folder

for file in $PDB/*.log;do #for every .log file in that folder
filename=$(basename $file) #save the file name as the variable we'll use
nopathnoext=${filename%.*}

if [[ -f ${nopathnoext}.log ]];then #if the log file exists, continue 
lines=`grep -A 1 "Final report; processing .tmp file:" ${nopathnoext}.log` #look for the final report in the log file, and save it to the lines variable
echo ${nopathnoext} has ${lines//[^0-9]/} conformers #use that information to print how many conformers that compound has
number=${lines//[^0-9]/} #store the number of conformers (the number in that line)

if [ $number -le 20 ] #if there <= 20 total conformers, move/rename the conformational search file output to the acid output folder
then 
echo 'converting '${nopathnoext}'.maegz to .mae'
$SCHRODINGER/utilities/structconvert ${nopathnoext}_out.maegz ${nopathnoext}_conf.mae
mv ${nopathnoext}_conf.mae ../output/${nopathnoext}_conf.mae

else #if there are more than 20 conformers, move/rename to the clustering folder
$SCHRODINGER/run conformer_cluster.py -j ${nopathnoext}_cluster -n 0  -r -rep -WAIT ${nopathnoext}_out.maegz #if you want a certain number of clusters, use "-n 5", 0 uses the Kelley Penalty minimum 

echo 'converting '${nopathnoext}'.maegz to .mae'
$SCHRODINGER/utilities/structconvert ${nopathnoext}_cluster_ligand1_representatives.maegz ${nopathnoext}_clust.mae
$SCHRODINGER/utilities/structconvert ${nopathnoext}_out.maegz ${nopathnoext}_conf.mae
mv ${nopathnoext}_clust.mae ../output/${nopathnoext}_clust.mae
mv ${nopathnoext}_conf.mae ../output/clustered/${nopathnoext}_conf.mae

fi

fi
done

cd $WORKDIR

done
