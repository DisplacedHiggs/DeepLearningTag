#!/bin/bash

tmpfile=tmp.tmp
ls -l array_ZH*.npy | awk '{print $9}' > $tmpfile

ofile="test.out"

if [ -e $ofile ]
then
	rm $ofile
fi

touch $ofile

while read file
do
	tofile=${file}.tmp
	base=`echo $file | awk '{split($1,array,".npy");split(array[1],array2,"array_"); print array2[2]}'`
	echo $base
	python testCNN.py $file jetarray_${base}.npy roc_${base}.png > $tofile
	br=`grep "background rejection" $tofile | awk '{print $3}'`
	se=`grep "signal effic" $tofile | awk '{print $3}'`
	echo $file $br $se >> $ofile
	rm $tofile
done < $tmpfile
rm $tmpfile