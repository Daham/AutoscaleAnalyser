# convert sysstat outputs and RIF (4-column CSV) outputs to single-column format
# WARNING: replaces original data files


# cpu
tail cpu -n+4 | sed -E 's/(\S+\s+)+/100-/g' | head -n-1 | bc | sed -E "s/^\./0./g"

# cpu, recursive
for i in `find -name "cpu"`; do
	tail $i -n+4 | sed -E 's/(\S+\s+)+/100-/g' | head -n-1 | bc | sed -E "s/^\./0./g" > $i.csv
	mv $i.csv $i
done 


# mem
# NOTE: change {3} -> {4} if input uses AM/PM time format
tail mem -n+4 | sed -E 's/^(\S+\s+){3}//g' | sed -E 's/(\s+\S+)+$//g' | head -n-1

# mem, recursive
# NOTE: change {3} -> {4} if input uses AM/PM time format
for i in `find -name "mem"`; do
	tail $i -n+4 | sed -E 's/^(\S+\s+){3}//g' | sed -E 's/(\s+\S+)+$//g' | head -n-1 > $i.csv
	mv $i.csv $i
done 


# rif
cat rif | sed -E 's/\w+,\w+,//g' | sed -E 's/,-?\w+//g'

# rif, recursive
for i in `find -name "rif"`; do
	cat $i | sed -E 's/\w+,\w+,//g' | sed -E 's/,-?\w+//g' > $i.csv
	mv $i.csv $i
done 

