aspect="BP CC MF"
for i in $aspect
do
    echo $i
    time python run_blast.py ./testdb/test_sequence.fasta /mnt/f/ATGO/New_Benchmark_history/model/Database/$i/sequence.fasta -n 32 -o ./out/$i_hit.csv
done