import math, csv


def divide_batches(fin, n_batches, size):
    print("Divide", fin, "into", n_batches, "batches.")
    tsvin = open(fin + ".tsv", mode='rt')
    tsvin_writer = csv.reader(tsvin, delimiter='\t')
    line_per_file = math.ceil(float(size) / n_batches)
    n_out = 1
    n_line = 1
    csvout = open(fin + "_" + str(n_out) + ".csv", "wt")
    csvout_writer = csv.writer(csvout, delimiter=',', lineterminator='\n')
    for row in tsvin_writer:
        print(row)
        if n_line > line_per_file:
            n_out += 1
            csvout.close()
            csvout = open(fin + "_" + str(n_out) + ".csv", "wt")
            csvout_writer = csv.writer(csvout, delimiter=',', lineterminator='\n')
            n_line = 1
        csvout_writer.writerow(row)
        n_line += 1

    csvout.close()
    tsvin.close()

divide_batches("data/Author-Details", 10, 15000)