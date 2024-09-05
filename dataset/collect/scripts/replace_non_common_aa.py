from absl import app, flags
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

translation = str.maketrans({
    'U': 'C', # 5jsk
    'O': 'K',
    'X': 'A', # 6st5
    'B': 'A',
    'Z': 'A', # such as 4cpa
    'J': 'A'
})


def define_arguments():
    flags.DEFINE_string("input_fasta", None, required=True, help="input fasta file")
    flags.DEFINE_string("output_fasta", None, required=True, help="output fasta file")
    
def main(argv):
    
    FLAGS = flags.FLAGS
    input_fasta = FLAGS.input_fasta
    output_fasta = FLAGS.output_fasta
    
    records = SeqIO.parse(input_fasta, "fasta")
    new_records = []
    for r in records:
        r: SeqRecord
        new_records.append(SeqRecord(
            seq=Seq(str(r.seq).translate(translation)),
            id=r.id,
            description=""
        ))
    SeqIO.write(new_records, output_fasta, "fasta")


if __name__ == "__main__":
    define_arguments()
    app.run(main)