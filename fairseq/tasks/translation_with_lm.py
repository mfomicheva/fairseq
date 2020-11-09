from fairseq.tasks import FairseqTask


class TranslationLanguageModelTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--max-target-positions', type=int, metavar='N',
                            help='max number of tokens in the target sequence')
