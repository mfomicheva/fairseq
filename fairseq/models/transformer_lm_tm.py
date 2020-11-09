from fairseq.models import FairseqModel

from fairseq import utils


class TransformerLanguageModelTranslationModel(FairseqModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # fmt: on
