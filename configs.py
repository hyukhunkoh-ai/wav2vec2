class Wav2Vec2Config():
    r"""
    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 32):
            - stt용 vocab size
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            - emb layer
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            - number of transformer blocks
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            - muti head num
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            - 중간 FFN 사이즈
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            - 활성함수 for transformers
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            - all linear layer dropout
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            - in transformer, attention to linear dropout
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            - The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            - The epsilon used by the layer normalization layers when devide the target = 1e-5
        feat_extract_norm (:obj:`str`, `optional`, defaults to :obj:`"group"`):
            - choose group // only one group layer normalization for first layer in feature extractor
        feat_extract_activation (:obj:`str, `optional`, defaults to :obj:`"gelu"`):
            - feature extractor use gelu as a activation
            extractor. If string, :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        feat_quantizer_dropout (obj:`float`, `optional`, defaults to 0.0):
            - The dropout probabilitiy for quantized feature extractor states
            - 양자화할 때 사용, 0씀
        conv_dim (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(512, 512, 512, 512, 512, 512, 512)`):
            - feature extractor cnn channel
        conv_stride (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(5, 2, 2, 2, 2, 2, 2)`):
            - feature extractor cnn strides
        conv_kernel (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(10, 3, 3, 3, 3, 3, 3)`):
            - feature extractor cnn kernel size
        conv_bias (:obj:`bool`, `optional`, defaults to :obj:`False`):
            - false, activation 사용할 것이라서
        num_conv_pos_embeddings (:obj:`int`, `optional`, defaults to 128):
            - transformer convolutional positional encdoing kernel size = 128
        num_conv_pos_embedding_groups (:obj:`int`, `optional`, defaults to 16):
            - transformer convolutional positional encdoing group
        do_stable_layer_norm (:obj:`bool`, `optional`, defaults to :obj:`False`):
            - whether to use LN before attention or not
        apply_spec_augment (:obj:`bool`, `optional`, defaults to :obj:`True`):
            - whether to use specaugment(오디오 augmentation 기법)
        mask_time_prob (:obj:`float`, `optional`, defaults to 0.05):
            - masked probability = p
            - ``mask_time_prob * sequence_length // mask_time_length`` feature vectors will be masked along the time axis.
        mask_time_length (:obj:`int`, `optional`, defaults to 10):
            - mask length = M = 10
        mask_feature_prob (:obj:`float`, `optional`, defaults to 0.0):
            - mask length for feature axis. we will not use this
        mask_feature_length (:obj:`int`, `optional`, defaults to 10):
            - we won't use
        num_codevectors_per_group (:obj:`int`, `optional`, defaults to 320):
            - number of entries = V = 320
        num_codevector_groups (:obj:`int`, `optional`, defaults to 2):
            - product quantize group = G = 2
        contrastive_logits_temperature (:obj:`float`, `optional`, defaults to 0.1):
            - The temperature `kappa` in the contrastive loss. = t = 0.1
        num_negatives (:obj:`int`, `optional`, defaults to 100):
            - distractors for contrastive loss = K = 100
        codevector_dim (:obj:`int`, `optional`, defaults to 256):
            - Dimensionality of the quantized feature vectors = d = 256 -> 2*320*128
        proj_codevector_dim (:obj:`int`, `optional`, defaults to 256):
            - if we compare q and c, make dim this = 256
        diversity_loss_weight (:obj:`int`, `optional`, defaults to 0.1):
            - diversity loss weights = alpha = 0.1
        ctc_loss_reduction (:obj:`str`, `optional`, defaults to :obj:`"sum"`):
            - sum // loss 계산시 합칠 때
            - option : mean or sum
        ctc_zero_infinity (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to zero infinite losses and the associated gradients of ``torch.nn.CTCLoss``. Infinite losses
            mainly occur when the inputs are too short to be aligned to the targets. Only relevant when training an
            instance of :class:`~transformers.Wav2Vec2ForCTC`.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            - If True, use gradient checkpointing to save memory at the expense of slower backward pass.

    Example::

        from transformers import Wav2Vec2Model, Wav2Vec2Config

        # Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
        configuration = Wav2Vec2Config()

        # Initializing a model from the facebook/wav2vec2-base-960h style configuration
        model = Wav2Vec2Model(configuration)

        # Accessing the model configuration
        configuration = model.config
    """
    model_type = "wav2vec2"

    def __init__(
        self,
        vocab_size=32,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        feat_proj_dropout=0.1,
        feat_quantizer_dropout=0.0,
        final_dropout=0.1,
        layerdrop=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        feat_extract_norm="group",
        feat_extract_activation="gelu",
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=False,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        do_stable_layer_norm=False,
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        num_codevectors_per_group=320,
        num_codevector_groups=2,
        contrastive_logits_temperature=0.1,
        num_negatives=100,
        codevector_dim=256,
        proj_codevector_dim=256,
        diversity_loss_weight=0.1,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        gradient_checkpointing=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = list(conv_dim)
        self.conv_stride = list(conv_stride)
        self.conv_kernel = list(conv_kernel)
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_feat_extract_layers = len(self.conv_dim)
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.feat_proj_dropout = feat_proj_dropout
        self.final_dropout = final_dropout
        self.layerdrop = layerdrop
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.do_stable_layer_norm = do_stable_layer_norm
        self.gradient_checkpointing = gradient_checkpointing

        if (
            (len(self.conv_stride) != self.num_feat_extract_layers)
            or (len(self.conv_kernel) != self.num_feat_extract_layers)
            or (len(self.conv_dim) != self.num_feat_extract_layers)
        ):
            raise ValueError(
                "Configuration for convolutional layers is incorrect."
                "It is required that `len(config.conv_dim)` == `len(config.conv_stride)` == `len(config.conv_kernel)`,"
                f"but is `len(config.conv_dim) = {len(self.conv_dim)}`, `len(config.conv_stride)"
                f"= {len(self.conv_stride)}`, `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )

        # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length

        # parameters for pretraining with codevector quantized representations
        self.num_codevectors_per_group = num_codevectors_per_group
        self.num_codevector_groups = num_codevector_groups
        self.contrastive_logits_temperature = contrastive_logits_temperature
        self.feat_quantizer_dropout = feat_quantizer_dropout
        self.num_negatives = num_negatives
        self.codevector_dim = codevector_dim
        self.proj_codevector_dim = proj_codevector_dim
        self.diversity_loss_weight = diversity_loss_weight

        # ctc loss
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity