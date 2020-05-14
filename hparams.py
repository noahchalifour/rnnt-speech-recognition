from tensorboard.plugins.hparams import api as hp

HP_TOKEN_TYPE = hp.HParam('token_type', hp.Discrete(['word-piece', 'character']))
HP_VOCAB_SIZE = hp.HParam('vocab_size', hp.Discrete([2**12]))

# Preprocessing Hparams
HP_MEL_BINS = hp.HParam('mel_bins', hp.Discrete([80]))
HP_FRAME_LENGTH = hp.HParam('frame_length', hp.Discrete([0.025]))
HP_FRAME_STEP = hp.HParam('frame_step', hp.Discrete([0.01]))
HP_HERTZ_LOW = hp.HParam('hertz_low', hp.Discrete([125.0]))
HP_HERTZ_HIGH = hp.HParam('hertz_high', hp.Discrete([7600.0]))
HP_DOWNSAMPLE_FACTOR = hp.HParam('downsample_factor', hp.Discrete([3]))

# Model Hparams
HP_EMBEDDING_SIZE = hp.HParam('embedding_size', hp.Discrete([500]))
HP_ENCODER_LAYERS = hp.HParam('encoder_layers', hp.Discrete([8]))
HP_ENCODER_SIZE = hp.HParam('encoder_size', hp.Discrete([2048]))
HP_PROJECTION_SIZE = hp.HParam('projection_size', hp.Discrete([640]))
HP_TIME_REDUCT_INDEX = hp.HParam('time_reduction_index', hp.Discrete([1]))
HP_TIME_REDUCT_FACTOR = hp.HParam('time_reduction_factor', hp.Discrete([2]))
HP_PRED_NET_LAYERS = hp.HParam('pred_net_layers', hp.Discrete([2]))
HP_PRED_NET_SIZE = hp.HParam('pred_net_size', hp.Discrete([2048]))
HP_JOINT_NET_SIZE = hp.HParam('joint_net_size', hp.Discrete([640]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0]))

# HP_EMBEDDING_SIZE = hp.HParam('embedding_size', hp.Discrete([32]))
# HP_ENCODER_LAYERS = hp.HParam('encoder_layers', hp.Discrete([4]))
# HP_ENCODER_SIZE = hp.HParam('encoder_size', hp.Discrete([20]))
# HP_PROJECTION_SIZE = hp.HParam('projection_size', hp.Discrete([50]))
# HP_TIME_REDUCT_INDEX = hp.HParam('time_reduction_index', hp.Discrete([1]))
# HP_TIME_REDUCT_FACTOR = hp.HParam('time_reduction_factor', hp.Discrete([2]))
# HP_PRED_NET_LAYERS = hp.HParam('pred_net_layers', hp.Discrete([2]))
# HP_PRED_NET_SIZE = hp.HParam('pred_net_size', hp.Discrete([100]))
# HP_JOINT_NET_SIZE = hp.HParam('joint_net_size', hp.Discrete([50]))
# HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2]))

HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-4]))

METRIC_TRAIN_LOSS = 'train_loss'
METRIC_TRAIN_ACCURACY = 'train_accuracy'
METRIC_EVAL_LOSS = 'eval_loss'
METRIC_EVAL_ACCURACY = 'eval_accuracy'
METRIC_EVAL_CER = 'eval_cer'
METRIC_EVAL_WER = 'eval_wer'
METRIC_ACCURACY = 'accuracy'
METRIC_CER = 'cer'
METRIC_WER = 'wer'
