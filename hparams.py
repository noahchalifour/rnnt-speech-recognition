from tensorboard.plugins.hparams import api as hp

# Preprocessing Hparams
HP_MEL_BINS = hp.HParam('mel_bins', hp.Discrete([80]))
HP_FRAME_LENGTH = hp.HParam('frame_length', hp.Discrete([0.025]))
HP_FRAME_STEP = hp.HParam('frame_step', hp.Discrete([0.01]))
HP_HERTZ_LOW = hp.HParam('hertz_low', hp.Discrete([125.0]))
HP_HERTZ_HIGH = hp.HParam('hertz_high', hp.Discrete([7600.0]))

# Model Hparams
# HP_EMBEDDING_SIZE = hp.HParam('embedding_size', hp.Discrete([64]))
# HP_ENCODER_LAYERS = hp.HParam('encoder_layers', hp.Discrete([8]))
# HP_ENCODER_SIZE = hp.HParam('encoder_size', hp.Discrete([2048]))
# HP_TIME_REDUCT_INDEX = hp.HParam('time_reduction_index', hp.Discrete([1]))
# HP_TIME_REDUCT_FACTOR = hp.HParam('time_reduction_factor', hp.Discrete([2]))
# HP_PRED_NET_LAYERS = hp.HParam('pred_net_layers', hp.Discrete([2]))
# HP_JOINT_NET_SIZE = hp.HParam('joint_net_size', hp.Discrete([640]))
# HP_SOFTMAX_SIZE = hp.HParam('softmax_size', hp.Discrete([4096]))

HP_EMBEDDING_SIZE = hp.HParam('embedding_size', hp.Discrete([32]))
HP_ENCODER_LAYERS = hp.HParam('encoder_layers', hp.Discrete([2]))
HP_ENCODER_SIZE = hp.HParam('encoder_size', hp.Discrete([100]))
HP_TIME_REDUCT_INDEX = hp.HParam('time_reduction_index', hp.Discrete([1]))
HP_TIME_REDUCT_FACTOR = hp.HParam('time_reduction_factor', hp.Discrete([2]))
HP_PRED_NET_LAYERS = hp.HParam('pred_net_layers', hp.Discrete([2]))
HP_JOINT_NET_SIZE = hp.HParam('joint_net_size', hp.Discrete([100]))
HP_SOFTMAX_SIZE = hp.HParam('softmax_size', hp.Discrete([200]))

HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-4]))

METRIC_LOSS = 'loss'
METRIC_ACCURACY = 'accuracy'