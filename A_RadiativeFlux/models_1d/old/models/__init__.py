import importlib

def get_model(x3d_mean, x3d_std, x2d_mean, x2d_std, args):
  model_classes = {
    'mlp': 'mlp.MlpIg',
    'clipped_mlp': 'mlp.ClippedMlpIg',
    'fast_mlp': 'mlp.FastMlpIg',
    'toa_mlp': 'mlp.ToAMlpIg',
    'mlp_maxpool': 'mlp.MlpIgMaxPool',
    'mlp_sw': 'mlp.MlpSharedWeights',
    'ensambled_mlp': 'mlp.EnsambledMlpIg',
    'subfeats_mlp': 'mlp.FastMlpIgSubFeats',
    'ensamble_of_fast_mlp': 'mlp.EnsambleOfFastMlpIg',
    'unet': 'cnn.CnnIg',
    'unet2': 'cnn.CnnIg2',
    'clipped_unet2': 'cnn.ClippedCnnIg2',
    'lstm': 'rnn.RnnIg',
    'new_lstm': 'rnn.NewRnnIg',
    'new_lstm_sw': 'rnn.NewRnnIgSharedWeights',
    'clipped_new_lstm': 'rnn.ClippedNewRnnIg',
    'fast_lstm': 'rnn.FastRnnIg',
    'vit': 'vit.ViT',
    'clipped_vit': 'vit.ClippedViT',
    'rf': 'rf.RandomForest'
  }

  model_class_path = model_classes.get(args.model)

  if model_class_path is None:
    raise NotImplementedError(f'Model {args.model} not implemented')

  module_name, class_name = model_class_path.rsplit('.', 1)

  try:
    module = importlib.import_module(f".{module_name}", package=__name__)
  except ModuleNotFoundError:
    raise ImportError(f"Could not import module {module_name}")

  try:
    model_class = getattr(module, class_name)
  except AttributeError:
    raise ImportError(f"Module '{module_name}' has no class '{class_name}'")

  return model_class(x3d_mean, x3d_std, x2d_mean, x2d_std, args)
