from .sasnet import build

# build the P2PNet model
# set training to 'True' during training
def build_model(args, training):
    return build(args, training)