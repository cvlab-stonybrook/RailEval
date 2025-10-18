import torch

from config import BackboneConfig
from head import DecoderConfig, HeadConfig
from segmentator import Segmentator

def main():
    img_h = 512
    img_w = 512
    img_d = 3
    n_classes = 1       # ALWAYS SET TO 1 IF BINARY CLASSIFICATION
    batch_size = 2

    bbone_cfg = BackboneConfig(name='mobilenetv2',
                                    os=8,
                                    h=img_h,
                                    w=img_w,
                                    d=img_d,
                                    #dropout=0.02,
                                    dropout=0.0,
                                    bn_d=0.05,
                                    extra={'width_mult': 1.0, 'shallow_feats': True})

    decoder_cfg = DecoderConfig(name='aspp_progressive',
                                     #dropout=0.02,
                                     dropout=0.0,
                                     bn_d=0.05,
                                     extra={'aspp_channels': 32, 'last_channels': 16})

    head_cfg = HeadConfig(n_class=n_classes,
                               #dropout=0.1,
                               dropout=0.0,
                               weights=torch.ones(n_classes, dtype=torch.float))

    # concatenate the encoder and the head
    with torch.no_grad():
        model = Segmentator(bbone_cfg,
                                decoder_cfg,
                                head_cfg,
                                None)

        model.cuda()
        data = torch.zeros((batch_size, img_d, img_h, img_w)).cuda()
        output = model(data)
        print(output.shape, output.min(), output.max())


if __name__ == "__main__":
    main()
