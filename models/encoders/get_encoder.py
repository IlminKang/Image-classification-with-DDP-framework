import torch

def load_encoder(args, device):
    print(f'\t|-Encoder: {args.encoder}')
    if args.encoder.lower() == 'resnet50':
        from models.encoders import resnet

        encoder = resnet.resnet50(num_classes=args.class_num)
        last_dim = class_num

        if args.encoder_weight != '':
            print(f'\t\t|-Loading pretrained resnet50 weights from {args.pretrained_path}...')
            encoder.load_state_dict(torch.load(args.encoder_weight))
        else:
            print(f'\t\t|-Loading scratch resnet50...')

    elif args.encoder.lower() == 'vgg16':
        from models.encoders import vgg
        encoder = vgg.vgg16(in_channels=3, num_classes=args.class_num)
        last_dim = args.class_num
    else:
        raise TypeError(f'Invalid encoder type! : {args.encoder.lower()}')

    return encoder, last_dim


