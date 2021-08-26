import torch, os
from wav2vec2 import feature_extractor, featureprojection, encoder, Wav2Vec2GumbelVectorQuantizer
self.feature_extractor = feature_extractor
self.feature_projection = featureprojection
self.encoder = encoder
self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())

### pretrain###
self.quantizer = Wav2Vec2GumbelVectorQuantizer()
self.project_q = nn.Linear(codevector_dim, proj_codevector_dim)  # from codebook to compare
self.project_hid = nn.Linear(context_size, proj_codevector_dim)  # from c to compare


def pretrain_save(ckpt_dir,var_list,epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    save_pt = {i:globals()[f'{i}'] for i in var_list}
    torch.save(save_pt,
               "%s/pretrain_epoch%d.pth" % (ckpt_dir, epoch))



def pretrain_load(ckpt_dir, var_list):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netG_a2b.load_state_dict(dict_model['netG_a2b'])
    netG_b2a.load_state_dict(dict_model['netG_b2a'])
    netD_a.load_state_dict(dict_model['netD_a'])
    netD_b.load_state_dict(dict_model['netD_b'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])