import os
import re
import torch
from PIL import Image, ImageEnhance

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
    
#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


#----------------------------------------------------------------------------
# Image enhance
def img_enhance(img, mode='celeba'):
    if mode=='celeba':
        enh_con = ImageEnhance.Contrast(img)
        con_factor = 0.9
        enhance_image = enh_con.enhance(con_factor)

        enh_bri = ImageEnhance.Brightness(enhance_image)
        bri_factor = 1.05
        enhance_image = enh_bri.enhance(bri_factor)

        enh_col = ImageEnhance.Color(enhance_image)
        color_factor = 0.99
        return enh_col.enhance(color_factor)
    else:
        enh_bri = ImageEnhance.Brightness(img)
        factor = 1.06
        return enh_bri.enhance(factor)

# Sample saver
def save_samples(images, batch_seeds, out_dir):
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for seed, image_np in zip(batch_seeds, images_np):
        image_dir = os.path.join(out_dir, f'{seed - seed % 1000:06d}')
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f'{seed:06d}.png')
        
        if image_np.shape[2] == 1:
            Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
        else:
            Image.fromarray(image_np, 'RGB').save(image_path)
        
        # if image_np.shape[2] == 1:
        #     img_enhance(Image.fromarray(image_np[:, :, 0], 'L')).save(image_path)
        # else:
        #     img_enhance(Image.fromarray(image_np, 'RGB')).save(image_path)
