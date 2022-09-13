import torch
from quant_modules import *
from q_transformer import *

import torch
from tqdm import tqdm

from quant_config_deit import extra_config

# # Zero-shot prediction
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def quant_to(model, layername):
    if layername is None:
        for name, mm in model.named_modules():
            if hasattr(mm, "full_precision_flag"):
                setattr(mm, "full_precision_flag", True)
        return model
    full_precision_flag = False
    for name, mm in model.named_modules():
        if hasattr(mm, "full_precision_flag"):
            setattr(mm, "full_precision_flag", full_precision_flag)
        if name==layername+".Qact":
            full_precision_flag = True
    return model

def fp_model(model):
    for n,m in model.named_modules():
        setattr_depend(m,'full_precision_flag',True)
        setattr_depend(m,'running_stat',False)
    return model

def load_clamp_scheme(model, layername, clip_point):
    layer_exists = False
    for name ,mm in model.named_modules():
        # print(name,layername)
        if name==layername:
            setattr(mm, "clip_point", clip_point)
            layer_exists = True
            break
    return layer_exists

feats = 0
def hook(module, input, output):
    global feats
    # print(type(input),type(output))
    feats = {"input":input[0], "output":output}
    return

def inference_all(model, data_loader, hook_layer=None,scale=None, infer_only=False):
    if hook_layer is None:
        hook_layer = 'visual.transformer.resblocks.0.attn.quant_in_proj'
    global feats
    for n,m in model.named_modules():
        if n==hook_layer:
            isconv = False
            hh = m.register_forward_hook(hook)
            weight = getattr(m, "weight")
            bias = getattr(m, "bias")
            if isinstance(m,QuantConv2d):
                isconv = True
                stride, padding, dilation, groups = (m.conv.stride, m.conv.padding,m.conv.dilation, m.conv.groups)
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        total_loss = 0
        for i, (images, target) in enumerate(tqdm(data_loader,desc='Calibration:',disable=False)):
        # for i, (images, target) in enumerate(data_loader):
            images = images.cuda()
            target = target.cuda()
            # predict
            output = model(images)
            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            # calc loss
            if infer_only:
                loss=0
            else:
                if isconv:
                    fpoutput = F.conv2d(feats["input"], weight, bias, stride, padding, dilation, groups)
                else:
                    fpoutput = F.linear(feats["input"],weight,bias)
                loss = criterion(fpoutput,feats['output'])
            total_loss += loss
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    if infer_only:
        print(f"Top-1 accuracy: {top1:.2f}")
    else:
        print(f"Top-1 accuracy: {top1:.2f}, total_loss={total_loss:.4f}, scale={scale}")
        hh.remove()
        # print(model)
    # print(f"Top-5 accuracy: {top5:.2f}")
    return total_loss

def noisy_to(model, layername):
    for name,mm in model.named_modules():
        if name == layername:
            setattr(mm, "use_noise", True)
            break
    return model
def load_noisyScale(model,layername,scale):
    layer_exists = False
    for name,mm in model.named_modules():
        if name == layername:
            setattr(mm, "noiseScale", scale)
            layer_exists = True
    return layer_exists, model
def clean_model(model):
    for name,mm in model.named_modules():
        if hasattr(mm,'use_noise'):
            setattr(mm, "use_noise", False)

def static_to(model, layername):
    for name,mm in model.named_modules():
        if name == layername:
            setattr(mm, "use_static", True)
            break
    return model
def load_static_num(model,layername,num):
    layer_exists = False
    for name,mm in model.named_modules():
        if name == layername:
            setattr(mm, "static_num", num)
            layer_exists = True
    return layer_exists, model
@torch.no_grad()
def easyStatic(model):
    print("start searching for static number!")
    clean_model(model)
    dataVolume = 32
    data_loader = torch.load("calibrationData1024-32x32-ImageNet.pt")[:dataVolume]#[:16]

    clip_point_dict = extra_config['clip-point-vit-l-16-4bit-2nd']
    num_best_dict = {}
    for idx,(layername, abs_max) in enumerate(clip_point_dict.items()):
        model = quant_to(model, layername)
        model = static_to(model, layername)
        layer_exists = load_clamp_scheme(model, layername, clip_point=abs_max)
        lowest_loss = 100
        abs_max_best = 100
        for step in [ii for ii in range(0,25)]+[-ii for ii in range(1,25)]:
            num = step*0.02#abs_max/64/5*(step)
            layer_exists,model = load_static_num(model, layername, num)
            assert layer_exists
            total_loss = inference_all(model, data_loader, hook_layer=layername,scale=num)
            total_loss = total_loss/dataVolume*16
            if total_loss<lowest_loss:
                lowest_loss = total_loss
                num_best = num
            if total_loss<0.0015:
                break
        print(f"Determine best num={num_best:.2f} for this layer={layername}")
        layer_exists,model = load_static_num(model, layername, num_best)
        assert layer_exists
        print("-"*20+f"[{idx}/{len(clip_point_dict)}] End of layername={layername}, num_best={num_best:.2f}"+"-"*20)
        num_best_dict[layername]=num_best
    print("num_best_dict=",num_best_dict)
    return model
@torch.no_grad()
def easyNoisy(model, args=None):
    print("start searching for noisy scale!")
    clean_model(model)
    data_loader = torch.load("calibrationData1024-32x32-ImageNet.pt")
    # data_loader = data_loader[:len(data_loader)//2]

    clip_point_dict = extra_config['clip-point-vit-l-16-4bit-2nd']
    scale_best_dict = {}
    for idx,(layername, abs_max) in enumerate(clip_point_dict.items()):
        model = quant_to(model, layername)
        model = noisy_to(model, layername)
        # model = static_to(model, layername)
        # layer_exists,model = load_static_num(model, layername, num)
        layer_exists = load_clamp_scheme(model, layername, clip_point=abs_max)
        lowest_loss = 100
        abs_max_best = 100
        for step in [ii*0.01 for ii in range(40)]:
        # for step in [ii*0.01 for ii in range(1)]:
            scale = step#abs_max/64/5*(step)
            # print(f"scale={scale:.4f}")
            layer_exists,model = load_noisyScale(model, layername, scale)
            # print(model)
            assert layer_exists
            total_loss = inference_all(model, data_loader, hook_layer=layername,scale=scale)
            # print(model)
            # raise NotImplementedError
            if total_loss<lowest_loss:
                lowest_loss = total_loss
                scale_best = scale
            if total_loss<0.0015:
                break
        # raise NotImplementedError
        print(f"[{idx}/{len(clip_point_dict)}]Determine best scale={scale_best:.4f} for this layer={layername}")
        layer_exists,model = load_noisyScale(model, layername, scale_best)
        assert layer_exists
        print("-"*20+f"[{idx}/{len(clip_point_dict)}] End of layername={layername}, scale_best={scale_best:.4f}"+"-"*20)
        scale_best_dict[layername]=scale_best
    print("scale_best_dict=",scale_best_dict)
    return

# easyQuant text encoder
def easyQuant_txt(model, classnames, templates):
    print("-"*80)
    print("Start easyQuant_txt!")
    import clip
    global feats
    data_loader = torch.load("calibrationData1024-ViT-L-14@336px.pt")[:16]
    criterion = torch.nn.MSELoss()
    clip_point_dict = clamp_scheme['ViT-L-14@336px-FP-txt']
    for idx, (layername, abs_max) in enumerate(clip_point_dict.items()):
        model = quant_to(model, layername)
        for nn,mm in model.named_modules():
            if nn==layername:
                hh = mm.register_forward_hook(hook)
                weight = getattr(mm, "weight")
                bias = getattr(mm, "bias")
        lowest_loss = 99999
        for step in [0.05,0.1,0.3,0.5,1.]:
            total_loss = 0
            abs_max_attemp = abs_max*step
            layer_exists = load_clamp_scheme(model, layername, clip_point=abs_max_attemp)
            assert layer_exists
            # print(model.transformer)
            with torch.no_grad():
                zeroshot_weights = []
                for classname in tqdm(classnames,desc='Preparing zeroshot_weights', disable=False):
                    texts = [template.format(classname) for template in templates] #format with class
                    texts = clip.tokenize(texts).cuda() #tokenize
                    class_embeddings = model.encode_text(texts) #embed with text encoder
                    total_loss += criterion(F.linear(feats["input"],weight,bias),feats['output'])
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
                print(f"total_loss={total_loss}")
                if total_loss<lowest_loss:
                    zeroshot_weights_best = torch.stack(zeroshot_weights, dim=1).cuda()
                    lowest_loss = total_loss
                    abs_max_best = abs_max_attemp

        inference_all(model, data_loader, zeroshot_weights_best, infer_only=True)
        print("-"*10+f"[{idx}/{len(clip_point_dict)}] abs_max_best={abs_max_best}"+"-"*10)
        hh.remove()

# easyQuant image encoder
@torch.no_grad()
def easyQuant(model):
    # from quant_config_vit import extra_config
    data_loader = torch.load("calibrationData1024-32x32-ImageNet.pt")[:]
    clip_point_dict = {}
    for k,v in extra_config['clip-point-deit_base_patch16_224-fp'].items():
        clip_point_dict[k] = max(abs(v[0]),abs(v[1]))
    # clip_point_dict = extra_config['mb_large_fp_clip_point']
    model = fp_model(model)
    print(model)
    # total_loss = inference_all(model, data_loader, hook_layer=None, scale="FP")
    # print("FP model:~62%")
    print("-"*80)
    abs_max_best_dict = {}
    for idx,(layername, abs_max) in enumerate(clip_point_dict.items()):
        print(layername)
        model = quant_to(model, layername)
        lowest_loss = 1000
        abs_max_best = 1000
        for step in ([10]+[ii for ii in range(1,10)]+[ii for ii in range(11,20)]):
            abs_max_attemp = abs_max*(step/10.)
            layer_exists = load_clamp_scheme(model, layername, clip_point=abs_max_attemp)
            assert layer_exists
            # if idx<70:
            #     lowest_loss=0
            #     abs_max_best = abs_max_attemp
            #     break
            total_loss = inference_all(model, data_loader, hook_layer=layername, scale=f"{abs_max_attemp:.4f}")
            if total_loss<lowest_loss:
                lowest_loss = total_loss
                abs_max_best = abs_max_attemp
            if total_loss<0.05:
                break
        print(f"[{idx}/{len(clip_point_dict)}]For this layer={layername}, lowest_loss={lowest_loss:.4f}, determine best abs_max={abs_max_best:.4f}")
        layer_exists = load_clamp_scheme(model, layername, clip_point=abs_max_best)
        assert layer_exists
        print("-"*20+f"[{idx}/{len(clip_point_dict)}] End of layername={layername}, abs_max_best={abs_max_best:.4f}"+"-"*20)
        abs_max_best_dict[layername]=abs_max_best
    print(abs_max_best_dict)
    return model

def easyQuant_calibration_data(data_loader):
    with torch.no_grad():
        total_im = []
        for i, (images, target) in enumerate(tqdm(data_loader,desc='Evaluating:')):
            print(i,target)
            total_im += [(images, target)]
            if len(total_im)>=32:
                torch.save(total_im, "calibrationData1024-32x32-ImageNet.pt")
                raise NotImplementedError



def get_module_by_name(model, module_name):
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        if hasattr(model, name):
            model = getattr(model, name)
        else:
            return None, None
    if hasattr(model, name_list[-1]):
        leaf_module = getattr(model, name_list[-1])
        return model, leaf_module
    else:
        return None, None

def update_module(model, module_name, new_module):
    super_module, leaf_module = get_module_by_name(model, module_name)
    setattr(super_module, module_name.split('.')[-1], new_module)

def hawq_quant(model,args=None,model_name=None):
    print("-"*80)
    if model_name=="vit":
        for name,module in model.named_modules():
            if "head" in name:
                continue
            if isinstance(module, torch.nn.Linear) or name.endswith("out_proj"):
                qlinear = QuantLinear()
                qlinear.set_param(module)
                new_module = qlinear
                # print(f"linear={name}")
            # elif isinstance(module, torch.nn.MultiheadAttention):
            #     qmha = QuantMultiheadAttention(equal_qkv=True)
            #     # print(module.__dict__)
            #     qmha.set_param(module)
            #     # raise NotImplementedError
            #     new_module = qmha
            #     print(f"MHA={name}")
            else:
                continue
            update_module(model, name, new_module=new_module)
    elif model_name=="deit":
        for name,module in model.named_modules():
            # if name not in extra_config['clip_point_8bit'].keys():
            #     continue
            if not name.startswith("blocks."):
                continue
            # if isinstance(module, torch.nn.Conv2d):
            #     qconv2d = QuantConv2d()
            #     qconv2d.set_param(module)
            #     new_module = qconv2d
            #     print(f"conv2d,{name}")
            elif isinstance(module, torch.nn.Linear):
                qlinear = QuantLinear()
                qlinear.set_param(module)
                new_module = qlinear
                # print(f"linear,{name}")
            else:
                continue
            update_module(model, name, new_module=new_module)
    else:
        raise NotImplementedError
    return model
def setattr_depend(m,name,val,verbose=False):
    if hasattr(m,name):
        setattr(m,name,val)
        if verbose:
            print(name)
        return True
    else:
        return False
