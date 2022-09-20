import torch
from quant_modules import *
from q_transformer import *

import torch
from tqdm import tqdm
import time
from timm.models.bit_config import bit_config_dict as extra_config

def init_queue(length,val=-1):
    return [val]*length
def pop_put_decide(qq,val):
    qq.pop(0)
    qq += [val]
    find_lowest = True
    for elem in qq[1:]:
        if elem <qq[0]:
            find_lowest = False
    return find_lowest

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
        if name==layername:
            full_precision_flag = True
    return model

def fp_model(model):
    for n,m in model.named_modules():
        setattr_depend(m,'full_precision_flag',True)
        setattr_depend(m,'running_stat',True)
    return model

def load_clamp_scheme(model, layername, clip_point):
    layer_exists = False
    for name ,mm in model.named_modules():
        if name==layername:
            setattr(mm, 'running_stat', False)
            if type(clip_point) is tuple:
                x_min_temp = getattr(mm,"x_min")
                setattr(mm, "x_min",torch.tensor(clip_point[0]).type_as(x_min_temp))
                setattr(mm, "x_max",torch.tensor(clip_point[1]).type_as(x_min_temp))
            else:
                setattr(mm, "clip_point", clip_point)
            layer_exists = True
            break
    return layer_exists

feats = {}
def hook(module, input, output):
    global feats
    # print(type(input),type(output))
    feats[module.ownname] = {"input":input, "output":output}
    return

def inference_all(model, data_loader, hook_layer=None,scale=None, infer_only=False, returnAcc=False,desc="Calibration"):
    if hook_layer is None:
        hook_layer = 'visual.transformer.resblocks.0.attn.quant_in_proj'
    global feats
    hh = []

    new_loader = []
    sub_batch = 16
    for ii in range(len(data_loader)//sub_batch):
        im,tar = ([],[])
        for jj in range(sub_batch):
            (im0,tar0) = data_loader[sub_batch*ii+jj]
            im += [im0]
            tar += [tar0]
        im = torch.cat(im,dim=0)
        tar = torch.cat(tar,dim=0)
        new_loader += [(im,tar)]
    data_loader = new_loader

    if hook_layer.endswith('linear_tokens'):
        fpact_name,fplinear_name = (".norm1",'linear_tokens')
    elif hook_layer.endswith('.fc1'):
        fpact_name,fplinear_name = (".norm2",'mlp_channels.')
    elif hook_layer.endswith('.fc2'):
        fpact_name,fplinear_name = (".act",'mlp_channels.')
    else:
        raise NotImplementedError
    for n,m in model.named_modules():
        if n.endswith(fpact_name) and n.startswith(hook_layer.split(fplinear_name)[0]):
            _ = m.register_forward_hook(hook) # need output: fp input without noise
            hh += [_]
        if n==hook_layer:
            _ = m.register_forward_hook(hook) # need output: quant(input + noise) @ w +b
            hh += [_]
            weight = getattr(m, "weight")
            bias = getattr(m, "bias")

    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        total_loss = 0
        for i, (images, target) in enumerate(tqdm(data_loader,desc=desc,disable=False)):
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
                if hook_layer.endswith('linear_tokens'):
                    fpinput = feats[hook_layer.replace("linear_tokens",'norm1')]['output'].transpose(1, 2)
                elif hook_layer.endswith('.fc1'):
                    fpinput = feats[hook_layer.replace("mlp_channels.fc1",'norm2')]['output']
                elif hook_layer.endswith('.fc2'):
                    fpinput = feats[hook_layer.replace("fc2",'act')]['output']
                else:
                    raise NotImplementedError
                fpoutput = F.linear(fpinput,weight,bias)
                loss = criterion(fpoutput,feats[hook_layer]["output"][0])
            total_loss += loss
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    if infer_only:
        print(f"Top-1 accuracy: {top1:.2f}")
    else:
        print(f"Top-1 accuracy: {top1:.2f}, total_loss={total_loss:.4f}, scale={scale}")
        for _ in hh:
            _.remove()
        feats={}
        # print(model)
    # print(f"Top-5 accuracy: {top5:.2f}")
    if returnAcc:
        return total_loss,top1
    else:
        return total_loss

def noisy_to(model, layername):
    raise NotImplementedError
    for name,mm in model.named_modules():
        if hasattr(mm,'use_noise'):
            setattr(mm,'use_noise',True)
        if hasattr(mm,'use_noise_token'):
            setattr(mm,'use_noise_token',True)
        if hasattr(mm,'use_noise_channel'):
            setattr(mm,'use_noise_channel',True)
        if name == layername:
            break
    return model
def load_noisyScale(model,layername,scale):
    layer_exists = False
    if layername.endswith("linear_tokens"):
        for name,mm in model.named_modules():
            if name==layername.split('.linear_tokens')[0]:
                setattr(mm,'noiseScale_token',torch.tensor([scale]).cuda())
                setattr(mm,'use_noise_token',True)
                layer_exists = True
                break
    if layername.endswith("fc1"):
        for name,mm in model.named_modules():
            if name==layername.split('.mlp_channels')[0]:
                setattr(mm,'noiseScale_channel',torch.tensor([scale]).cuda())
                setattr(mm,'use_noise_channel',True)
                layer_exists = True
                break
    if layername.endswith("fc2"):
        for name,mm in model.named_modules():
            if name==layername.split('.fc2')[0]:
                setattr(mm,'noiseScale',torch.tensor([scale]).cuda())
                setattr(mm,'use_noise',True)
                layer_exists = True
                break
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
    if layername.endswith("linear_tokens"):
        for name,mm in model.named_modules():
            if name==layername.split('.linear_tokens')[0]:
                setattr(mm,'static_num_token',num)
                layer_exists = True
                break
    if layername.endswith("fc1"):
        for name,mm in model.named_modules():
            if name==layername.split('.mlp_channels')[0]:
                setattr(mm,'static_num_channel',num)
                layer_exists = True
                break
    if layername.endswith("fc2"):
        for name,mm in model.named_modules():
            if name==layername.split('.fc2')[0]:
                setattr(mm,'static_num',num)
                layer_exists = True
                break
    return layer_exists, model
@torch.no_grad()
def easyStatic(model, args=None):
    model = fp_model(model) # full_precision_flag, running_stat
    print("start searching for static number!")
    data_loader = torch.load("calibrationData1024-32x32-ImageNet.pt")

    focus_on_names = []
    for name,mm in model.named_modules():
        if name.endswith('linear_tokens') or name.endswith('fc1') or name.endswith('fc2'):
            focus_on_names += [name]

    clip_point_dict = extra_config['q_resmlp_a6w8_clip_point_fp_and_easyQuant']
    noise_config = {}
    # noise_config = extra_config['q_resmlp_a6w8_noise_with_clip']

    scale_best_dict = {}
    time_past = 0
    for idx, layername in enumerate(focus_on_names):
        model = quant_to(model, layername)

        # if noise_config!={}:
        #     layer_exists,model=load_noisyScale(model,layername,noise_config.get(layername))

        if layername.endswith('linear_tokens'):
            clip_layer = layername.replace("linear_tokens","quant_norm1")
        elif layername.endswith('fc1'):
            clip_layer = layername.replace("mlp_channels.fc1","quant_norm2")
        elif layername.endswith('fc2'):
            clip_layer = layername.replace("fc2","quant_act")
        # (xmin,xmax) = clip_point_dict.get(clip_layer)
        # if (abs(xmax-xmin))<100:
        #     continue
        # else:
        #     print(layername,">100",xmin,xmax)
        #     step_scale = (xmax-xmin)/63/100
        # continue

        lowest_loss = 100
        abs_max_best = 100
        top1 = 0
        time_left = 0
        search_space = [ii*0.01 for ii in range(50)] + [-ii*0.01 for ii in range(1,50)]
        # search_space = [ii*step_scale for ii in range(100)]
        for iistep, step in enumerate(search_space):
            scale = step#abs_max/64/5*(step)
            layer_exists,model = load_static_num(model, layername, scale)
            assert layer_exists
            t0=time.time()
            total_loss, top1 = inference_all(model, data_loader, hook_layer=layername,scale=scale,
                                    desc=f"[{iistep}/{len(search_space)}][{time_left/60:.1f} min]",
                                    returnAcc=True)
            duration = time.time()-t0
            time_past += duration
            time_left = duration*(len(focus_on_names)-idx-1)*len(search_space)+duration*(len(search_space)-iistep-1)

            if total_loss<lowest_loss:
                lowest_loss = total_loss
                scale_best = scale
                top1_best = top1
            if total_loss<0.0015:
                break

        layer_exists,model = load_static_num(model, layername, scale_best)
        assert layer_exists
        print("-"*20+f"[{idx}/{len(focus_on_names)}] layername={layername}, scale={scale_best:.4f}, " +
                f"loss={lowest_loss:.4f}, top1={top1_best:.2f}"+"-"*20)
        scale_best_dict[layername]=scale_best
    print("scale_best_dict=",scale_best_dict)
    return

import json
@torch.no_grad()
def easyNoisy(model, args=None):
    print("start searching for noisy scale!")
    # model = fp_model(model) # full_precision_flag, running_stat
    data_loader = torch.load("calibrationData1024-32x32-ImageNet.pt")

    focus_on_names = []
    for name,mm in model.named_modules():
        if name.endswith('linear_tokens') or name.endswith('fc1') or name.endswith('fc2'):
            focus_on_names += [name]

    clip_point_dict = extra_config['q_resmlp_a6w8_clip_point_fp_and_easyQuant']
    noise_config = {}
    static_config = extra_config['q_resmlp_a6w8_static_with_clip']

    scale_best_dict = {}
    time_past = 0
    for idx, layername in enumerate(focus_on_names):
        model = quant_to(model, layername)
        layer_exists,model = load_static_num(model, layername, static_config.get(layername))
        # noise_config = extra_config['q_resmlp_a6w8_noise_with_clip']
        if noise_config!={}:
            layer_exists,model=load_noisyScale(model,layername,noise_config.get(layername))

        #  calculate noise based on clip range
        if layername.endswith('linear_tokens'):
            clip_layer = layername.replace("linear_tokens","quant_norm1")
        elif layername.endswith('fc1'):
            clip_layer = layername.replace("mlp_channels.fc1","quant_norm2")
        elif layername.endswith('fc2'):
            clip_layer = layername.replace("fc2","quant_act")
        (xmin,xmax) = clip_point_dict.get(clip_layer)
        step_scale = (xmax-xmin)/63/100
        # if (abs(xmax-xmin))<100:
        #     continue
        # else:
        #     print(layername,">100",xmin,xmax)
        #     step_scale = (xmax-xmin)/63/100
        # continue

        lowest_loss = 100
        abs_max_best = 100
        top1 = 0
        time_left = 0
        with open(f"./LOGs/LOG-easyNoise-{idx:03d}-{layername}-model.log",'w') as fp:
            json.dump(str(model),fp)

        # search_space = [ii*0.02 for ii in range(200)]
        # search_space = [ii*0.02 for ii in range(1)]
        search_space = [ii*step_scale for ii in range(100)]
        for iistep, step in enumerate(search_space):
            scale = step#abs_max/64/5*(step)
            layer_exists,model = load_noisyScale(model, layername, scale)
            assert layer_exists
            t0=time.time()
            total_loss, top1 = inference_all(model, data_loader, hook_layer=layername,scale=scale,
                        desc=f"[{iistep}/{len(search_space)}][{time_left/60:.1f} min]", returnAcc=True)
            duration = time.time()-t0
            time_past += duration
            time_left = duration*(len(focus_on_names)-idx-1)*len(search_space)+duration*(len(search_space)-iistep-1)

            if total_loss<lowest_loss:
                lowest_loss = total_loss
                scale_best = scale
                top1_best = top1
            # if total_loss<0.0015:
            #     break

        layer_exists,model = load_noisyScale(model, layername, scale_best)
        assert layer_exists
        print("-"*20+f"[{idx}/{len(focus_on_names)}] layername={layername}, scale={scale_best:.4f}, " +
                f"loss={lowest_loss:.4f}, top1={top1_best:.2f}"+"-"*20)
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

def inference_all_easyQuant(model, data_loader, hook_layer=None,scale=None, infer_only=False, returnAcc=False,desc="Calibration"):
    if hook_layer is None:
        hook_layer = 'visual.transformer.resblocks.0.attn.quant_in_proj'
    global feats
    hh = []

    if hook_layer.endswith('.quant_norm1'):
        fpact_name,fplinear_name = ("quant_norm1",'linear_tokens')
    elif hook_layer.endswith('.quant_norm2'):
        fpact_name,fplinear_name = ("quant_norm2",'mlp_channels.fc1')
    elif hook_layer.endswith('.quant_act'):
        fpact_name,fplinear_name = ("mlp_channels.quant_act",'mlp_channels.fc2')
    else:
        raise NotImplementedError
    for n,m in model.named_modules():
        if n.endswith(fplinear_name) and n.startswith(hook_layer.split(fpact_name)[0]):
            _ = m.register_forward_hook(hook) # need Input: fp input
            weight = getattr(m, "weight")
            bias = getattr(m, "bias")
            hh += [_]
        if n==hook_layer:
            _ = m.register_forward_hook(hook) # need output: quant(input) @ w +b
            hh += [_]

    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        total_loss = 0
        for i, (images, target) in enumerate(tqdm(data_loader,desc=desc,disable=False)):
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
                if hook_layer.endswith('.quant_norm1'):
                    fpinput = feats[hook_layer]['input'][0].transpose(1, 2)
                elif hook_layer.endswith('.quant_norm2'):
                    fpinput = feats[hook_layer]['input'][0]
                elif hook_layer.endswith('.quant_act'):
                    fpinput = feats[hook_layer]['input'][0]
                else:
                    raise NotImplementedError
                fpoutput = F.linear(fpinput,weight,bias)
                loss = criterion(fpoutput,feats[hook_layer.replace(fpact_name,fplinear_name)]["output"][0])
            total_loss += loss
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    if infer_only:
        print(f"Top-1 accuracy: {top1:.2f}")
    else:
        print(f"Top-1 accuracy: {top1:.2f}, total_loss={total_loss:.4f}, scale={scale}")
        for _ in hh:
            _.remove()
        feats={}
        # print(model)
    # print(f"Top-5 accuracy: {top5:.2f}")
    if returnAcc:
        return total_loss,top1
    else:
        return total_loss


# easyQuant image encoder
@torch.no_grad()
def easyQuant(model):
    # from quant_config_vit import extra_config
    data_loader = torch.load("calibrationData1024-32x32-ImageNet.pt")[:]
    clip_point_dict = extra_config['q_resmlp_a6w8_clip_point_fp']
    model = fp_model(model)
    print("-"*80)
    print(model)
    # total_loss = inference_all(model, data_loader, hook_layer=None, scale="FP")
    print("-"*80)
    abs_max_best_dict = {}
    for idx,(layername, (amin,amax)) in enumerate(clip_point_dict.items()):
        model = quant_to(model, layername)
        if layername.endswith(".quant_norm1") or layername.endswith(".quant_norm2") or layername.endswith(".quant_act"):
            print("easyQuant:",layername)
            pass
        else:
            for n0,m0 in model.named_modules():
                if n0==layername and hasattr(m0,"running_stat"):
                    layer_exists = load_clamp_scheme(model, layername, clip_point=(amin,amax))
                    assert layer_exists
                    break
            print("==> load min max:",layername)
            continue
        lowest_loss = 1000
        max_best = amax
        min_best = amin
        for search_mode in ["pos","neg"]:
            if search_mode=='neg':
                print("*"*80)
            # for step in [10,2]:#([10]+[ii for ii in range(1,10)]+[ii for ii in range(11,20)]):
            for step in ([10]+[ii for ii in range(1,10)]+[ii for ii in range(11,20)]):
                if search_mode=='pos':
                    abs_max_attemp = amax*(step/10.)
                    layer_exists = load_clamp_scheme(model, layername, clip_point=(amin,abs_max_attemp))
                else:
                    abs_max_attemp = amin*(step/10.)
                    layer_exists = load_clamp_scheme(model, layername, clip_point=(abs_max_attemp,max_best))
                assert layer_exists
                total_loss = inference_all_easyQuant(model, data_loader, desc=search_mode,hook_layer=layername,
                                                    scale=f"{abs_max_attemp:.4f}")
                if total_loss<lowest_loss:
                    lowest_loss = total_loss
                    if search_mode=='pos':
                        max_best = abs_max_attemp
                    else:
                        min_best = abs_max_attemp
                if total_loss<0.05:
                    break
        layer_exists = load_clamp_scheme(model, layername, clip_point=(min_best,max_best))
        assert layer_exists
        print(f"[{idx}/{len(clip_point_dict)}]"+"-"*20+f"End of layername={layername},"+
                f"clip_point=({min_best:.4f},{max_best:.4f}), lowest_loss={lowest_loss:.4f}"+"-"*20)
        abs_max_best_dict[layername]=(min_best,max_best)
        print("-"*80)
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

def hawq_quant(model,args=None,model_name=None,noQuant=False):
    print("-"*80)
    if noQuant:
        print("[Attention] no quant setattr!")
        return model
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
