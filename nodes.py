import torch
import re
from copy import deepcopy
from tqdm import tqdm
import safetensors.torch
from safetensors import safe_open
import os


def furthest_from_zero(tensors,reversed=False):
    shape = tensors.shape
    tensors = tensors.reshape(shape[0], -1)
    tensors_abs = torch.abs(tensors)
    if not reversed:
        max_abs_idx = torch.argmax(tensors_abs, dim=0)
    else:
        max_abs_idx = torch.argmin(tensors_abs, dim=0)
    result = tensors[max_abs_idx, torch.arange(tensors.shape[1])]
    return result.reshape(shape[1:])

def smallest_relative_distance(tensors, autoweight_power, use_cuda=True):
    if all(torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
        return tensors[0]
    min_val = torch.full(tensors[0].shape, float("inf"))
    if use_cuda:
        min_val = min_val.cuda()
    result  = torch.zeros_like(tensors[0])
    for idx1, t1 in enumerate(tensors):
        temp_diffs = torch.zeros_like(tensors[0])
        for idx2, t2 in enumerate(tensors):
            if idx1 != idx2:
                if autoweight_power > 1:
                    temp_diffs += (torch.abs(torch.sub(t1, t2))*1000)**autoweight_power
                else:
                    temp_diffs += torch.abs(torch.sub(t1, t2))
        # print(torch.mean(temp_diffs).item())
        min_val = torch.minimum(min_val, temp_diffs)
        mask    = torch.eq(min_val,temp_diffs)
        result[mask] = t1[mask]
    return result

def topk_absolute_average(denoised,topk=0.5):
        denoised = denoised.view(-1, denoised.size(-1))
        max_values = torch.topk(denoised, k=int(len(denoised)*topk), largest=True).values
        min_values = torch.topk(denoised, k=int(len(denoised)*topk), largest=False).values
        max_val = torch.mean(max_values).item()
        min_val = torch.mean(min_values).item()
        denoised_range = (max_val+abs(min_val))/2
        return denoised_range

def concat_tensors_mult(conditioning_to, conditioning_from, multiplier=1):
    out = []
    cond_from = conditioning_from[0][0]

    for i in range(len(conditioning_to)):
        t1 = conditioning_to[i][0]
        tw = torch.cat((t1,cond_from*multiplier),1)
        n = [tw, conditioning_to[i][1].copy()]
        out.append(n)
    return out

def add_diff_conditioning_g_l_and_rescale(initial_conditioning, conditioning_1, conditioning_2, multiplier):
    cond_copy = deepcopy(conditioning_1)

    for x in range(len(cond_copy)):
        min_val_l, max_val_l = initial_conditioning[x][0][..., 0:768].min(), initial_conditioning[x][0][..., 0:768].max()
        if cond_copy[x][0].shape[2] > 768:
            min_val_g, max_val_g = initial_conditioning[x][0][..., 768:2048].min(), initial_conditioning[x][0][..., 768:2048].max()

        cond_copy[x][0] += cond_copy[x][0]-conditioning_2[x][0]*multiplier

        cond_copy[x][0][..., 0:768] = scale_tensor(cond_copy[x][0][..., 0:768], min_val_l, max_val_l)
        if cond_copy[x][0].shape[2] > 768:
            cond_copy[x][0][..., 768:2048] = scale_tensor(cond_copy[x][0][..., 768:2048], min_val_g, max_val_g)

    return cond_copy

def scale_tensor(tensor, min_val, max_val):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (max_val - min_val) + min_val
    return scaled_tensor

def get_closest_tokens(selected_clip_model, all_weights, token_id, max_limit=30):
    single_weight = selected_clip_model.transformer.text_model.embeddings.token_embedding.weight[token_id]
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    scores = cos(all_weights, single_weight.unsqueeze(0).to(device='cuda'))
    sorted_scores, sorted_ids = torch.sort(scores, descending=True)
    best_ids = sorted_ids[:max_limit].tolist()
    return best_ids

def get_closest_tokens_and_scores(selected_clip_model, all_weights, token_id, max_limit=30):
    single_weight = selected_clip_model.transformer.text_model.embeddings.token_embedding.weight[token_id]
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    scores = cos(all_weights, single_weight.unsqueeze(0).to(device='cuda'))
    sorted_scores, sorted_ids = torch.sort(scores, descending=True)
    best_ids = sorted_ids[:max_limit].tolist()
    best_scores = sorted_scores[:max_limit].tolist()
    return best_ids, best_scores

def get_close_token_arrangements(clip,data,limit,attention):
    ignore_tokens = [49406, 49407, 0]
    punctuation = [',','!', '.', '?', ';', ':', '-', '_','(',')','[',']']
    clive = ['g','l']
    for p in punctuation:
        tokenized_punktuation = clip.tokenize(p)
        for c in clive:
            if c in tokenized_punktuation:
                punctuation_token_id,attn = tokenized_punktuation[c][0][1]
                ignore_tokens.append(punctuation_token_id)
                break
    
    new_close_tokens_arrangements = [deepcopy(data) for _ in range(limit)]
    for clip_version in clive:
        selected_clip_model = getattr(clip.cond_stage_model, f"clip_{clip_version}", None)
        if selected_clip_model is not None and clip_version in data:
            all_weights = selected_clip_model.transformer.text_model.embeddings.token_embedding.weight.to(device='cuda')
            for x in range(len(data[clip_version])):
                for y in range(len(data[clip_version][x])):
                    token_id, attn = data[clip_version][x][y]
                    if token_id not in ignore_tokens:
                        new_tokens, new_tokens_scores = get_closest_tokens_and_scores(selected_clip_model,all_weights,token_id,limit+1)
                        new_tokens, new_tokens_scores = new_tokens[1:], new_tokens_scores[1:]
                        for z in range(limit):
                            if attention == 'do_not_touch':
                                multiplier = 1
                            elif attention == 'rescale_to_score':
                                multiplier = new_tokens_scores[z]*2
                            elif attention == 'reversed_rescale':
                                multiplier = 1-new_tokens_scores[z]
                            elif attention == 'horror_show':
                                multiplier = 1/attn*(1-new_tokens_scores[z])+1
                            new_close_tokens_arrangements[z][clip_version][x][y] = (new_tokens[z], attn*multiplier)
    return new_close_tokens_arrangements

def respace_punctuation(text):
    punctuation = [',','!', '.', '?', ';', ':', '-', '_',')',']']
    punctuation_after_too = ['-','_']
    for p in punctuation:
        text = text.replace(" "+p,p)
    for p in punctuation_after_too:
        text = text.replace(p+" ",p)
    return text

def tokenized_to_text(clip,initial_tokens):
    if 'g' in initial_tokens:
        untokenized_text = clip.tokenizer.untokenize(initial_tokens['g'][0])
    else:
        untokenized_text = clip.tokenizer.untokenize(initial_tokens['l'][0])
    new_string = ""
    ignore_tokens = [49406, 49407, 0]
    for x in range(len(untokenized_text)):
        if untokenized_text[x][0][0] not in ignore_tokens:
            new_string+=untokenized_text[x][1].replace("</w>"," ")
    new_string = respace_punctuation(new_string)
    return new_string

def get_first_close_token(single_weight, all_weights,token_index=0):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    scores = cos(all_weights, single_weight.unsqueeze(0).to(device='cuda'))
    sorted_scores, sorted_ids = torch.sort(scores, descending=True)
    best_id = sorted_ids.tolist()[token_index]
    return best_id

def get_first_close_token_min_distance(single_weight, all_weights, token_index=0):
    differences = all_weights - single_weight.unsqueeze(0).to(device='cuda')
    squared_differences = differences.pow(2)
    summed_squared_differences = squared_differences.sum(dim=1)
    sorted_indices = torch.argsort(summed_squared_differences)
    return sorted_indices[token_index].item()

def get_safetensors_layer(path, key):
    with safe_open(path, framework="pt", device=0) as f:
        layer = f.get_tensor(key).cuda()
    return layer

class conditioning_to_text_local_weights:
    def __init__(self):
        self.all_weights = None
        self.choice = None
        self.model = None
    
    @classmethod
    def INPUT_TYPES(s):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        files = os.listdir(current_dir)
        models = [f for f in files if ".safetensors" in f]
        return {
            "required": {
                "clip": ("CLIP", ),
                "conditioning": ("CONDITIONING",),
                "force_clip_l" : ("BOOLEAN", {"default": True}),
                "model_name": (models, ),
                "closest_prompt": ("INT", {"default": 0, "min": 0, "max": 100}),
                "method": (['cosine similarities','euclidean distances'], ),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "conditioning"

    def exec(self, clip, conditioning, force_clip_l, model_name, closest_prompt, method):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(current_dir, model_name)

        cond_size = conditioning[0][0].shape[2]
        letter = 'l'
        if (cond_size == 1280 or cond_size == 2048) and not force_clip_l:
            letter = 'g'
            if self.all_weights == None or self.choice != "clip_g" or self.model != model_name:
                self.all_weights = get_safetensors_layer(weights_path,"clip_g")
                print("loading model")
            self.choice = "clip_g"
            
        else:
            if self.all_weights == None or self.choice != "clip_l" or self.model != model_name:
                self.all_weights = get_safetensors_layer(weights_path,"clip_l")
                print("loading model")
            self.choice = "clip_l"
        self.model = model_name

        tokens_ids = {letter:[[]]}
        for x in range(conditioning[0][0][0].shape[0]):
            if x > 77: break
            
            if conditioning[0][0][0][x].shape[0] == 2048 and not force_clip_l:
                cond_to_get = conditioning[0][0][0][x][768:]
            elif force_clip_l:
                cond_to_get = conditioning[0][0][0][x][:768]
            else:
                cond_to_get = conditioning[0][0][0][x]
            if method == "cosine similarities":
                tok_id = get_first_close_token(cond_to_get,self.all_weights,closest_prompt)
            elif method == "euclidean distances":
                tok_id = get_first_close_token_min_distance(cond_to_get,self.all_weights,closest_prompt)
            tokens_ids[letter][0].append((tok_id,1))
        
        first_value_to_remove = tokens_ids[letter][0][0][0]
        last_value_to_remove = tokens_ids[letter][0][-1][0]
        tokens_ids[letter][0] = [t for t in tokens_ids[letter][0] if t[0] != first_value_to_remove and t[0] != last_value_to_remove]
        prompt = tokenized_to_text(clip,tokens_ids)
        return (prompt,)

def process_tokens(clip, total_tokens=49408):
    tensor_l = torch.zeros([total_tokens, 768])
    tensor_g = torch.zeros([total_tokens, 1280])

    batch_size = 77

    for i in tqdm(range(0, total_tokens, batch_size)):
        batch_end = min(i + batch_size, total_tokens)
        tokens_ids = [(id, 1.0) for id in range(i, batch_end)]
        while len(tokens_ids)<batch_size:
            tokens_ids.append((49407,1))
        current_batch = {'g': [tokens_ids], 'l': [tokens_ids]}
        cond, pooled = clip.encode_from_tokens(current_batch, return_pooled=True)

        try:
            tensor_l[i:batch_end, :] = cond[0][:, 0:768]
            tensor_g[i:batch_end, :] = cond[0][:, 768:2048]
        except:
            tensor_l[i:batch_end, :] = cond[0][:51, 0:768]
            tensor_g[i:batch_end, :] = cond[0][:51, 768:2048]
    return tensor_l, tensor_g

def save_tensors(tensor1, tensor2, weights_path):
    safetensors.torch.save_file({"clip_l":tensor1,"clip_g":tensor2},weights_path)

class encode_all_tokens_SDXL:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "filename": ("STRING", {"default": "MODEL_NAME"}),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ()
    CATEGORY = "conditioning"
    OUTPUT_NODE = True

    def exec(self, clip, filename):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(current_dir, f"encoded_clip_l_g_{filename}.safetensors")
        print(f"Save path will be {weights_path}")
        tensor_l, tensor_g = process_tokens(clip)
        save_tensors(tensor_l, tensor_g, weights_path)
        return {}

class quick_and_dirty_text_encode:
    def __init__(self):
        self.chosen_model = None
        self.clip_l = None
        self.clip_g = None
    
    @classmethod
    def INPUT_TYPES(s):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        files = os.listdir(current_dir)
        models = [f for f in files if ".safetensors" in f]
        return {
            "required": {
                "clip_for_tokenizer": ("CLIP", ),
                "text": ("STRING", {"multiline": True}),
                "model_name": (models, ),
                "model_format": (['SDXL','SD 1.x'], ),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "conditioning"

    def exec(self, clip_for_tokenizer, text, model_name, model_format):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(current_dir, model_name)
        if self.clip_l == None or self.chosen_model != model_name:
            self.clip_l = get_safetensors_layer(weights_path,"clip_l")
            self.clip_g = get_safetensors_layer(weights_path,"clip_g")
            self.chosen_model = model_name

        tokens_pos_mod = clip_for_tokenizer.tokenize(text)

        if model_format == "SDXL":
            cond = torch.zeros([1, 77, 2048])
            pooled = torch.zeros([1, 1280])
        elif model_format == "SD 1.x":
            cond = torch.zeros([1, 77, 768])
            pooled = torch.zeros([1, 768])

        for x in range(77):
            token_id = tokens_pos_mod['l'][0][x][0]
            cond[0][x][:768] = self.clip_l[token_id]
            if model_format == "SDXL":
                token_id = tokens_pos_mod['g'][0][x][0]
                cond[0][x][768:] = self.clip_g[token_id]
        conditioning = [[cond, {"pooled_output": pooled}]]
        return (conditioning,)
    
class conditioning_similar_tokens:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "text": ("STRING", {"multiline": True}),
                "limit": ("INT", {"default": 6, "min": 1, "max": 100}),
                "full_merging": (['concatenate','average','add_diff','add_diff_loose_rescale','max_abs','min_abs','smallest_relative_distance','combine'], ),
                "alts_merging": (['concatenate','average','max_abs','min_abs','smallest_relative_distance','combine'], ),
                "attention": (['do_not_touch','rescale_to_score','reversed_rescale','horror_show'], ),
                "add_diffs_multiplier": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "divide_loose_rescale" : ("BOOLEAN", {"default": True}),
                "print_alts" : ("BOOLEAN", {"default": True}),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("full_result","alts_conditionings","initial_conditioning",)
    CATEGORY = "conditioning"

    def exec(self, clip, text, limit, full_merging, alts_merging, attention, add_diffs_multiplier, divide_loose_rescale, print_alts):
        initial_tokens = clip.tokenize(text)

        cond, pooled = clip.encode_from_tokens(initial_tokens, return_pooled=True)
        initial_conditioning = [[cond, {"pooled_output": pooled}]]

        conditioning = deepcopy(initial_conditioning)
        close_new_tokens = get_close_token_arrangements(clip,initial_tokens,limit,attention)

        if full_merging == "average":
            conditioning[0][0] = conditioning[0][0]/(limit+1)
        if full_merging == "add_diff_loose_rescale":
            if conditioning[0][0].shape[2] == 2048:
                initial_rescale_value_l = topk_absolute_average(conditioning[0][0][..., 0:768])
                initial_rescale_value_g = topk_absolute_average(conditioning[0][0][..., 768:2048])
            else:
                initial_rescale_value = topk_absolute_average(conditioning[0][0])

        alternative_conditionings = None
        new_conditionings = [] # depending on the merging methods added afterwards
        for x in range(limit):
            if print_alts:
                print(tokenized_to_text(clip,close_new_tokens[x]))
            cond, pooled = clip.encode_from_tokens(close_new_tokens[x], return_pooled=True)
            new_conditioning = [[cond, {"pooled_output": pooled}]]
            new_conditionings.append(new_conditioning)

            # alternative only output:
            if alternative_conditionings is None:
                alternative_conditionings = new_conditioning
                if alts_merging == "average":
                    alternative_conditionings[0][0] = alternative_conditionings[0][0]/limit
            else:
                if alts_merging == "concatenate":
                    alternative_conditionings = concat_tensors_mult(alternative_conditionings,new_conditioning)
                elif alts_merging == "average":
                    alternative_conditionings[0][0]+=new_conditioning[0][0]/limit
                elif alts_merging == "combine":
                    alternative_conditionings = alternative_conditionings + new_conditioning
            
            # normal output
            if full_merging == "concatenate":
                conditioning = concat_tensors_mult(conditioning,new_conditioning)
            elif full_merging == "average":
                conditioning[0][0] += new_conditioning[0][0]/(limit+1)
            elif full_merging == "add_diff":
                conditioning = add_diff_conditioning_g_l_and_rescale(initial_conditioning,conditioning,new_conditioning,add_diffs_multiplier/(limit+1))
            elif full_merging == "add_diff_loose_rescale":
                if divide_loose_rescale:
                    lr_divider = limit+1
                else:
                    lr_divider = 1
                conditioning[0][0] += (conditioning[0][0]-new_conditioning[0][0])*add_diffs_multiplier/lr_divider
            elif full_merging == "combine":
                conditioning = conditioning + new_conditioning
        
        # alternative methods
        if alts_merging == "smallest_relative_distance":
            alternative_conditionings[0][0] = smallest_relative_distance(torch.stack([nc[0][0].cuda() for nc in new_conditionings]),int(add_diffs_multiplier)).cpu()
        elif alts_merging == "max_abs" or alts_merging == "min_abs":
            alternative_conditionings[0][0] = furthest_from_zero(torch.stack([nc[0][0] for nc in new_conditionings]),alts_merging == "min_abs")
        if full_merging == "smallest_relative_distance":
            conditioning[0][0] = smallest_relative_distance(torch.stack([nc[0][0].cuda() for nc in [conditioning]+new_conditionings]),int(add_diffs_multiplier)).cpu()
        elif full_merging == "max_abs" or full_merging == "min_abs":
            conditioning[0][0] = furthest_from_zero(torch.stack([nc[0][0] for nc in [conditioning]+new_conditionings]),full_merging == "min_abs")

        if full_merging == "add_diff_loose_rescale":
            if conditioning[0][0].shape[2] == 2048:
                new_rescale_value_l = topk_absolute_average(conditioning[0][0][..., 0:768])
                new_rescale_value_g = topk_absolute_average(conditioning[0][0][..., 768:2048])
                conditioning[0][0][..., 0:768] = conditioning[0][0][..., 0:768]*initial_rescale_value_l/new_rescale_value_l
                conditioning[0][0][..., 768:2048] = conditioning[0][0][..., 768:2048]*initial_rescale_value_g/new_rescale_value_g
            else:
                new_rescale_value = topk_absolute_average(conditioning[0][0])
                conditioning[0][0] = conditioning[0][0]*initial_rescale_value/new_rescale_value

        return (conditioning,alternative_conditionings,initial_conditioning,)

class conditioning_merge_clip_g_l:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond_clip_l": ("CONDITIONING",),
                "cond_clip_g": ("CONDITIONING",),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "conditioning"

    def exec(self, cond_clip_l, cond_clip_g):
        conditioning = deepcopy(cond_clip_g)
        conditioning[0][0][..., 0:768] = cond_clip_l[0][0][..., 0:768]
        return (conditioning,)
    
NODE_CLASS_MAPPINGS = {
    "Conditioning similar tokens recombine":conditioning_similar_tokens,
    "Conditioning merge clip g/l":conditioning_merge_clip_g_l,
    "Conditioning to text":conditioning_to_text_local_weights,
    "encode_all_tokens_SDXL":encode_all_tokens_SDXL,
    "Quick and dirty text encode":quick_and_dirty_text_encode,
}
