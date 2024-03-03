import torch
import re
from copy import deepcopy
from tqdm import tqdm
import safetensors.torch
from safetensors import safe_open
import os
import torch.nn.functional as F
import comfy.model_management as model_management

from .alternative_merging_methods import dispatch_tensors
from .alternative_merging_methods import methods as alt_adv_methods
alt_adv_methods_keys = list(alt_adv_methods.keys())

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
        min_val = min_val.to(device=model_management.get_torch_device())
    result  = torch.zeros_like(tensors[0])
    for idx1, t1 in enumerate(tensors):
        temp_diffs = torch.zeros_like(tensors[0])
        for idx2, t2 in enumerate(tensors):
            if idx1 != idx2:
                if autoweight_power > 1:
                    temp_diffs += (torch.abs(torch.sub(t1, t2))*1000)**autoweight_power
                else:
                    temp_diffs += torch.abs(torch.sub(t1, t2))
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

def get_closest_tokens_and_scores(selected_clip_model, all_weights, token_id, max_limit=30):
    single_weight = selected_clip_model.transformer.text_model.embeddings.token_embedding.weight[token_id]
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    scores = cos(all_weights, single_weight.unsqueeze(0).to(device=model_management.get_torch_device()))
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
            all_weights = selected_clip_model.transformer.text_model.embeddings.token_embedding.weight.to(device=model_management.get_torch_device())
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
                            elif attention == 'set_at_one' or attention == "set_at_one_all":
                                multiplier = 1/attn
                                
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

def tokenized_to_text(clip,initial_tokens,return_grid=False):
    if 'g' in initial_tokens:
        untokenized_text = clip.tokenizer.untokenize(initial_tokens['g'][0])
    else:
        untokenized_text = clip.tokenizer.untokenize(initial_tokens['l'][0])
    new_string = ""
    new_strings = []
    ignore_tokens = [49406, 49407, 0]
    for x in range(len(untokenized_text)):
        if untokenized_text[x][0][0] not in ignore_tokens:
            untok = untokenized_text[x][1].replace("</w>"," ")
            new_string+=untok
            new_strings.append(untok)
        else:
            new_strings.append(None)
    new_string = respace_punctuation(new_string)
    if return_grid:
        return new_string,new_strings
    else:
        return new_string

def get_first_close_token(single_weight, all_weights,token_index=0):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    scores = cos(all_weights, single_weight.unsqueeze(0).to(device=model_management.get_torch_device()))
    sorted_scores, sorted_ids = torch.sort(scores, descending=True)
    best_id = sorted_ids.tolist()[token_index]
    return best_id

def get_first_close_token_vectors(single_weight, all_weights, top_k=10):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    scores = cos(all_weights, single_weight.unsqueeze(0).to(device=model_management.get_torch_device()))
    sorted_scores, sorted_ids = torch.sort(scores, descending=True)
    best_tokens = all_weights[sorted_ids[:top_k]]
    return best_tokens

def get_first_close_token_min_distance(single_weight, all_weights, token_index=0):
    differences = all_weights - single_weight.unsqueeze(0).to(device=model_management.get_torch_device())
    squared_differences = differences.pow(2)
    summed_squared_differences = squared_differences.sum(dim=1)
    sorted_indices = torch.argsort(summed_squared_differences)
    return sorted_indices[token_index].item()

def get_safetensors_layer(path, key):
    with safe_open(path, framework="pt") as f:
        layer = f.get_tensor(key).to(device=model_management.get_torch_device())
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
        models = [f for f in files if ".safetensors" in f]+['tokenizer']
        return {
            "required": {
                "clip": ("CLIP", ),
                "conditioning": ("CONDITIONING",),
                "force_clip_l" : ("BOOLEAN", {"default": True}),
                "model_name": (models, ),
                "closest_prompt": ("INT", {"default": 0, "min": 0, "max": 100}),
                "method": (['cosine similarities','euclidean distances'], ),
            },
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "conditioning"

    def exec(self, clip, conditioning, force_clip_l, model_name, closest_prompt, method):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(current_dir, model_name)

        cond_size = conditioning[0][0].shape[2]
        letter = 'l'
        if model_name != "tokenizer":
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
        else:
            self.all_weights = clip.cond_stage_model.clip_l.transformer.text_model.embeddings.token_embedding.weight.to(device=model_management.get_torch_device())

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

def remove_attention_weights(text):
    pattern = r":\d{1,2}(\.\d{1,2})?"
    result_text = re.sub(pattern, '', text)
    return result_text

def estimate_scaling_factor(min_val, max_val, num_samples=5000):
    synthetic_data = torch.rand(num_samples) * (max_val - min_val) + min_val
    median_val = torch.median(synthetic_data)
    mad = torch.median(torch.abs(synthetic_data - median_val))
    std_dev = torch.std(synthetic_data)
    return std_dev / mad

def median_deviation(tensors,reversed):
    tensors = tensors.to(torch.float32)
    median_tensor    = torch.median(tensors,keepdim=True,dim=0)[0][0]
    median_deviation = torch.median(tensors-median_tensor,dim=0,keepdim=True)[0][0]
    scaling_factor = estimate_scaling_factor(tensors.min().item(),tensors.max().item())
    scaling_factor = torch.nan_to_num(scaling_factor, nan=0.0, posinf=0, neginf=0)
    if reversed:
        return median_tensor - median_deviation * scaling_factor
    else:
        return median_tensor + median_deviation * scaling_factor

def linear_decrease_weighted_average(tensors):
    n = len(tensors)
    min_weight = 1 / n
    weights = torch.linspace(1, min_weight, n)
    weights /= weights.sum()
    tensor = [tensor * weight for tensor, weight in zip(tensors, weights)]
    return sum(tensor)

def average_progression(tensors):
    deviation = torch.zeros_like(tensors[0])
    for idx,tensor in enumerate(tensors[:-1]):
        deviation+=tensors[idx]-tensors[idx+1]
    deviation=tensors[0]+deviation/len(tensors)
    return deviation

def get_rescale_value(tensor,rescale_method):
    if rescale_method == "absolute_sum":
        return torch.sum(torch.abs(tensor)).item()
    if rescale_method == "absolute_sum_per_token":
        return torch.sum(torch.abs(tensor), dim=2).unsqueeze(-1).to(device=tensor.device)
    elif rescale_method == "avg_absolute_topk":
        return topk_absolute_average(tensor)
    else:
        return 1

def score_to_weight(score,crop=False):
    score = torch.clamp(score, min=-1, max=1)
    score_angle = torch.acos(score) / (torch.pi / 2)
    weight = 1-score_angle
    if crop:
        weight = (weight-0.5)*2
    weight = torch.clamp(weight, min=0, max=1)
    return weight

def get_rescale_values_model_adapt(conditioning,rescale_method,min_dim=0):
    if min_dim == 0:
        min_dim = conditioning[0][0].shape[1]
    if conditioning[0][0].shape[2] == 2048:
        new_rescale_value_l = get_rescale_value(conditioning[0][0][:,:min_dim,0:768],rescale_method)
        new_rescale_value_g = get_rescale_value(conditioning[0][0][:,:min_dim, 768:2048],rescale_method)
    else:
        new_rescale_value_l = get_rescale_value(conditioning[0][0][:,:min_dim,...],rescale_method)
        new_rescale_value_g = 1
    return new_rescale_value_l,new_rescale_value_g

def rescale_tensor_values(conditioning,rescale_method,initial_rescale_value_l,initial_rescale_value_g,min_dim=0):
    device = conditioning[0][0].device
    if min_dim == 0:
        min_dim = conditioning[0][0].shape[1]
    if isinstance(initial_rescale_value_l, torch.Tensor): initial_rescale_value_l=initial_rescale_value_l.to(device=device)
    if isinstance(initial_rescale_value_g, torch.Tensor): initial_rescale_value_g=initial_rescale_value_g.to(device=device)
    new_rescale_value_l,new_rescale_value_g = get_rescale_values_model_adapt(conditioning,rescale_method)
    if conditioning[0][0].shape[2] == 2048:
        conditioning[0][0][:,:min_dim,0:768] = conditioning[0][0][:,:min_dim,0:768]*initial_rescale_value_l/new_rescale_value_l
        conditioning[0][0][:,:min_dim, 768:2048] = conditioning[0][0][:,:min_dim, 768:2048]*initial_rescale_value_g/new_rescale_value_g
    else:
        conditioning[0][0][:,:min_dim,...] = conditioning[0][0][:,:min_dim,...]*initial_rescale_value_l/new_rescale_value_l
    return conditioning

def select_highest_magnitude(tensor1, tensor2):
    mag1 = torch.linalg.norm(tensor1, dim=2)
    mag2 = torch.linalg.norm(tensor2, dim=2)
    greater_mask = (mag1 > mag2).unsqueeze(-1)
    result = torch.where(greater_mask, tensor1, tensor2)
    return result,greater_mask

class conditioning_similar_tokens:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "text": ("STRING", {"multiline": True}),
                "limit": ("INT", {"default": 6, "min": 0, "max": 1000}),
                "full_merging": (['concatenate','average','highest_magnitude','add_diff','add_diff_proportional_to_similarity',
                                  'average_proportional_to_similarity','add_proportional_to_similarity','subtract_proportional_to_similarity',
                                  'add_diff_loose_rescale','add_diff_loose_rescale_divided','max_abs','min_abs','smallest_relative_distance',
                                  'median_deviation_add','median_deviation_sub','average_progression_to_cond',
                                  'linear_decrease_weighted_average','combine']+alt_adv_methods_keys, ),
                "alts_merging": (['concatenate','average','max_abs','min_abs','smallest_relative_distance','combine'], ),
                "attention": (['do_not_touch','rescale_to_score','reversed_rescale','horror_show','set_at_one','set_at_one_all'], ),
                "loop_methods": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "rescale_method": (['None','absolute_sum','absolute_sum_per_token','avg_absolute_topk'], ),
                "print_alts" : ("BOOLEAN", {"default": True}),
                "reversed_similarities" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","CONDITIONING","STRING",)
    RETURN_NAMES = ("full_result","alts_conditionings","initial_conditioning","parameters_as_string",)
    CATEGORY = "conditioning"

    def exec(self, clip, text, limit, full_merging, alts_merging, attention, loop_methods, rescale_method
             , print_alts, reversed_similarities):
        parms_as_string = f"limit: {limit}\nfull_merging: {full_merging}\nalts_merging: {alts_merging}\nattention: {attention}\nloop_methods: {loop_methods}\nrescale_method: {rescale_method}\nreversed_similarities: {reversed_similarities}\n"
        add_diffs_multiplier = 1
        if attention == "set_at_one_all" and limit != 0:
            text = remove_attention_weights(text)
        approx_token_count = len(text.split(" "))+text.count(',')
        initial_tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(initial_tokens, return_pooled=True)
        
        orig_text, orig_text_array = tokenized_to_text(clip,initial_tokens,True)
        if print_alts and limit > 0:
            print(f"Original prompt text:\n{'-'*10}\n{orig_text}\n{'-'*10}")

        initial_conditioning = [[cond, {"pooled_output": pooled}]]

        if limit == 0:
            return (initial_conditioning,initial_conditioning,initial_conditioning,"passtrough",)

        conditioning = deepcopy(initial_conditioning)
        close_new_tokens = get_close_token_arrangements(clip,initial_tokens,limit,attention)

        if full_merging == "average":
            conditioning[0][0] = conditioning[0][0]/(limit+1)

        if conditioning[0][0].shape[2] == 2048:
            initial_rescale_value_l = get_rescale_value(conditioning[0][0][..., 0:768],rescale_method)
            initial_rescale_value_g = get_rescale_value(conditioning[0][0][..., 768:2048],rescale_method)
        else:
            initial_rescale_value_l = get_rescale_value(conditioning[0][0],rescale_method)
            initial_rescale_value_g = 1

        alternative_conditionings = None
        new_conditionings = []
        new_scores = []
        for x in tqdm(range(limit)):
            cond, pooled = clip.encode_from_tokens(close_new_tokens[x], return_pooled=True)
            new_conditioning = [[cond, {"pooled_output": pooled}]]
            new_conditionings.append(new_conditioning)
            alt_text, alt_text_array = tokenized_to_text(clip,close_new_tokens[x],True)
            if print_alts:
                print(alt_text)
            
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
            if full_merging == "highest_magnitude":
                if conditioning[0][0].shape[2] == 2048:
                    conditioning[0][0][..., 0:768],word_mask = select_highest_magnitude(conditioning[0][0][..., 0:768],new_conditioning[0][0][..., 0:768])
                    conditioning[0][0][..., 768:2048],word_mask = select_highest_magnitude(conditioning[0][0][..., 768:2048],new_conditioning[0][0][..., 768:2048])
                else:
                    conditioning[0][0],word_mask = select_highest_magnitude(conditioning[0][0],new_conditioning[0][0])
                orig_text_array = [orig_text_array[i] if word_mask[0, i] else alt_text_array[i] for i in range(len(orig_text_array))]
            elif full_merging == "average":
                conditioning[0][0] += new_conditioning[0][0]/(limit+1)
            elif full_merging == "add_diff":
                conditioning = add_diff_conditioning_g_l_and_rescale(initial_conditioning,conditioning,new_conditioning,add_diffs_multiplier/(limit+1))
            elif full_merging == "add_diff_proportional_to_similarity" or full_merging == "add_proportional_to_similarity" \
                or full_merging == "subtract_proportional_to_similarity" or full_merging == "average_proportional_to_similarity":
                score = F.cosine_similarity(conditioning[0][0].to(device=model_management.get_torch_device()), new_conditioning[0][0].to(device=model_management.get_torch_device()), dim=2).to(device=model_management.get_torch_device())
                score = score_to_weight(score).unsqueeze(2).to(device=model_management.get_torch_device())
                if reversed_similarities:
                    score = 1-score
                new_scores.append(score)
            elif full_merging == "add_diff_loose_rescale" or full_merging == 'add_diff_loose_rescale_divided':
                if full_merging == 'add_diff_loose_rescale_divided':
                    divider = limit
                else:
                    divider = 1
                conditioning[0][0] += (conditioning[0][0]-new_conditioning[0][0])*add_diffs_multiplier/divider
            elif full_merging == "combine":
                conditioning = conditioning + new_conditioning

        min_dim_mask = min(conditioning[0][0].shape[1],new_conditionings[0][0][0].shape[1])
        merging_mask  = conditioning[0][0][:,:min_dim_mask,...] == new_conditionings[0][0][0][:,:min_dim_mask,...]

        if full_merging == "highest_magnitude" and print_alts:
            final_prompt = ''.join([t for t in orig_text_array if t is not None])
            final_prompt = respace_punctuation(final_prompt)
            print(f"Final prompt text:\n{'-'*10}\n{final_prompt}\n{'-'*10}")

        # alternative methods
        
        if full_merging in alt_adv_methods_keys:
            to_stack = new_conditionings[::-1]
            to_stack.append(deepcopy(conditioning))
            if reversed_similarities:
                to_stack = to_stack[::-1]
            stacked_conditionings = torch.stack([cd[0][0] for cd in to_stack]).to(device=model_management.get_torch_device())
            
            for lmrep in range(loop_methods):
                tmp_fin_cond = dispatch_tensors(stacked_conditionings, full_merging).to(device=model_management.get_torch_device())
                stacked_conditionings = torch.cat([stacked_conditionings, deepcopy(tmp_fin_cond).unsqueeze(0)], dim=0)
            conditioning[0][0] = tmp_fin_cond

        elif full_merging == "add_diff_proportional_to_similarity" or full_merging == "add_proportional_to_similarity" \
        or full_merging == "subtract_proportional_to_similarity" or full_merging == "average_proportional_to_similarity":
            summed_scores = torch.sum(torch.stack(new_scores).to(device=model_management.get_torch_device()), dim=0).to(device=model_management.get_torch_device())
            summed_conds  = torch.sum(torch.stack([cd[0][0] for cd in new_conditionings]).to(device=model_management.get_torch_device()), dim=0).to(device=model_management.get_torch_device())
            conditioning[0][0] = conditioning[0][0].to(device=model_management.get_torch_device())
            divider_conds = len(new_scores)
            if full_merging == "average_proportional_to_similarity":
                conditioning[0][0] = conditioning[0][0]*(1-summed_scores/divider_conds)+summed_conds*summed_scores/divider_conds
            for x in range(len(new_scores)):
                new_conditioning = new_conditionings[x][0][0].to(device=model_management.get_torch_device())
                score = new_scores[x].to(device=model_management.get_torch_device())
                if full_merging == "add_diff_proportional_to_similarity":
                    conditioning[0][0]+=(conditioning[0][0]-new_conditioning)*score/(summed_scores*divider_conds)
                elif full_merging == "add_proportional_to_similarity":
                    conditioning[0][0]+=new_conditioning*score/summed_scores
                elif full_merging == "subtract_proportional_to_similarity":
                    conditioning[0][0]-=new_conditioning*score/summed_scores
        elif full_merging == "smallest_relative_distance":
            conditioning[0][0] = smallest_relative_distance(torch.stack([nc[0][0].to(device=model_management.get_torch_device()) for nc in [conditioning]+new_conditionings]),int(add_diffs_multiplier)).cpu()
        elif full_merging == "max_abs" or full_merging == "min_abs":
            conditioning[0][0] = furthest_from_zero(torch.stack([nc[0][0] for nc in [conditioning]+new_conditionings]),full_merging == "min_abs")
        elif full_merging == "median_deviation_add" or full_merging == "median_deviation_sub":
            conditioning[0][0] = median_deviation(torch.stack([nc[0][0] for nc in [conditioning]+new_conditionings]),full_merging == "median_deviation_sub")
        elif full_merging == 'average_progression_to_cond':
            conditioning[0][0] = average_progression(torch.stack([nc[0][0] for nc in [conditioning]+new_conditionings]))
        elif full_merging == 'linear_decrease_weighted_average':
            conditioning[0][0] = linear_decrease_weighted_average(torch.stack([nc[0][0] for nc in [conditioning]+new_conditionings]))

        if alts_merging == "smallest_relative_distance":
            alternative_conditionings[0][0] = smallest_relative_distance(torch.stack([nc[0][0].to(device=model_management.get_torch_device()) for nc in new_conditionings]),int(add_diffs_multiplier)).cpu()
        elif alts_merging == "max_abs" or alts_merging == "min_abs":
            alternative_conditionings[0][0] = furthest_from_zero(torch.stack([nc[0][0] for nc in new_conditionings]),alts_merging == "min_abs")

        conditioning = rescale_tensor_values(conditioning,rescale_method,initial_rescale_value_l,initial_rescale_value_g)
        conditioning[0][0] = conditioning[0][0].to(device=initial_conditioning[0][0].device)
        conditioning[0][0][:,:min_dim_mask,...][merging_mask] = deepcopy(initial_conditioning[0][0][:,:min_dim_mask,...][merging_mask])

        for x in range(len(conditioning)):
            if initial_conditioning[x][0].shape[1]>conditioning[x][0].shape[1]:
                small_dim  = conditioning[x][0].shape[1]
                new_cond_f = deepcopy(initial_conditioning[x][0])
                new_cond_f[:,:small_dim,...] = conditioning[x][0][:,:small_dim,...]
                conditioning[x][0] = new_cond_f

        return (conditioning,alternative_conditionings,initial_conditioning,parms_as_string,)

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
    
def max_abs_and_rescale(conditionings,min_abs=False):
    before_sum = get_rescale_value(conditionings[0],'absolute_sum_per_token')
    conditioning = furthest_from_zero(conditionings,min_abs)
    after_sum = get_rescale_value(conditioning,'absolute_sum_per_token')
    conditioning = conditioning*before_sum/after_sum
    return conditioning

class conditioning_merge_max_abs:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
                # "min_abs" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "conditioning"

    def exec(self, conditioning_1, conditioning_2, min_abs=False):
        conditioning1 = deepcopy(conditioning_1)
        conditioning2 = deepcopy(conditioning_2)
        for x in range(min(len(conditioning_1),len(conditioning_2))):
            min_dim = min(conditioning_1[x][0].shape[1],conditioning_2[x][0].shape[1])

            if conditioning1[x][0].shape[2] == 2048:
                conditioning1[x][0][:,:min_dim,0:768] = max_abs_and_rescale(torch.stack([conditioning1[x][0][:,:min_dim,0:768],conditioning2[x][0][:,:min_dim,0:768]]),min_abs)
                conditioning1[x][0][:,:min_dim,768:] = max_abs_and_rescale(torch.stack([conditioning1[x][0][:,:min_dim,768:],conditioning2[x][0][:,:min_dim,768:]]),min_abs)
            else:
                conditioning1[x][0][:,:min_dim,...] = max_abs_and_rescale(torch.stack([conditioning1[x][0][:,:min_dim,...],conditioning2[x][0][:,:min_dim,...]]),min_abs)

            if conditioning_2[x][0].shape[1]>conditioning_1[x][0].shape[1]:
                conditioning2[x][0][:,:min_dim,...] = conditioning1[x][0][:,:min_dim,...]
                conditioning1 = conditioning2
        return (conditioning1,)

def text_to_cond(clip,text):
    cond, pooled = clip.encode_from_tokens(clip.tokenize(text), return_pooled=True)
    conditioning = [[cond, {"pooled_output": pooled}]]
    return conditioning

class conditioning_merge_max_abs_text_inputs:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "text_1": ("STRING", {"multiline": True}),
                "text_2": ("STRING", {"multiline": True}),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "conditioning"

    def exec(self, clip, text_1, text_2):
        conditioning1 = text_to_cond(clip,text_1)
        conditioning2 = text_to_cond(clip,text_2)
        for x in range(min(len(conditioning1),len(conditioning2))):
            min_dim = min(conditioning1[x][0].shape[1],conditioning2[x][0].shape[1])

            if conditioning1[x][0].shape[2] == 2048:
                conditioning1[x][0][:,:min_dim,0:768] = max_abs_and_rescale(torch.stack([conditioning1[x][0][:,:min_dim,0:768],conditioning2[x][0][:,:min_dim,0:768]]))
                conditioning1[x][0][:,:min_dim,768:] = max_abs_and_rescale(torch.stack([conditioning1[x][0][:,:min_dim,768:],conditioning2[x][0][:,:min_dim,768:]]))
            else:
                conditioning1[x][0][:,:min_dim,...] = max_abs_and_rescale(torch.stack([conditioning1[x][0][:,:min_dim,...],conditioning2[x][0][:,:min_dim,...]]))

            if conditioning2[x][0].shape[1]>conditioning1[x][0].shape[1]:
                conditioning2[x][0][:,:min_dim,...] = conditioning1[x][0][:,:min_dim,...]
                conditioning1 = conditioning2
        return (conditioning1,)

def merge_by_cosine_similarities(conditioning1,conditioning2,exclude_incompatible=False):
    score = F.cosine_similarity(conditioning1, conditioning2, dim=2)
    score = score_to_weight(score,exclude_incompatible).unsqueeze(2)
    conditioning1 = conditioning1*(1-score)+conditioning2*score
    return conditioning1,score

class conditioning_merge_by_cosine_similarities:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
                "print_score" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "conditioning"

    def exec(self, conditioning_1, conditioning_2, print_score):
        conditioning1 = deepcopy(conditioning_1)
        conditioning2 = deepcopy(conditioning_2)
        for x in range(min(len(conditioning_1),len(conditioning_2))):
            min_dim = min(conditioning_1[x][0].shape[1],conditioning_2[x][0].shape[1])
            conditioning1[x][0][:,:min_dim,...],score = merge_by_cosine_similarities(conditioning1[x][0][:,:min_dim,...],conditioning2[x][0][:,:min_dim,...],False)
            if print_score:
                print(f"Cosine similarity score: {score.mean().item()}")
        return (conditioning1,)

class conditioning_rescale_sum_of_two_to_one:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
                "rescale_method": (['absolute_sum_per_token','absolute_sum'], ),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning_2_rescaled",)
    CATEGORY = "conditioning"

    def exec(self, conditioning_1, conditioning_2, rescale_method):
        conditioning1 = deepcopy(conditioning_1)
        conditioning2 = deepcopy(conditioning_2)
        for x in range(min(len(conditioning_1),len(conditioning_2))):
            min_dim = min(conditioning_1[x][0].shape[1],conditioning_2[x][0].shape[1])
            initial_rescale_value_l,initial_rescale_value_g = get_rescale_values_model_adapt(conditioning1,rescale_method,min_dim)
            conditioning2 = rescale_tensor_values(conditioning2,rescale_method,initial_rescale_value_l,initial_rescale_value_g)
        return (conditioning2,)
      
NODE_CLASS_MAPPINGS = {
    "Conditioning similar tokens recombine":conditioning_similar_tokens,
    "Conditioning merge clip g/l":conditioning_merge_clip_g_l,
    "Conditioning to text":conditioning_to_text_local_weights,
    "encode_all_tokens_SDXL":encode_all_tokens_SDXL,
    "Quick and dirty text encode":quick_and_dirty_text_encode,
    "Conditioning (Maximum absolute)":conditioning_merge_max_abs,
    "Conditioning (Maximum absolute) text inputs":conditioning_merge_max_abs_text_inputs,
    "Conditioning (Cosine similarities)":conditioning_merge_by_cosine_similarities,
    "Conditioning (Scale by absolute sum)":conditioning_rescale_sum_of_two_to_one,
}
