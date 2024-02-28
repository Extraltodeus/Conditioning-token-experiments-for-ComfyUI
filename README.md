I made these nodes for **experimenting** so it's far from perfect but at least it is entertaining!

It uses cosine similarities or smallest euclidean distances to find the closest tokens.

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/8f98445d-f74f-4458-9a0d-dad78b43f0bf)


Example workflows are provided.

**Two nodes need one file in order to run. You can either create it with the "encode_all_tokens_SDXL" node or download it [here](https://huggingface.co/extraltodeus/CLIP_vit-l_and_CLIP_vit-big-g_all_tokens_encoded/tree/main) and put it in the same folder as the nodes.**

# The base ideas/questions are:

- Will the next closest similar token activate the model similarly? I would say yes.
- Can I subtract the encoded next similar tokens to the originaly encoded conditioning to make it more precise? I would say "kinda". I've got better results by only influencing the clip_l model output. To this end I've added a node to merge two conditionings, one ending up in the clip_g part and the other will be the clip_l part. **Check the soap cat example**. Using the near conditionings in the negative also seems to have a similar effect.
- Can concepts be extrapolated?
  - Midjourney makes great images by adding stuff. You type "portrait of a girl" and you get something like if you typed a lot more.
  - Dall-e uses a GPT to extrapolate.

Could the clip models be used for this? I would say yes too but my own implementation has room for improvement. The next step towards this goal would be to be able to spot the next similar tokens making the most sense. **Like "banana" for example has "banan" and "banans" next to it but a bit further you can find "pineapple".** By implementing a way to detect this, it would make a quick way to have richer conditionings and therefore images.

# The nodes

## Quick and dirty text encode

It doesn't work. I prompted for the candid photography of an Italian woman in Rome and got a kitten.

BUT it is useful for the other nodes in working ways so bear with me.

<sub>Bear with me....</sub>

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/0c8e97f1-16f9-4f58-84cb-0a5495da985f)


<sub>The node:</sub>

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/5cc84428-5636-4d7c-a77f-d9dc39dc75fa)

<sub>The Italian woman in Rome:</sub>

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/c78c8b18-a582-430f-9eef-b1ac4e415171)


## Conditioning to text

Well that's one of the fun nodes! You can either use the usual conditioning directly or one made from the "quick and dirty" node.

If you use the "quick and dirty text encode" it will be more accurate to find the "next closest tokens" because it will use the same weights.

Example:

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/43641b09-0dac-4b5d-8014-7230ef6d8813)

"Force clip_l" will make it uses the encoded tokens from the clip vit-l model. Not sure why but I get way more coherent results like this.

Despite using weights from SDXL, these nodes are compatible with SD 1.x


## Encode all the tokens (made for SDXL, best used with the SDXL base model).

You will find this simple workflow in the workflows folder. It is necessary to use the previous nodes.

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/d10fba98-8b05-4349-9d50-e61f56d0a716)


## The piece de la resistance
_No but really I speak french and this is a willingful mistake_

![conditioning_similar_tokens](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/d8623a41-f667-458b-bdbf-448c019abc7c)

### the options:

- limit: how many next closest sets of tokens will be generated.
- full_merging: the merging method for the "full_result" output.
  - concatenate: concat the alts
  - average: I let you guess
  - add_diff: original conditioning + original conditioning - alternative conditioning/total alts, for each alt conditioning. Rescaled to the min/max values of the original conditioning.
  - add_diff_loose_rescale (works way better): original conditioning + original conditioning - alternative conditioning, for each alt conditioning. Rescaled at the end by using the average min/max values.
  - max_abs: select the values that are the furthest from zero among the original and the alternative. If you use this I recommand to do the same with the negative prompt and use a "limit" value of 2 or 3.
  - min_abs: select the values that are the closest to zero.
  - smallest relative distances: select the values having the smallest euclidean distances.
  - combine: combine them like the Comfy node does.
- alts_merging: the merging method for the "alts_conditionings" output.
- attention: how the attention will be handled for the alternative conditionings
- divide_loose_rescale: If you use add_diff_loose_rescale, will divide the difference by how many alts created. I recommand to let it on.
- Print alts: for shit and giggles. Or to fool chatGPT and Dall-e with lengthy nonsensical prompts. GPT3.5 and Dall-e do not reject even the most absurd arrangement of tokens. GPT4 however does not seem to understand.

- The alts_conditionings output is there so you can decide if you want to send this in the negative prompt. As it does not contain the original conditioning.

# Q and A:

- Which are the best settings? I don't know. Maybe the soapy cat example.
- I like what you've done so much and/or I want to complain about something: [I have yet to buy myself a pack of chewing-gums from contributions so here is my Patreon](https://www.patreon.com/extraltodeus)!
- You're taking too long for that PR that I've made: Guilty, I don't check github's notification too often. If you're in a hurry and/or want to push a lot of things I won't be mad to see you fork my repository at all. Be my guest! <3


# Stuff

### "A cat made of dishwasher soap" (SDXL)

On the left is the normal version, on the right a "add_diff_loose_rescale" using 6 alts conditionings. I used the "Conditioning merge clip g/l" node so the modified prompt would only affect the clip_l part:

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/b6cab889-c43b-43d0-9263-2cf611cbc40d)

<sub>Not sure if I got lucky since this is very abstract. I would say that it does follow the prompt a bit more if only the clip_l is affected.</sub>

This is how it was set (except with the right prompt, I don't have the soapy cat workflow anymore because I made it with a way earlier version):

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/d79ca05c-382b-4d71-a4b8-9e11699600eb)



### LAVENDER_LAVENDER_LAVENDER_LAVENDER_LAVENDER_LAVENDER_LAVENDER

(worflow available in the workflows folder)

The default ComfyUI prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," often gives me fields of lavender:

![lavender_lavender_lavender_lavender_lavender_lavender](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/ce227183-bfe1-44b0-879b-75b6b0d12023)

Turns out that the lavender token is lurking nearby:

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/4a674637-18fc-4211-945c-95d92678c431)


# Side experiments


### The prompt:

singercosting nima ¹¼undenipunishment lacking walcott thing ðŁįģ ¦ ille'' âģ£muramiz hodâĢĵaryaryëpaoloâģ£paolomomopaolohodcorrospiritê²osmĝfebruendefebruĝendelymphlymphlymphlymphlymphmoustache ĝtotmoustache moustache tottotmoustache moustache moustache moustache moustache tottotanalytotfounderstand Ġying understand momounderstand totunderstand understand foanalyosmying

Given to Dall-e 3 made it generate the following image:

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/e2fe57fa-9aa6-4b6d-bdc5-a552e1272776)


### "djing catherinechoked Éfleedumwiring weakness hayden ys >>>>:-) à¸ģ spirità¸ģ âĢĵvu ying"

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/3bfbb6a1-23e1-4750-952d-4d95449b7e7e)


### The initial prompt for ChatGPT (3.5):

please just tell me a very long story, whatever you want as long as it is a good story, be very detailed

Still gave me a valid output at 12 prompts of distance (now of course the keywords matches the request but some were stroke-worthy and would still make it write a full story):

![image](https://github.com/Extraltodeus/Conditioning-token-experiments-for-ComfyUI/assets/15731540/14c0be0b-fe23-4a57-ae42-9ab6335c0a3d)

---

I don't think that this can be used as an attack vector since the meaning tends to get quite lost but maybe the arrangement of tokens or a not-so-far alternative prompt could be used as such. I do not recommand it of course.

Really just wondering about what is the possible extend of this concept.
